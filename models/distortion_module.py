import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalGeometryDistortionModule(nn.Module):
    """
    SG-DAF: Spherical Geometry-aware Distortion Adaptive Fusion

    作用：
    1. 显式构建 ERP 全景图的球面几何先验
    2. 根据纬度位置估计畸变强度
    3. 用可学习门控融合原始特征与畸变补偿特征
    4. 通过水平 circular padding 缓解全景图左右边界断裂问题
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()

        hidden_channels = max(in_channels // reduction, 16)

        # 几何先验编码：输入包括 lon, lat, cos(lat), distortion
        self.geo_encoder = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 局部畸变补偿分支
        self.local_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 水平方向环绕上下文，用于处理 ERP 左右边界不连续
        self.horizontal_context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 0), groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # 自适应融合门控
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.out_norm = nn.BatchNorm2d(in_channels)

    def build_spherical_prior(self, h, w, device):
        """
        构造球面几何先验：
        lon: 经度 [-pi, pi]
        lat: 纬度 [-pi/2, pi/2]
        area_weight: cos(lat)，表示 ERP 不同纬度对应球面面积变化
        distortion: 1 - cos(lat)，越靠近上下边缘畸变越大
        """
        y = torch.linspace(-math.pi / 2, math.pi / 2, h, device=device)
        x = torch.linspace(-math.pi, math.pi, w, device=device)

        lat, lon = torch.meshgrid(y, x, indexing='ij')

        lat_norm = lat / (math.pi / 2)
        lon_norm = lon / math.pi

        area_weight = torch.cos(lat).clamp(min=0.0)
        distortion = 1.0 - area_weight

        prior = torch.stack(
            [lon_norm, lat_norm, area_weight, distortion],
            dim=0
        ).unsqueeze(0)

        return prior

    def circular_horizontal_conv(self, x):
        """
        ERP 图像左右是连续的，因此水平方向使用 circular padding。
        """
        x_pad = F.pad(x, pad=(3, 3, 0, 0), mode='circular')
        return self.horizontal_context(x_pad)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        device = x.device

        prior = self.build_spherical_prior(h, w, device)
        prior = prior.repeat(b, 1, 1, 1)

        geo_weight = self.geo_encoder(prior)

        # 畸变感知加权
        geo_feat = x * geo_weight

        # 局部细化
        local_feat = self.local_refine(geo_feat)

        # 水平环绕上下文
        pano_feat = self.circular_horizontal_conv(geo_feat)

        # 融合两类畸变补偿特征
        compensated_feat = local_feat + pano_feat

        gate = self.fusion_gate(torch.cat([x, compensated_feat], dim=1))

        out = x * (1.0 - gate) + compensated_feat * gate
        out = self.out_norm(out)

        return out