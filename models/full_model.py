import torch.nn as nn

from models.vit_backbone import TransformerBackbone
from models.distortion_module import SphericalGeometryDistortionModule
from models.saliency_head import SimpleSaliencyHead


class DistortionAwareSaliencyModel(nn.Module):
    def __init__(self, img_size=(512, 1024)):
        super().__init__()

        self.backbone = TransformerBackbone(img_size=img_size)
        self.distortion_module = SphericalGeometryDistortionModule(in_channels=1024)
        self.saliency_head = SimpleSaliencyHead(in_channels=1024)

        self.img_size = img_size

    def forward(self, x):
        feats = self.backbone(x)
        top_feat = feats[-1]

        # swin 输出是 [B, H, W, C]，先转成 [B, C, H, W]
        if top_feat.dim() == 4 and top_feat.shape[-1] == 1024:
            top_feat = top_feat.permute(0, 3, 1, 2).contiguous()

        top_feat = self.distortion_module(top_feat)

        saliency_map = self.saliency_head(top_feat, out_size=self.img_size)

        return saliency_map