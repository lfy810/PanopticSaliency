import torch.nn as nn
import torch.nn.functional as F


class SimpleSaliencyHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x, out_size):
        """
        x: [B, C, h, w]
        out_size: (H, W)
        """
        sal = self.conv(x)  # [B, 1, h, w]
        sal = F.interpolate(sal, size=out_size, mode='bilinear', align_corners=False)
        return sal
