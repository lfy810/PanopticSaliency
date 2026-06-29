import timm
import torch.nn as nn


class TransformerBackbone(nn.Module):
    def __init__(self, img_size=(512, 1024)):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,  # 禁用自动下载，使用本地预训练权重
            features_only=True,
            img_size=img_size
        )

    def forward(self, x):
        return self.backbone(x)
