import torch.nn as nn
import timm


class ConvNextEncoder(nn.Module):
    def __init__(self, pretrained=True, backbone="convnext_small", out_index=1):
        super().__init__()
        self.backbone_name = backbone
        self.out_index = out_index
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[out_index],
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        feature_info = self.backbone.feature_info
        self.out_channels = feature_info.channels()[0]

    def forward(self, x):
        return self.backbone(x)[0]
