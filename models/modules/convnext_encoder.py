import torch, torch.nn as nn, timm
 
class ConvNextEncoder(nn.Module):
    """
    ConvNext-Small, first 2 stages only.
    Input:  [B, 3, H, W]
    Output: [B, 192, H/8, W/8]  (implicit space)
    FROZEN — never update these weights.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_small", pretrained=pretrained,
            features_only=True, out_indices=[1])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.out_channels = 192
 
    def forward(self, x):
        return self.backbone(x)[0]  # [B, 192, H/8, W/8]
