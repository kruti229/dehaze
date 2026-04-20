import torch, torch.nn as nn
 
class DimensionalDecompressionUnit(nn.Module):
    """
    Expands clean latent [B,16,H,W] → implicit [B,192,H,W].
    CEE from encoding step guides what was lost during compression.
    Progressive: 16→32→64→96→192 channels.
    """
    def __init__(self, in_ch=16, out_ch=192, cee_ch=64):
        super().__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.GroupNorm(4, 32), nn.GELU())
        self.cee_gate = nn.Sequential(
            nn.Conv2d(cee_ch, 32, 1), nn.Sigmoid())
        self.s2 = nn.Sequential(
            nn.Conv2d(32+cee_ch, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU())
        self.s3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GroupNorm(8, 96), nn.GELU())
        self.s4 = nn.Sequential(
            nn.Conv2d(96, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.GELU())
 
    def forward(self, z_clean, cee):
        x = self.s1(z_clean)
        x = x * self.cee_gate(cee)
        x = self.s2(torch.cat([x, cee], 1))
        x = self.s3(x)
        return self.s4(x)   # [B, 192, H, W]
