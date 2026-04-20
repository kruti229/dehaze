import torch, torch.nn as nn, torch.nn.functional as F
 
class ContextEmbeddingElement(nn.Module):
    def __init__(self, in_ch=192, cee_ch=64):
        super().__init__()
        self.fine = nn.Sequential(
            nn.Conv2d(in_ch, cee_ch//2, 1), nn.GELU())
        self.coarse = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//4, 8, stride=8, groups=in_ch//4),
            nn.Conv2d(in_ch//4, cee_ch//2, 1), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Conv2d(cee_ch//2, cee_ch, 1),
            nn.GroupNorm(8, cee_ch), nn.GELU())
 
    def forward(self, x):
        B, C, H, W = x.shape
        f = self.fine(x)                               # [B,32,H,W]
        c = F.interpolate(self.coarse(x),
            size=(H, W), mode="bilinear", align_corners=False)
        return self.fusion(f + c)                      # [B,64,H,W]
 
 
class DimensionalCompressionUnit(nn.Module):
    def __init__(self, in_ch=192, out_ch=16, cee_ch=64):
        super().__init__()
        mid = in_ch // 2  # 96
        self.path_a = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.GELU())
        self.path_b = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.GELU())
        self.interact = nn.Sequential(
            nn.Conv2d(mid*2, mid, 1), nn.Sigmoid())
        self.cee_gate = nn.Sequential(
            nn.Conv2d(cee_ch, mid, 1), nn.Sigmoid())
        self.compress = nn.Sequential(
            nn.Conv2d(mid, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch), nn.GELU())
 
    def forward(self, x, cee):
        a, b = self.path_a(x), self.path_b(x)
        w = self.interact(torch.cat([a, b], 1))
        f = a * w + b * (1 - w)
        f = f * self.cee_gate(cee)
        return self.compress(f)                        # [B,16,H,W]
 
 
class DehazeDiffCodec(nn.Module):
    """Full codec: implicit → (z0, cee)"""
    def __init__(self, impl_ch=192, lat_ch=16, cee_ch=64):
        super().__init__()
        self.cee = ContextEmbeddingElement(impl_ch, cee_ch)
        self.dcu = DimensionalCompressionUnit(impl_ch, lat_ch, cee_ch)
 
    def encode(self, implicit):
        cee = self.cee(implicit)
        z0  = self.dcu(implicit, cee)
        return z0, cee   # z0:[B,16,H,W]  cee:[B,64,H,W]
