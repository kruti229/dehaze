import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEmbeddingElement(nn.Module):
    def __init__(self, in_ch=192, cee_ch=64):
        super().__init__()

        self.fine = nn.Sequential(
            nn.Conv2d(in_ch, cee_ch // 2, 1),
            nn.GELU(),
        )

        self.coarse = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch // 4,
                3,
                stride=4,
                padding=1,
                groups=max(1, in_ch // 4),
            ),
            nn.Conv2d(in_ch // 4, cee_ch // 2, 1),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(cee_ch, cee_ch, 1),
            nn.GroupNorm(8, cee_ch),
            nn.GELU(),
        )

    def forward(self, x):
        _, _, h, w = x.shape

        f = self.fine(x)
        c = self.coarse(x)
        c = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)

        return self.fusion(torch.cat([f, c], dim=1))


class DimensionalCompressionUnit(nn.Module):
    """
    Compresses ConvNeXt implicit features into latent z.

    Main change:
    The CEE gate is now residual:
        old: f = f * gate
        new: f = f * (1 + 0.5 * gate)

    This avoids suppressing feature/color information.
    """

    def __init__(self, in_ch=192, out_ch=32, cee_ch=64):
        super().__init__()

        mid = in_ch // 2

        self.path_a = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.GroupNorm(8, mid),
            nn.GELU(),
        )

        self.path_b = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.GroupNorm(8, mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid),
            nn.GELU(),
        )

        self.interact = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 1),
            nn.Sigmoid(),
        )

        self.cee_gate = nn.Sequential(
            nn.Conv2d(cee_ch, mid, 1),
            nn.Sigmoid(),
        )

        self.compress = nn.Sequential(
            nn.Conv2d(mid, out_ch, 3, padding=1),
            nn.GroupNorm(max(1, out_ch // 4), out_ch),
            nn.GELU(),
        )

    def forward(self, x, cee):
        a = self.path_a(x)
        b = self.path_b(x)

        w = self.interact(torch.cat([a, b], dim=1))
        f = a * w + b * (1.0 - w)

        gate = self.cee_gate(cee)

        # residual gate: preserves information instead of suppressing it
        f = f * (1.0 + 0.5 * gate)

        return self.compress(f)


class DehazeDiffCodec(nn.Module):
    def __init__(self, impl_ch=192, lat_ch=32, cee_ch=64):
        super().__init__()

        self.cee = ContextEmbeddingElement(impl_ch, cee_ch)
        self.dcu = DimensionalCompressionUnit(impl_ch, lat_ch, cee_ch)

    def encode(self, implicit):
        cee = self.cee(implicit)
        z = self.dcu(implicit, cee)

        return z, cee