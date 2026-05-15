import torch
import torch.nn as nn
import torch.nn.functional as F


class RefineBlock(nn.Module):
    def __init__(self, ch, groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(groups, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(groups, ch),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class DimensionalDecompressionUnit(nn.Module):
    """
    Decompresses latent z + context cee back to ConvNeXt-like implicit features.

    Main change:
    Context gates are now residual boosting gates:
        old: x = x * gate
        new: x = x * (1 + 0.5 * gate)

    This helps preserve details and color information.
    """

    def __init__(self, in_ch=32, out_ch=192, cee_ch=64):
        super().__init__()

        self.s1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )

        self.cee_gate_early = nn.Sequential(
            nn.Conv2d(cee_ch, 32, 1),
            nn.Sigmoid(),
        )

        self.s2 = nn.Sequential(
            nn.Conv2d(32 + cee_ch, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.refine2 = RefineBlock(64, 8)

        self.s3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )

        self.cee_gate_late = nn.Sequential(
            nn.Conv2d(cee_ch, 96, 1),
            nn.Sigmoid(),
        )

        self.refine3 = RefineBlock(96, 8)

        self.s4 = nn.Sequential(
            nn.Conv2d(96, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def _match_spatial(self, src, ref):
        if src.shape[-2:] != ref.shape[-2:]:
            src = F.interpolate(
                src,
                size=ref.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return src

    def forward(self, z, cee):
        x = self.s1(z)

        cee = self._match_spatial(cee, x)
        gate_early = self.cee_gate_early(cee)

        # residual gate: avoids suppressing low-level detail
        x = x * (1.0 + 0.5 * gate_early)

        cee = self._match_spatial(cee, x)
        x = self.s2(torch.cat([x, cee], dim=1))
        x = self.refine2(x)

        x = self.s3(x)

        cee = self._match_spatial(cee, x)
        gate_late = self.cee_gate_late(cee)

        # residual gate: helps preserve context/color features
        x = x * (1.0 + 0.5 * gate_late)

        x = self.refine3(x)

        return self.s4(x)