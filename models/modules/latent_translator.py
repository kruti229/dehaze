import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentTranslatorBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, ch), ch)

        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, ch), ch)

        self.act = nn.GELU()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return self.act(residual + x)


class LatentTranslator(nn.Module):
    def __init__(
        self,
        lat_ch=32,
        cee_ch=64,
        hidden_ch=128,
        num_blocks=6,
    ):
        super().__init__()

        in_ch = lat_ch + cee_ch + 1

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1),
            nn.GroupNorm(8, hidden_ch),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[LatentTranslatorBlock(hidden_ch) for _ in range(num_blocks)]
        )

        self.output_proj = nn.Conv2d(hidden_ch, lat_ch, 1)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_h, c_guided, t_h):
        if c_guided.shape[-2:] != z_h.shape[-2:]:
            c_guided = F.interpolate(
                c_guided,
                size=z_h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if t_h.shape[-2:] != z_h.shape[-2:]:
            t_h = F.interpolate(
                t_h,
                size=z_h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([z_h, c_guided, t_h], dim=1)

        x = self.input_proj(x)
        x = self.blocks(x)

        delta = self.output_proj(x)

        return z_h + delta, delta