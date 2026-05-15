import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_group_count(channels):
    for g in [16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class MediumResidualBlock(nn.Module):
    def __init__(self, channels, expansion=2, use_se=True):
        super().__init__()

        hidden = channels * expansion

        self.norm1 = nn.GroupNorm(choose_group_count(channels), channels)
        self.conv1 = nn.Conv2d(channels, hidden, 3, padding=1)

        self.norm2 = nn.GroupNorm(choose_group_count(hidden), hidden)
        self.conv2 = nn.Conv2d(hidden, channels, 3, padding=1)

        self.se = SEBlock(channels) if use_se else nn.Identity()

        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x

        y = self.norm1(x)
        y = F.silu(y, inplace=True)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.silu(y, inplace=True)
        y = self.conv2(y)

        y = self.se(y)

        return residual + self.res_scale * y


class MediumLatentResidualMapper(nn.Module):
    """
    Direct Stage 2 latent mapper.

    Inference path:
        z_h -> mapper -> delta
        z_pred = z_h + residual_scale * tanh(delta)

    This keeps train/val/test identical.
    """

    def __init__(
        self,
        latent_channels=16,
        width=96,
        num_blocks=10,
        expansion=2,
        use_se=True,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_channels * 2, width, 3, padding=1),
            nn.GroupNorm(choose_group_count(width), width),
            nn.SiLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[
                MediumResidualBlock(
                    channels=width,
                    expansion=expansion,
                    use_se=use_se,
                )
                for _ in range(num_blocks)
            ]
        )

        self.refine = nn.Sequential(
            nn.GroupNorm(choose_group_count(width), width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.out_proj = nn.Conv2d(width, latent_channels, 3, padding=1)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_h, cond=None):
        if cond is None:
            cond = z_h

        x = torch.cat([z_h, cond], dim=1)

        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.refine(x)

        return self.out_proj(x)