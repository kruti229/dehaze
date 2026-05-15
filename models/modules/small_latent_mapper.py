import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SmallLatentResidualMapper(nn.Module):
    """
    Small latent mapper for Stage2-F.

    Input:
        z_h: hazy latent
        cond: condition latent, usually z_h

    Output:
        delta latent

    Final conv is zero-initialized so the model starts from identity:
        z_pred = z_h + scale * tanh(delta)
    """

    def __init__(
        self,
        latent_channels=16,
        width=64,
        num_blocks=6,
        cond_channels=None,
    ):
        super().__init__()

        if cond_channels is None:
            cond_channels = latent_channels

        in_channels = latent_channels + cond_channels

        self.in_conv = nn.Conv2d(in_channels, width, 3, padding=1)

        self.blocks = nn.Sequential(
            *[ResidualBlock(width) for _ in range(num_blocks)]
        )

        self.mid = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )

        self.out_conv = nn.Conv2d(width, latent_channels, 3, padding=1)

        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z, cond=None, t=None):
        if cond is None:
            cond = z

        x = torch.cat([z, cond], dim=1)
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.mid(x)
        delta = self.out_conv(x)

        return delta