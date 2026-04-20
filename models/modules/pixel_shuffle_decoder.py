import torch
import torch.nn as nn

class PixelShuffleDecoder(nn.Module):
    def __init__(self, in_channels=192, upscale_factor=8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, 3 * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.body(x)