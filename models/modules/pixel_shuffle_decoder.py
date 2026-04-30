import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class PixelShuffleDecoder(nn.Module):
    def __init__(self, in_channels=192, upscale_factor=8):
        super().__init__()
        if upscale_factor != 8:
            raise ValueError("Only upscale_factor=8 supported")
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GELU(),
        )
        self.up1 = UpBlock(128, 96)
        self.up2 = UpBlock(96, 64)
        self.up3 = UpBlock(64, 32)
        self.tail = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.tail(x)
