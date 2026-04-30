import torch
import torch.nn as nn


class SimplePreDehazer(nn.Module):
    def __init__(self, ch=32):
        super().__init__()

        self.e1 = self._block(3, ch)          # H
        self.e2 = self._block(ch, ch * 2)     # H/2
        self.e3 = self._block(ch * 2, ch * 4) # H/4
        self.pool = nn.MaxPool2d(2)

        self.bot = self._block(ch * 4, ch * 8)  # H/8

        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)  # H/8 -> H/4
        self.d3 = self._block(ch * 8, ch * 4)

        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)  # H/4 -> H/2
        self.d2 = self._block(ch * 4, ch * 2)

        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)      # H/2 -> H
        self.d1 = self._block(ch * 2, ch)

        self.out = nn.Sequential(
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def _block(self, ic, oc):
        g = min(8, oc)
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GroupNorm(g, oc),
            nn.GELU(),
            nn.Conv2d(oc, oc, 3, padding=1),
            nn.GroupNorm(g, oc),
            nn.GELU(),
        )

    def forward(self, x):
        e1 = self.e1(x)                 # H
        e2 = self.e2(self.pool(e1))     # H/2
        e3 = self.e3(self.pool(e2))     # H/4
        b = self.bot(self.pool(e3))     # H/8

        d3 = self.up3(b)                # H/4
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)               # H/2
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)               # H
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        return (self.out(d1) + x).clamp(0, 1)
