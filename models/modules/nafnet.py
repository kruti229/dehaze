import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, device):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        t = t.float().view(-1, 1)
        half = self.dim // 2
        scale = math.log(10000) / max(1, half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -scale)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ChannelGate(nn.Module):
    def __init__(self, c, squeeze=4):
        super().__init__()
        hidden = max(8, c // squeeze)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class StrongBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        groups = max(1, min(8, c // 8))
        self.norm1 = nn.GroupNorm(groups, c)
        self.pw1 = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.pw2 = nn.Conv2d(c * 2, c, 1)
        self.act = nn.GELU()
        self.gate = ChannelGate(c)

        self.norm2 = nn.GroupNorm(groups, c)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            nn.GELU(),
            nn.Conv2d(c * 2, c, 1),
        )

    def forward(self, x):
        r = x
        x = self.norm1(x)
        x = self.act(self.pw1(x))
        x = self.act(self.dw(x))
        x = self.pw2(x)
        x = self.gate(x)
        x = x + r

        r = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + r


class Downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.op = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_out * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.op(x)


class CondProject(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1),
        )

    def forward(self, x):
        return self.proj(x)


class ConditionalNAFNet(nn.Module):
    def __init__(
        self,
        img_channel=16,
        width=96,
        middle_blk_num=12,
        enc_blk_nums=(2, 2, 4),
        dec_blk_nums=(2, 2, 2),
        **kwargs,
    ):
        super().__init__()

        self.intro = nn.Conv2d(img_channel * 2, width, 3, padding=1)
        self.cond_proj0 = CondProject(img_channel, width)

        self.time_embed = SinusoidalTimeEmbedding(width)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

        self.encoders = nn.ModuleList()
        self.cond_projs = nn.ModuleList()
        self.downs = nn.ModuleList()

        ch = width
        for i, n_blk in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[StrongBlock(ch) for _ in range(n_blk)]))
            self.cond_projs.append(CondProject(img_channel, ch))
            if i < len(enc_blk_nums) - 1:
                self.downs.append(Downsample(ch, ch * 2))
                ch *= 2

        self.middle = nn.Sequential(*[StrongBlock(ch) for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_fuse = nn.ModuleList()

        for i, n_blk in enumerate(dec_blk_nums):
            if i > 0:
                self.ups.append(Upsample(ch, ch // 2))
                self.skip_fuse.append(nn.Conv2d(ch, ch // 2, 1))
                ch //= 2
            self.decoders.append(nn.Sequential(*[StrongBlock(ch) for _ in range(n_blk)]))

        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

    def forward(self, z_t, z_condition, t):
        x = self.intro(torch.cat([z_t, z_condition], dim=1))
        cond0 = self.cond_proj0(z_condition)

        t_emb = self.time_embed(t, x.device)
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.expand(x.shape[0], -1)
        t_bias = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)

        x = x + cond0 + t_bias

        skips = []
        cond_pyramid = [z_condition]

        for _ in range(len(self.encoders) - 1):
            cond_pyramid.append(F.avg_pool2d(cond_pyramid[-1], kernel_size=2, stride=2))

        d = 0
        for i, enc in enumerate(self.encoders):
            x = x + self.cond_projs[i](cond_pyramid[i])
            x = enc(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[d](x)
                d += 1

        x = self.middle(x)

        u = 0
        for i, dec in enumerate(self.decoders):
            if i > 0:
                x = self.ups[u](x)
                skip = skips[-(i + 1)]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = self.skip_fuse[u](torch.cat([x, skip], dim=1))
                u += 1
            x = dec(x)

        return self.ending(x)
