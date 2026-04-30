import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        groups = max(1, min(8, c // 8 if c >= 8 else 1))
        self.norm = nn.GroupNorm(groups, c)
        self.pw1 = nn.Conv2d(c, c * 2, 1)
        self.dw = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2)
        self.pw2 = nn.Conv2d(c * 2, c, 1)
        self.act = nn.GELU()

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = self.act(self.pw1(x))
        x = self.act(self.dw(x))
        x = self.pw2(x)
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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, device):
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        t = t.float().view(-1, 1)
        half = self.dim // 2
        emb_scale = math.log(10000) / max(1, half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb_scale)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditionalNAFNet(nn.Module):
    def __init__(
        self,
        img_channel=16,
        width=64,
        middle_blk_num=8,
        enc_blk_nums=(1, 1, 1),
        dec_blk_nums=(1, 1, 1),
        **kwargs
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel * 2, width, 3, padding=1)
        self.cond_proj = nn.Conv2d(img_channel, width, 1)

        self.time_embed = SinusoidalTimeEmbedding(width)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for i, n_blk in enumerate(enc_blk_nums):
            self.enc_stages.append(nn.Sequential(*[SimpleBlock(ch) for _ in range(n_blk)]))
            if i != len(enc_blk_nums) - 1:
                self.downs.append(Downsample(ch, ch * 2))
                ch *= 2

        self.middle = nn.Sequential(*[SimpleBlock(ch) for _ in range(middle_blk_num)])

        self.ups = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i, n_blk in enumerate(dec_blk_nums):
            if i != 0:
                self.ups.append(Upsample(ch, ch // 2))
                ch //= 2
            self.dec_stages.append(nn.Sequential(*[SimpleBlock(ch) for _ in range(n_blk)]))

        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

    def forward(self, z_t, z_condition, t):
        x = self.intro(torch.cat([z_t, z_condition], dim=1))
        cond_feat = self.cond_proj(z_condition)

        t_emb = self.time_embed(t, x.device)
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.expand(x.shape[0], -1)
        t_bias = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)

        x = x + cond_feat + t_bias

        skips = []
        d = 0
        for i, enc in enumerate(self.enc_stages):
            x = enc(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[d](x)
                d += 1

        x = self.middle(x)

        u = 0
        for i, dec in enumerate(self.dec_stages):
            if i > 0:
                x = self.ups[u](x)
                u += 1
                skip = skips[-(i + 1)]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = x + skip
            x = dec(x)

        return self.ending(x)
