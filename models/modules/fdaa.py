import torch
import torch.nn as nn


def fft_decompose(z, low_freq_ratio=0.25):
    b, c, h, w = z.shape
    fft = torch.fft.fftshift(torch.fft.fft2(z, norm="ortho"))
    mask = torch.zeros(h, w, device=z.device, dtype=z.dtype)
    ch = max(1, int(h * low_freq_ratio / 2))
    cw = max(1, int(w * low_freq_ratio / 2))
    mask[h // 2 - ch:h // 2 + ch, w // 2 - cw:w // 2 + cw] = 1.0
    mask = mask.unsqueeze(0).unsqueeze(0)
    f_low = torch.fft.ifft2(torch.fft.ifftshift(fft * mask), norm="ortho").real
    f_high = torch.fft.ifft2(torch.fft.ifftshift(fft * (1 - mask)), norm="ortho").real
    return f_low, f_high


class HazeLocationPrior(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.v_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.encode = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        fh = self.h_pool(x).expand_as(x)
        fv = self.v_pool(x).expand_as(x)
        haze_loc = self.encode(fh * fv)
        return x * haze_loc + x


class CrossFrequencyAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.nh = num_heads
        self.dh = max(1, channels // num_heads)
        inner = self.nh * self.dh

        self.to_q = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_k = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_v = nn.Conv2d(channels, inner, 1, bias=False)
        self.proj = nn.Conv2d(inner, channels, 1, bias=False)

        hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.scale = self.dh ** -0.5

    def forward(self, f_low, f_high):
        b, _, h, w = f_high.shape
        x = torch.cat([f_low, f_high], dim=1)
        x = (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True, unbiased=False) + 1e-6)
        alpha = self.gate(x).view(b, 1, 1, 1)

        def heads(t):
            return t.reshape(b, self.nh, self.dh, h * w).permute(0, 1, 3, 2)

        q = heads(self.to_q(f_high))
        k = heads(self.to_k(f_low))
        v = heads(self.to_v(f_low))

        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(b, -1, h, w)
        out = self.proj(out)

        mixed = alpha * out + (1.0 - alpha) * f_high
        return mixed, alpha.squeeze(-1).squeeze(-1).squeeze(-1)


class FDAA(nn.Module):
    def __init__(self, channels, num_heads=4, low_freq_ratio=0.25):
        super().__init__()
        self.ratio = low_freq_ratio
        self.haze_loc_prior = HazeLocationPrior(channels)
        self.cfab = CrossFrequencyAttentionBlock(channels, num_heads)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
        )

    def forward(self, z0):
        z0 = self.haze_loc_prior(z0)
        f_low, f_high = fft_decompose(z0, self.ratio)
        attn, alpha = self.cfab(f_low, f_high)
        fused = self.fuse(torch.cat([attn, f_low, f_high], dim=1))
        z_fdaa = z0 + fused
        return z_fdaa, f_low, f_high, alpha
