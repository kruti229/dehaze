import torch, torch.nn as nn
 
def fft_decompose(z, low_freq_ratio=0.25):
    """
    Split latent z into low-freq and high-freq components.
    No learnable parameters. O(n log n). Perfect reconstruction:
    f_low + f_high == z  (verified in pipeline step 4)
    """
    B, C, H, W = z.shape
    fft = torch.fft.fftshift(torch.fft.fft2(z, norm="ortho"))
    # Low-freq mask: keep center region
    mask = torch.zeros(H, W, device=z.device)
    ch, cw = int(H * low_freq_ratio / 2), int(W * low_freq_ratio / 2)
    mask[H//2-ch:H//2+ch, W//2-cw:W//2+cw] = 1.0
    m = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    f_low  = torch.fft.ifft2(
        torch.fft.ifftshift(fft * m),     norm="ortho").real
    f_high = torch.fft.ifft2(
        torch.fft.ifftshift(fft * (1-m)), norm="ortho").real
    return f_low, f_high
 
 
class CrossFrequencyAttentionBlock(nn.Module):
    """
    Q = f_high  (what to restore)
    K = f_low   (haze pattern — where is haze?)
    V = f_low   (haze content — how much haze?)
    alpha = learned gating scalar per image
    output = alpha * attn(Q,K,V) + (1-alpha) * Q
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.nh = num_heads
        self.dh = max(1, channels // num_heads)
        inner  = self.nh * self.dh
        self.to_q = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_k = nn.Conv2d(channels, inner, 1, bias=False)
        self.to_v = nn.Conv2d(channels, inner, 1, bias=False)
        self.proj = nn.Conv2d(inner, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels*2, max(8, channels//4)), nn.GELU(),
            nn.Linear(max(8, channels//4), 1), nn.Sigmoid())
        self.scale = self.dh ** -0.5
 
    def forward(self, f_low, f_high):
        B, C, H, W = f_high.shape
        alpha = self.gate(torch.cat([f_low, f_high], 1)).view(B,1,1,1)
        def heads(x):
            return x.reshape(B, self.nh, self.dh, H*W).permute(0,1,3,2)
        Q = heads(self.to_q(f_high))  # [B,nh,HW,dh]
        K = heads(self.to_k(f_low))
        V = heads(self.to_v(f_low))
        attn = torch.softmax(Q @ K.transpose(-2,-1) * self.scale, -1)
        out  = (attn @ V).permute(0,1,3,2).reshape(B,-1,H,W)
        out  = self.proj(out)
        return alpha * out + (1-alpha) * f_high, alpha
 
 
class FDAA(nn.Module):
    def __init__(self, channels, num_heads=8, low_freq_ratio=0.25):
        super().__init__()
        self.ratio = low_freq_ratio
        self.cfab  = CrossFrequencyAttentionBlock(channels, num_heads)
        self.fuse  = nn.Sequential(
            nn.Conv2d(channels*3, channels, 1),
            nn.GroupNorm(min(8,channels), channels), nn.GELU())
 
    def forward(self, z0):
        """
        z0     : [B, C, H, W] latent from DCU
        returns: z_fdaa [B, C, H, W], f_low, f_high, alpha
        """
        f_low, f_high = fft_decompose(z0, self.ratio)
        attn, alpha   = self.cfab(f_low, f_high)
        fused = self.fuse(torch.cat([attn, f_low, f_high], 1))
        return z0 + fused, f_low, f_high, alpha
