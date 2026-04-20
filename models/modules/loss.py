import torch, torch.nn as nn
 
class FrequencyConsistencyLoss(nn.Module):
    """
    L_freq = || FFT(I_pred) - FFT(I_gt) ||_1
    Enforces spectral alignment — correct global color tone,
    contrast, and illumination. DehazeDiff has no freq loss.
    """
    def forward(self, pred, target):
        fp = torch.fft.fft2(pred,   norm="ortho")
        ft = torch.fft.fft2(target, norm="ortho")
        return (torch.abs(fp.real - ft.real).mean() +
                torch.abs(fp.imag - ft.imag).mean())
 
 
class DifferentialPerceptionLoss(nn.Module):
    """DehazeDiff DP loss — unchanged from base paper."""
    def __init__(self):
        super().__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3))
 
    def forward(self, pred, target):
        diff = torch.abs(target - pred)
        w = torch.sigmoid(
            self.mlp(self.gmp(diff).squeeze(-1))
            .unsqueeze(-1).unsqueeze(-1))
        return (w * diff).mean()
 
 
class AILDFreqLoss(nn.Module):
    """
    Total loss = lambda1*L1 + lambda2*L_freq + lambda3*L_dp
    Stage 1: only L1 + L_dp (same as DehazeDiff)
    Stage 2: full loss with L_freq
    """
    def __init__(self, l1=1.0, l2=0.5, l3=0.1):
        super().__init__()
        self.l1_fn   = nn.L1Loss()
        self.freq_fn = FrequencyConsistencyLoss()
        self.dp_fn   = DifferentialPerceptionLoss()
        self.l1, self.l2, self.l3 = l1, l2, l3
 
    def forward(self, pred, target, stage=2):
        l1 = self.l1_fn(pred, target)
        dp = self.dp_fn(pred, target)
        if stage == 1:
            return l1 + dp
        freq = self.freq_fn(pred, target)
        return self.l1*l1 + self.l2*freq + self.l3*dp
