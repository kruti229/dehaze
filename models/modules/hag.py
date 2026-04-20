import torch, torch.nn as nn
 
class HAG(nn.Module):
    """
    Estimates haze density from spectral gap between f_low and f_high.
    Dense haze  → small gap (low-freq dominates) → more diffusion steps
    Light haze  → large gap (high-freq active)   → fewer steps
    Output: t_eff (int) = effective number of reverse SDE steps to run.
    """
    def __init__(self, t_max=100, min_steps=20):
        super().__init__()
        self.t_max     = t_max
        self.min_steps = min_steps
        self.gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid())
 
    def forward(self, f_low, f_high):
        d_h = torch.abs(
            f_low.mean([1,2,3]) - f_high.mean([1,2,3])  # [B]
        ).unsqueeze(1)                                    # [B,1]
        g     = self.gate(d_h)          # [B,1] in (0,1)
        g_inv = 1.0 - g                 # dense haze → g_inv → 1
        t_eff = (self.min_steps +
                 g_inv * (self.t_max - self.min_steps))
        t_eff = t_eff.squeeze().int().clamp(self.min_steps, self.t_max)
        return t_eff, g.squeeze()
