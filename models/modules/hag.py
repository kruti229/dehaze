import torch
import torch.nn as nn


class HAG(nn.Module):
    def __init__(self, t_max=16, min_steps=4):
        super().__init__()
        self.t_max = t_max
        self.min_steps = min_steps
        self.gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def extract_stats(self, f_low, f_high):
        low_energy = f_low.abs().mean(dim=[1, 2, 3])
        high_energy = f_high.abs().mean(dim=[1, 2, 3])
        gap = torch.abs(low_energy - high_energy)
        ratio = low_energy / (high_energy + 1e-6)
        return torch.stack([low_energy, high_energy, gap, ratio], dim=1)

    def forward(self, f_low, f_high):
        feat = self.extract_stats(f_low, f_high)
        g = self.gate(feat).squeeze(1)
        t_eff = self.min_steps + g * (self.t_max - self.min_steps)
        return t_eff, g
