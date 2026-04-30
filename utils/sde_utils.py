import math

import torch


class IRSDE:
    def __init__(self, max_sigma=20, T=16, schedule="cosine", device="cuda"):
        self.T = int(T)
        self.max_sigma = float(max_sigma)
        self.device = device
        self.model = None
        self.mu = None

        t = torch.linspace(0, 1, self.T + 1, device=device)
        if schedule == "cosine":
            theta_bars = self.max_sigma * (1 - torch.cos(math.pi * t / 2)) / 2
        else:
            theta_bars = self.max_sigma * t

        self.theta_bars = theta_bars
        self.mu_bars = torch.exp(-theta_bars)
        self.sigma_bars = torch.sqrt(1.0 - torch.exp(-2.0 * theta_bars) + 1e-8)

    def set_model(self, model):
        self.model = model

    def set_mu(self, mu):
        self.mu = mu

    def get_noisy_state(self, z_clean, z_degraded, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=z_clean.device, dtype=torch.long)
        t = t.long().view(-1)
        mu_bar = self.mu_bars[t].to(z_clean.device).view(-1, 1, 1, 1)
        sigma_bar = self.sigma_bars[t].to(z_clean.device).view(-1, 1, 1, 1)
        eps = torch.randn_like(z_clean)
        z_noisy = mu_bar * z_clean + (1.0 - mu_bar) * z_degraded + sigma_bar * eps
        return z_noisy, eps

    # Backward-compatible alias for older code
    def get_noise_target(self, z_clean, z_degraded, t):
        return self.get_noisy_state(z_clean, z_degraded, t)

    def noise_state(self, z_lq):
        sigma_T = self.sigma_bars[self.T].to(z_lq.device)
        return z_lq + sigma_T * torch.randn_like(z_lq)

    @torch.no_grad()
    def reverse_sde(self, xt, t_eff=None, z_fdaa=None, latent_clip=3.0):
        assert self.model is not None, "Call sde.set_model(net) first"

        cond = z_fdaa if z_fdaa is not None else self.mu
        if cond is None:
            raise ValueError("reverse_sde needs z_fdaa or set_mu().")

        if t_eff is None:
            t_run = self.T
        elif torch.is_tensor(t_eff):
            t_run = int(torch.round(t_eff.float().mean()).item())
        else:
            t_run = int(t_eff)
        t_run = max(1, min(t_run, self.T))

        x = xt.clone()

        for t in range(t_run, 0, -1):
            t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            z0_pred = self.model(x, cond, t_vec)

            if t == 1:
                x = z0_pred.clamp(-latent_clip, latent_clip)
            else:
                mu_bar_t1 = self.mu_bars[t - 1].to(x.device)
                x = (
                    mu_bar_t1 * z0_pred
                    + (1.0 - mu_bar_t1) * cond
                ).clamp(-latent_clip, latent_clip)

        return x
