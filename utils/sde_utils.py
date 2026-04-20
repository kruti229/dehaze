import torch
 
class IRSDE:
    def __init__(self, max_sigma=50, T=100, schedule="cosine", device="cuda"):
        self.T, self.max_sigma = T, max_sigma
        t  = torch.linspace(0, 1, T+1, device=device)
        # Cosine schedule (smoother gradient flow than linear)
        th = max_sigma * (1 - torch.cos(torch.pi * t / 2)) / 2
        self.thetas = th
        self.sigmas = torch.sqrt(th)
        self.model  = None
        self.mu     = None
 
    def set_mu(self, mu):    self.mu = mu
    def set_model(self, m):  self.model = m
 
    def noise_state(self, z_lq):
        """Add noise to LQ latent for reverse SDE start point."""
        noise = torch.randn_like(z_lq) * self.max_sigma
        return z_lq + noise
 
    def reverse_sde(self, xt, t_eff=None, z_fdaa=None):
        """
        Reverse diffusion loop.
        ★ HAG HOOK: t_eff controls how many steps to run.
          DehazeDiff always uses self.T (100).
          AILD-Freq passes t_eff from HAG (20-100).
        """
        T_run = int(t_eff) if t_eff is not None else self.T
        x = xt
        for t in range(T_run, 0, -1):
            # z_fdaa replaces mu as the conditioning signal
            cond = z_fdaa if z_fdaa is not None else self.mu
            noise_pred = self.model(x, cond, t)
            score = self._noise_to_score(noise_pred, t)
            x = self._reverse_step(x, score, t)
        return x
 
    def _noise_to_score(self, noise, t):
        vt = self.sigmas[t] ** 2
        return -noise / (vt + 1e-8)
 
    def _reverse_step(self, x, score, t):
        th_t  = self.thetas[t]
        th_t1 = self.thetas[t-1] if t > 1 else torch.zeros_like(th_t)
        mu    = self.mu if self.mu is not None else torch.zeros_like(x)
        drift = th_t * (mu - x)
        sigma2_score = self.sigmas[t]**2 * score
        x_new = x + drift - sigma2_score
        if t > 1:
            noise = torch.randn_like(x) * torch.sqrt(th_t - th_t1)
            x_new = x_new + noise
        return x_new
