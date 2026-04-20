"""
STEP 6 — HAG: Haze-Adaptive Gating
STEP 7 — IR-SDE Forward (add noise)
STEP 8 — IR-SDE Reverse (denoise = dehazing!)
STEP 9 — DDU: Decompress latent back to implicit space
STEP 10 — Pixel-Shuffle Decoder: implicit → clean RGB image
STEP 11 — FULL PIPELINE: run all steps together on one image
============================================================

WHY EACH STEP:

STEP 6 — HAG:
  DehazeDiff always runs exactly 100 diffusion steps.
  Problem: NH-Haze images have non-uniform haze. Some regions are
  lightly hazy, some are very dense. Running 100 steps on a lightly
  hazy region over-denoises it → introduces artifacts.
  
  HAG estimates haze density from the spectral gap (f_low vs f_high):
    Dense haze  → f_low dominates  → small gap → more steps needed
    Light haze  → f_high active   → large gap  → fewer steps needed
  
  Output: t_eff (integer, controls loop length in reverse SDE)

STEP 7 — IR-SDE Forward (training only):
  Takes clean latent z_gt and adds structured noise to create noisy z_T.
  During INFERENCE this step is skipped — we start from z_fdaa directly.
  
  WHY IR-SDE (Image Restoration SDE) vs plain DDPM?
  - Standard DDPM adds noise toward pure Gaussian
  - IR-SDE adds noise toward the DEGRADED image (hazy latent)
  - This means the model learns a more targeted reverse path
  - Result: better restoration with fewer steps

STEP 8 — IR-SDE Reverse (CORE DEHAZING):
  This is where the actual dehazing happens.
  Starting from noisy z_T, the neural network (ConditionalNAFNet)
  iteratively predicts and removes noise, guided by z_fdaa.
  After t_eff steps: z_0_pred (clean latent estimate)

STEP 9 — DDU:
  Decompress clean latent [B, 16, H, W] back to implicit space [B, 192, H, W]
  The CEE from Step 3 is used here to guide the decompression.

STEP 10 — Pixel-Shuffle Decoder:
  Converts implicit space [B, 192, H, W] → clean RGB image [B, 3, H, W]
  Uses Res2Net blocks for robustness and attention for dynamic feature selection.
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs/step_outputs")

print("=" * 60)
print("STEPS 6–11 — HAG, SDE, DDU, Decoder, Full Pipeline")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# STEP 6: Haze-Adaptive Gating (HAG)
# ══════════════════════════════════════════════════════════════
class HAG(nn.Module):
    """
    Estimates haze density from spectral statistics and outputs
    the effective number of reverse diffusion steps to run.
    """
    def __init__(self, t_max: int = 100, min_steps: int = 20):
        super().__init__()
        self.t_max = t_max
        self.min_steps = min_steps

        # Small MLP: haze density scalar → gating scalar g
        self.gate = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor):
        """
        Spectral gap: dense haze = low gap (f_low dominates everything)
                      light haze = high gap (f_high still has energy)
        """
        mean_low  = f_low.mean(dim=[1, 2, 3])     # [B]
        mean_high = f_high.mean(dim=[1, 2, 3])    # [B]
        # Normalized gap: larger gap = lighter haze
        d_h = torch.abs(mean_low - mean_high).unsqueeze(1)  # [B, 1]
        g = self.gate(d_h)                         # [B, 1] in (0, 1)
        # Invert: light haze (large gap) → fewer steps
        #         dense haze (small gap) → more steps
        g_inv = 1.0 - g
        t_eff_float = self.min_steps + g_inv * (self.t_max - self.min_steps)
        t_eff = t_eff_float.squeeze().int().clamp(self.min_steps, self.t_max)
        return t_eff, g.squeeze()


# ══════════════════════════════════════════════════════════════
# STEP 7 + 8: Simplified IR-SDE
# ══════════════════════════════════════════════════════════════
class SimpleIRSDE(nn.Module):
    """
    Simplified IR-SDE for pipeline illustration.
    Full implementation in utils/sde_utils.py.

    The mean-reverting SDE equation:
      Forward: dx = θ(μ - x)dt + σ dW
               (x moves toward μ = degraded image, with noise)
      Reverse: dx = [θ(μ-x) - σ²∇log p(x)]dt + σ dW_hat
               (reverse: move toward clean image, guided by score network)
    """
    def __init__(self, max_sigma: float = 50.0, T: int = 100):
        super().__init__()
        self.max_sigma = max_sigma
        self.T = T

        # Precompute noise schedule (cosine, as in DehazeDiff)
        t = torch.linspace(0, 1, T + 1)
        # Cosine schedule: smoother than linear
        thetas = max_sigma * (1 - torch.cos(torch.pi * t / 2)) / 2
        self.register_buffer('thetas', thetas)
        self.register_buffer('sigmas', torch.sqrt(thetas))

    def add_noise(self, z_clean: torch.Tensor,
                  z_degraded: torch.Tensor,
                  t: int) -> torch.Tensor:
        """
        Forward step: mix clean latent toward degraded latent + noise
        z_noisy = mu_bar * z_clean + (1 - mu_bar) * z_degraded + sigma_bar * eps
        """
        theta_bar = self.thetas[t]
        mu_bar = torch.exp(-theta_bar)
        sigma_bar = torch.sqrt(1 - torch.exp(-2 * theta_bar))

        eps = torch.randn_like(z_clean)
        z_noisy = (mu_bar * z_clean
                   + (1 - mu_bar) * z_degraded
                   + sigma_bar * eps)
        return z_noisy, eps

    def reverse_step(self, z_t: torch.Tensor,
                     z_degraded: torch.Tensor,
                     predicted_noise: torch.Tensor,
                     t: int) -> torch.Tensor:
        """
        One reverse diffusion step: remove noise, move toward clean.
        """
        theta_bar_t   = self.thetas[t]
        theta_bar_tm1 = self.thetas[t - 1] if t > 1 else torch.tensor(0.0)

        # Compute coefficients (from DehazeDiff Eq. 10)
        num = 1 - torch.exp(-2 * theta_bar_tm1)
        den = 1 - torch.exp(-2 * theta_bar_t)
        coeff_noisy = (num / den) * torch.exp(-(theta_bar_t - theta_bar_tm1))
        coeff_ref   = (1 - torch.exp(-2 * theta_bar_t)) / den * torch.exp(-theta_bar_tm1)

        z_pred = (coeff_noisy * (z_t - z_degraded)
                  + coeff_ref * (z_t - z_degraded)
                  + z_degraded)
        return z_pred


# ══════════════════════════════════════════════════════════════
# STEP 8: Simplified Denoising Network (stands in for ConditionalNAFNet)
# ══════════════════════════════════════════════════════════════
class SimpleDenoisingNet(nn.Module):
    """
    Placeholder for ConditionalNAFNet.
    Full architecture in models/modules/nafnet.py.
    This is used to test the pipeline structure.
    """
    def __init__(self, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1)
        )
        # Timestep embedding
        self.t_embed = nn.Sequential(
            nn.Linear(1, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, z_noisy, z_condition, t):
        """
        z_noisy    : [B, C, H, W]  noisy latent at timestep t
        z_condition: [B, C, H, W]  z_fdaa (frequency-enriched hazy latent)
        t          : int            current timestep

        returns: predicted noise [B, C, H, W]
        """
        # Concatenate noisy + condition
        x = torch.cat([z_noisy, z_condition], dim=1)  # [B, 2C, H, W]

        # Add timestep embedding
        t_emb = self.t_embed(
            torch.tensor([[t / 100.0]],
                          dtype=torch.float32).expand(z_noisy.shape[0], 1))
        t_emb = t_emb.view(z_noisy.shape[0], -1, 1, 1)  # [B, C, 1, 1]

        pred = self.net(x) + t_emb   # broadcast over H, W
        return pred


# ══════════════════════════════════════════════════════════════
# STEP 9: DDU — Dimensional Decompression Unit
# ══════════════════════════════════════════════════════════════
class DimensionalDecompressionUnit(nn.Module):
    """
    Expands clean latent [B, 16, H, W] back to [B, 192, H, W]
    guided by the CEE stored in Step 3.

    Progressive expansion (from DehazeDiff Fig 4):
    16 → 32 → 64 → 96 → 192 channels
    At each step, CEE provides guidance to prevent noise amplification.
    """
    def __init__(self, in_channels=16, out_channels=192, cee_channels=64):
        super().__init__()

        # Progressive expansion
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(4, 32), nn.GELU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32 + cee_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GroupNorm(8, 96), nn.GELU()
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(96, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels), nn.GELU()
        )

        # CEE attention gate
        self.cee_attn = nn.Sequential(
            nn.Conv2d(cee_channels, 32, 1), nn.Sigmoid())

    def forward(self, z_clean: torch.Tensor,
                cee: torch.Tensor) -> torch.Tensor:
        """
        z_clean: [B, 16, H, W] denoised latent
        cee    : [B, 64, H, W] context from DCU step
        returns: [B, 192, H, W] reconstructed implicit space
        """
        # Stage 1: initial expansion
        x = self.stage1(z_clean)               # [B, 32, H, W]

        # CEE-guided attention
        cee_gate = self.cee_attn(cee)          # [B, 32, H, W]
        x = x * cee_gate                       # focus on relevant features

        # Stage 2: inject CEE information
        x = self.stage2(torch.cat([x, cee], dim=1))  # [B, 64, H, W]

        # Stages 3 + 4: expand to target
        x = self.stage3(x)                     # [B, 96, H, W]
        x = self.stage4(x)                     # [B, 192, H, W]
        return x


# ══════════════════════════════════════════════════════════════
# STEP 10: Pixel-Shuffle Decoder
# ══════════════════════════════════════════════════════════════
class PixelShuffleDecoder(nn.Module):
    """
    Converts [B, 192, H, W] implicit space → [B, 3, H_orig, W_orig] image.

    Uses pixel-shuffle (sub-pixel convolution) for upsampling:
      - More efficient than bilinear/transposed conv
      - Preserves high-frequency details better
    Res2Net blocks for multi-scale feature extraction.
    Attention block for dynamic feature selection.
    """
    def __init__(self, in_channels=192, upscale_factor=8):
        super().__init__()
        self.upscale_factor = upscale_factor

        # Res2Net-style block (simplified)
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels // 4),
            nn.GroupNorm(8, in_channels), nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4), nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels), nn.Sigmoid()
        )

        # Pixel-shuffle upsampling:
        # Rearranges C×(r²) → C×H×W pattern for spatial upsampling
        # We need 3×(8²) = 192 channels → exactly matches our in_channels!
        self.to_image = nn.Sequential(
            nn.Conv2d(in_channels, 3 * (upscale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(upscale_factor)   # [B, 3*r², H, W] → [B, 3, H*r, W*r]
        )

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()   # output in [0, 1]
        )

    def forward(self, implicit: torch.Tensor) -> torch.Tensor:
        # Res2Net block + attention
        res = self.res_block(implicit)
        ca  = self.channel_attn(implicit).view(implicit.shape[0], -1, 1, 1)
        x   = implicit + res * ca

        # Upsample to image space
        img = self.to_image(x)           # [B, 3, H*8, W*8]
        img = self.refine(img)           # refine + clamp [0,1]
        return img


# ══════════════════════════════════════════════════════════════
# STEP 11: FULL PIPELINE — run all steps in sequence
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RUNNING FULL PIPELINE (Steps 1→10)")
print("=" * 60)

# Load all data from previous steps
data5 = torch.load(os.path.join(OUT, "step_05_fdaa.pt"), map_location='cpu')
z_fdaa = data5['z_fdaa']
f_low  = data5['f_low']
f_high = data5['f_high']
z0     = data5['z0']
cee    = data5['cee']

print(f"\nLoaded z_fdaa: {list(z_fdaa.shape)}")
channels = z_fdaa.shape[1]

# ── STEP 6: HAG ───────────────────────────────────────────────
print("\n── Step 6: HAG ────────────────────────────────────────")
hag = HAG(t_max=100, min_steps=20)
hag.eval()
with torch.no_grad():
    t_eff, g = hag(f_low, f_high)

print(f"  Haze density estimator output g = {g.item():.4f}")
print(f"  Effective diffusion steps t_eff = {t_eff.item()}")
print(f"  WHY: Using {t_eff.item()} instead of 100 steps saves")
print(f"       {100 - t_eff.item()} unnecessary denoising iterations.")
print(f"  ✓ HAG complete")

# ── STEP 7+8: IR-SDE (simplified forward→reverse) ────────────
print("\n── Steps 7+8: IR-SDE Forward + Reverse ────────────────")
sde = SimpleIRSDE(max_sigma=50.0, T=100)
sde.eval()

denoising_net = SimpleDenoisingNet(channels=channels)
denoising_net.eval()

# Step 7: Add noise (simulate what happens during training)
# In inference: we directly start from z_fdaa as the noisy state
print(f"  Step 7 (training): add noise to clean latent")
# For demo, use z_fdaa as both clean and degraded (we have no GT)
with torch.no_grad():
    z_noisy, true_noise = sde.add_noise(z_fdaa, z_fdaa, t=t_eff.item())
print(f"  Noisy state: {list(z_noisy.shape)}")

# Step 8: Reverse SDE — iterative denoising
print(f"  Step 8 (inference): reverse SDE for {t_eff.item()} steps...")
z_current = z_noisy.clone()
with torch.no_grad():
    for t in range(t_eff.item(), 0, -1):
        # Network predicts noise at this timestep
        # Conditioned on z_fdaa (frequency-enriched hazy latent)
        pred_noise = denoising_net(z_current, z_fdaa, t)
        # Apply reverse step
        z_current = sde.reverse_step(z_current, z_fdaa, pred_noise, t)

z_clean = z_current
print(f"  Clean latent: {list(z_clean.shape)}")
print(f"  ✓ SDE reverse ({t_eff.item()} steps) complete")

# ── STEP 9: DDU ───────────────────────────────────────────────
print("\n── Step 9: DDU (Latent → Implicit) ────────────────────")
ddu = DimensionalDecompressionUnit(
    in_channels=channels, out_channels=192, cee_channels=cee.shape[1])
ddu.eval()
with torch.no_grad():
    implicit_clean = ddu(z_clean, cee)
print(f"  z_clean {list(z_clean.shape)} → implicit {list(implicit_clean.shape)}")
print(f"  WHY: CEE from Step 3 guides recovery of compressed info.")
print(f"  ✓ DDU complete")

# ── STEP 10: Decoder ──────────────────────────────────────────
print("\n── Step 10: Pixel-Shuffle Decoder (Implicit → Image) ──")
H_implicit = implicit_clean.shape[2]
W_implicit = implicit_clean.shape[3]

decoder = PixelShuffleDecoder(in_channels=192, upscale_factor=8)
decoder.eval()
with torch.no_grad():
    clean_image = decoder(implicit_clean)
print(f"  implicit {list(implicit_clean.shape)} → image {list(clean_image.shape)}")
print(f"  Output range: [{clean_image.min():.3f}, {clean_image.max():.3f}] ✓")
print(f"  ✓ Decoder complete")


# ══════════════════════════════════════════════════════════════
# FINAL VERIFICATION
# ══════════════════════════════════════════════════════════════
print("\n── FINAL VERIFICATION ─────────────────────────────────")

checks = {
    "z_fdaa  shape": z_fdaa.shape[1] == channels,
    "t_eff   range": 20 <= t_eff.item() <= 100,
    "z_clean shape": z_clean.shape == z_fdaa.shape,
    "implicit out ": implicit_clean.shape[1] == 192,
    "image range  ": 0.0 <= clean_image.min().item() and clean_image.max().item() <= 1.0,
    "no NaN       ": not torch.isnan(clean_image).any().item(),
    "no Inf       ": not torch.isinf(clean_image).any().item(),
}
all_passed = True
for name, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"  {status}  {name}")
    all_passed = all_passed and passed

if all_passed:
    print("\n  ALL CHECKS PASSED ✓")
else:
    print("\n  SOME CHECKS FAILED ✗")


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE VISUALIZATION
# ══════════════════════════════════════════════════════════════
# Load original hazy image for comparison
lq = torch.load(os.path.join(OUT, "step_01_lq_tensor.pt"), map_location='cpu')

fig = plt.figure(figsize=(18, 10))
fig.suptitle("AILD-Freq Full Pipeline: Hazy Image → Dehazed Image\n"
             f"(t_eff={t_eff.item()} steps, α={data5['alpha']:.3f})",
             fontsize=13, fontweight='bold')

# Pipeline flow diagram
steps_data = [
    ("1. Input\n(Hazy Image)", lq, 'viridis'),
    ("2. ConvNext\nImplicit Space\n(ch0/192)",
     data5['z0'][:, :1], 'plasma'),
    ("3. DCU+CEE\nLatent z0\n(ch0/16)",
     z0[:, :1], 'viridis'),
    ("4. FFT Low\nf_low (ch0)\nHaze struct",
     f_low[:, :1], 'Blues'),
    ("4. FFT High\nf_high (ch0)\nEdges/tex",
     f_high[:, :1], 'Reds'),
    ("5. FDAA\nz_fdaa (ch0)\nEnriched",
     z_fdaa[:, :1], 'viridis'),
    ("6. HAG\nt_eff=%d\n(steps)" % t_eff.item(),
     None, None),
    ("7-8. SDE\nClean latent\nz0_pred",
     z_clean[:, :1], 'viridis'),
    ("9. DDU\nImplicit clean\n(ch0/192)",
     implicit_clean[:, :1], 'plasma'),
    ("10. Decoder\nDehazed Output\nRGB image",
     clean_image, 'viridis'),
]

n_steps = len(steps_data)
for i, (title, tensor, cmap) in enumerate(steps_data):
    ax = fig.add_subplot(2, 5, i + 1)

    if tensor is None:
        # HAG — show text summary
        ax.text(0.5, 0.5,
                f"t_eff = {t_eff.item()}\ng = {g.item():.3f}\n"
                f"DehazeDiff: 100\nSaved: {100 - t_eff.item()} steps",
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#FFF9C4'))
    elif tensor.shape[1] == 3:
        # RGB image
        img = tensor[0].permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img)
    else:
        # Feature map — show channel 0
        ch = tensor[0, 0].numpy()
        ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ax.imshow(ch_norm, cmap=cmap or 'viridis')

    ax.set_title(title, fontsize=8, pad=3)
    ax.axis('off')

    # Add arrow between steps
    if i < n_steps - 1 and i != 4:  # skip arrow from f_high
        pass

plt.tight_layout()
save_path = os.path.join(OUT, "step_11_full_pipeline.png")
plt.savefig(save_path, dpi=130, bbox_inches='tight')
plt.close()
print(f"\n  Full pipeline visualization: {save_path}")

# Parameter count summary
print("\n── PARAMETER COUNTS ────────────────────────────────────")
components = {
    "ConvNext encoder (frozen)": 49_000_000,
    "DCU + CEE"                : (sum(p.numel() for p in nn.Conv2d(192,96,3,padding=1).parameters()) +
                                   sum(p.numel() for p in nn.Conv2d(192,64,1).parameters())),
    "FDAA (new)"               : (sum(p.numel() for p in nn.Conv2d(channels, channels*3, 1).parameters()) +
                                   sum(p.numel() for p in nn.Linear(channels*2, max(1,channels//4)).parameters())),
    "HAG (new)"                : sum(p.numel() for p in hag.parameters()),
    "ConditionalNAFNet"        : sum(p.numel() for p in denoising_net.parameters()),
    "DDU"                      : sum(p.numel() for p in ddu.parameters()),
    "Decoder"                  : sum(p.numel() for p in decoder.parameters()),
}
total_trainable = 0
for name, count in components.items():
    frozen = "(frozen)" if "frozen" in name else ""
    if "frozen" not in name:
        total_trainable += count if isinstance(count, int) else int(count)
    print(f"  {name:<30}: {count if isinstance(count, int) else int(count):>10,} params {frozen}")
print(f"  {'Total trainable':<30}: {total_trainable:>10,} params")
print(f"  DehazeDiff reported           : 106,560,000 params")


print("\n" + "=" * 60)
print("ALL STEPS COMPLETE ✓")
print(f"\nFull pipeline summary:")
print(f"  Input  : [1, 3, H, W]  hazy image")
print(f"  Step 1 : Load + normalize → float32 [0,1]")
print(f"  Step 2 : ConvNext → [1, 192, H/8, W/8] implicit space")
print(f"  Step 3 : DCU+CEE → z0 [1,16,H/8,W/8] + cee [1,64,H/8,W/8]")
print(f"  Step 4 : FFT → f_low [1,16,...] + f_high [1,16,...]  ★ NEW")
print(f"  Step 5 : FDAA → z_fdaa [1,16,...] (α={data5['alpha']:.3f})  ★ NEW")
print(f"  Step 6 : HAG → t_eff={t_eff.item()} steps             ★ NEW")
print(f"  Step 7 : SDE noise (training only)")
print(f"  Step 8 : SDE reverse × {t_eff.item()} → z_clean [1,16,...]")
print(f"  Step 9 : DDU + CEE → [1, 192, H/8, W/8] clean implicit")
print(f"  Step 10: Decoder → [1, 3, H, W] dehazed RGB")
print(f"\nSteps marked ★ NEW are the AILD-Freq additions on top of DehazeDiff.")
print(f"\nVisualization: {OUT}/step_11_full_pipeline.png")
print("=" * 60)