"""
STEP 4 — FFT FREQUENCY DECOMPOSITION (★ NEW in AILD-Freq)
===========================================================

WHY THIS STEP EXISTS:
  This is where AILD-Freq diverges from DehazeDiff.
  DehazeDiff feeds z0 directly into the diffusion model.
  We first analyze it in the frequency domain.

  THE PHYSICS OF HAZE:
    Haze is not uniform — it affects frequencies differently:

    LOW FREQUENCY (smooth, global changes):
      - Global illumination → haze makes everything uniformly brighter
      - Contrast reduction → the "foggy white" overlay is low-frequency
      - Atmospheric light A → constant or slowly varying = low-freq

    HIGH FREQUENCY (sharp, local changes):
      - Edges → blurred by haze particles scattering light
      - Fine textures → degraded / smeared
      - Local contrast → severely reduced in dense haze

    If you just treat all latent values the same (as DehazeDiff does),
    you're asking one model to solve two very different problems at once:
      - Problem A: Remove the global brightness bias
      - Problem B: Recover the sharp edges and textures

    By SEPARATING these into f_low and f_high, FDAA (Step 5) can
    address each problem with a targeted attention mechanism.

  WHY FFT specifically (not wavelets, not DCT)?
    - FFT is O(n log n) — essentially free computation
    - torch.fft.fft2 is natively implemented in PyTorch (GPU-optimized)
    - Clean amplitude/phase decomposition maps exactly to haze physics
    - No extra learnable parameters (unlike wavelet learned filters)

WHAT HAPPENS:
  z0 [B, 16, H, W]  →  FFT  →  frequency domain (complex)
                      → fftshift (move DC to center)
                      → split at center: inner = low-freq, outer = high-freq
                      → IFFT back to spatial domain
                      → f_low [B, 16, H, W]  (haze distribution info)
                      → f_high [B, 16, H, W] (edge/texture info)
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs/step_outputs")

print("=" * 60)
print("STEP 4 — FFT Frequency Decomposition")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# The FFT decomposition function
# ──────────────────────────────────────────────────────────────
def fft_decompose(z: torch.Tensor, low_freq_ratio: float = 0.25):
    """
    Decompose latent z into low and high frequency components.

    Args:
        z              : [B, C, H, W] latent tensor
        low_freq_ratio : fraction of spatial frequencies kept as 'low'
                         0.25 means inner 25% of H and W is low-frequency
                         This corresponds to changes happening over >4 pixels

    Returns:
        f_low  : [B, C, H, W] low-frequency component (global structure, haze)
        f_high : [B, C, H, W] high-frequency component (edges, textures)
        magnitude_spectrum : [B, C, H, W] for visualization
    """
    B, C, H, W = z.shape

    # ── Step A: 2D FFT ────────────────────────────────────────
    # Converts spatial domain → frequency domain
    # Each pixel in freq domain = one sinusoidal component of the image
    # norm='ortho' ensures energy is preserved (Parseval's theorem)
    fft = torch.fft.fft2(z, norm='ortho')          # complex [B, C, H, W]

    # ── Step B: FFT shift ─────────────────────────────────────
    # By default, FFT puts DC (zero-frequency) at corners.
    # fftshift moves DC to CENTER, making the mask easier to define.
    # After shift: center = low-freq, edges = high-freq
    fft_shifted = torch.fft.fftshift(fft)

    # ── Step C: Build low/high frequency masks ────────────────
    # Low-pass mask: keep only the central region
    # High-pass mask = 1 - low-pass mask
    mask_low = torch.zeros(H, W, device=z.device, dtype=torch.float32)
    ch = int(H * low_freq_ratio / 2)    # half-height of low-freq region
    cw = int(W * low_freq_ratio / 2)    # half-width
    # Center the mask
    h_center, w_center = H // 2, W // 2
    mask_low[h_center - ch: h_center + ch,
             w_center - cw: w_center + cw] = 1.0

    mask_high = 1.0 - mask_low

    # Broadcast masks to [B, C, H, W]
    mask_low  = mask_low.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    mask_high = mask_high.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # ── Step D: Apply masks in frequency domain ───────────────
    # Complex multiplication with real mask = element-wise masking
    fft_low  = fft_shifted * mask_low    # keep center (low-freq)
    fft_high = fft_shifted * mask_high   # keep edges (high-freq)

    # ── Step E: Convert back to spatial domain ────────────────
    # We take the REAL part of the IFFT (imaginary should be ~0
    # due to the original signal being real-valued)
    # Use torch.abs to get magnitude (always positive, stable)
    f_low  = torch.fft.ifft2(
        torch.fft.ifftshift(fft_low),  norm='ortho').real
    f_high = torch.fft.ifft2(
        torch.fft.ifftshift(fft_high), norm='ortho').real

    # Magnitude spectrum for visualization
    magnitude = torch.log1p(torch.abs(fft_shifted))

    return f_low, f_high, magnitude, mask_low.squeeze()


# ──────────────────────────────────────────────────────────────
# Load z0 from Step 3
# ──────────────────────────────────────────────────────────────
latent_data = torch.load(os.path.join(OUT, "step_03_latent.pt"),
                          map_location='cpu')
z0 = latent_data['z0']
print(f"\nLoaded z0 from Step 3: {list(z0.shape)}")

# Run decomposition
print("\nRunning FFT decomposition...")
f_low, f_high, magnitude, mask = fft_decompose(z0, low_freq_ratio=0.25)


# ──────────────────────────────────────────────────────────────
# VERIFY
# ──────────────────────────────────────────────────────────────
print("\n── VERIFICATION ──────────────────────────────────────")

print(f"\n[A] Output shapes")
print(f"    z0     : {list(z0.shape)}")
print(f"    f_low  : {list(f_low.shape)}  ← global structure, haze distribution")
print(f"    f_high : {list(f_high.shape)}  ← edges, textures, fine details")
assert f_low.shape == z0.shape,  "f_low shape mismatch!"
assert f_high.shape == z0.shape, "f_high shape mismatch!"
print("    ✓ Shapes correct")

print(f"\n[B] Energy distribution (how much info in each band)")
total_energy = (z0 ** 2).sum().item()
low_energy   = (f_low ** 2).sum().item()
high_energy  = (f_high ** 2).sum().item()
print(f"    Total z0 energy   : {total_energy:.2f}")
print(f"    Low-freq energy   : {low_energy:.2f}  ({100*low_energy/total_energy:.1f}%)")
print(f"    High-freq energy  : {high_energy:.2f} ({100*high_energy/total_energy:.1f}%)")
print(f"    WHY: Low-freq usually carries most energy (global illumination)")
print(f"         High-freq is sparser but carries structural detail.")

print(f"\n[C] Reconstruction check")
# f_low + f_high should reconstruct z0 (they're orthogonal decomposition)
reconstructed = f_low + f_high
max_error = torch.abs(reconstructed - z0).max().item()
print(f"    f_low + f_high max reconstruction error: {max_error:.2e}")
print(f"    (Should be < 1e-5 — perfect decomposition)")
assert max_error < 1e-4, f"Reconstruction error too large: {max_error}"
print("    ✓ f_low + f_high perfectly reconstructs z0")

print(f"\n[D] Statistics per component")
print(f"    f_low  — mean: {f_low.mean():.4f}  std: {f_low.std():.4f}")
print(f"    f_high — mean: {f_high.mean():.4f}  std: {f_high.std():.4f}")
print(f"    Note: f_low has higher mean (global brightness = haze bias)")
print(f"          f_high has lower mean (edges have zero mean)")

print(f"\n[E] No NaN check")
assert not torch.isnan(f_low).any(),  "NaN in f_low!"
assert not torch.isnan(f_high).any(), "NaN in f_high!"
print("    ✓ No NaN values")

# ──────────────────────────────────────────────────────────────
# Visualize the decomposition
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Step 4: FFT Frequency Decomposition\n"
             "Shows: what z0 looks like split into low and high frequencies",
             fontsize=12, fontweight='bold')

# Row 1: spatial domain views
def norm_for_viz(t):
    t = t[0, 0].numpy()
    return (t - t.min()) / (t.max() - t.min() + 1e-8)

axes[0, 0].imshow(norm_for_viz(z0), cmap='viridis')
axes[0, 0].set_title("z0 (latent, ch0)\nOriginal")
axes[0, 0].axis('off')

axes[0, 1].imshow(norm_for_viz(f_low), cmap='viridis')
axes[0, 1].set_title("f_low (ch0)\n↑ Haze distribution\n(global, smooth)")
axes[0, 1].axis('off')

axes[0, 2].imshow(norm_for_viz(f_high), cmap='RdBu_r')
axes[0, 2].set_title("f_high (ch0)\n↑ Edges/textures\n(sharp, local)")
axes[0, 2].axis('off')

axes[0, 3].imshow(norm_for_viz(f_low + f_high), cmap='viridis')
axes[0, 3].set_title("f_low + f_high\nPerfect reconstruction\n✓")
axes[0, 3].axis('off')

# Row 2: Frequency domain views
mag_np = magnitude[0, 0].numpy()
mag_norm = (mag_np - mag_np.min()) / (mag_np.max() - mag_np.min() + 1e-8)

axes[1, 0].imshow(mag_norm, cmap='hot')
axes[1, 0].set_title("Magnitude spectrum\n(freq domain view)\nCenter=low freq")
axes[1, 0].axis('off')

# Draw the low-freq mask
mask_np = mask.numpy()
axes[1, 1].imshow(mask_np, cmap='gray')
axes[1, 1].set_title("Low-freq mask\n(white = kept)\n25% of frequencies")
axes[1, 1].axis('off')

# Energy chart
labels = ['Low-freq\n(haze bias,\nglobal illum.)',
          'High-freq\n(edges,\ntextures)']
energies = [low_energy / total_energy * 100,
            high_energy / total_energy * 100]
colors = ['#2196F3', '#FF5722']
axes[1, 2].bar(labels, energies, color=colors, edgecolor='black', linewidth=0.8)
axes[1, 2].set_ylabel('Energy %')
axes[1, 2].set_title('Energy Distribution')
axes[1, 2].set_ylim(0, 100)
for i, (l, e) in enumerate(zip(labels, energies)):
    axes[1, 2].text(i, e + 1, f'{e:.1f}%', ha='center', va='bottom')

# Flow diagram
ax = axes[1, 3]
ax.text(0.5, 0.88, "WHY we need both:", ha='center', fontsize=9,
        fontweight='bold')
ax.text(0.5, 0.72, "f_low → Query: where is haze?",
        ha='center', fontsize=8, color='#1565C0',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD'))
ax.text(0.5, 0.52, "f_high → Key/Value: what to restore?",
        ha='center', fontsize=8, color='#BF360C',
        bbox=dict(boxstyle='round', facecolor='#FBE9E7'))
ax.text(0.5, 0.32, "FDAA cross-attention:", ha='center', fontsize=8,
        fontweight='bold')
ax.text(0.5, 0.15, '"Given the haze pattern (f_low),\nwhich textures (f_high) to restore?"',
        ha='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.axis('off')

plt.tight_layout()
save_path = os.path.join(OUT, "step_04_output.png")
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\n    Visualization saved: {save_path}")

# Save for next step
torch.save({'z0': z0, 'f_low': f_low, 'f_high': f_high,
            'cee': latent_data['cee']},
           os.path.join(OUT, "step_04_fft.pt"))

print("\n" + "=" * 60)
print("STEP 4 COMPLETE ✓")
print(f"\nWhat just happened:")
print(f"  z0 {list(z0.shape)} → FFT → split by frequency band")
print(f"  f_low  {list(f_low.shape)} → global illumination, haze structure")
print(f"  f_high {list(f_high.shape)} → edges, textures, fine detail")
print(f"  f_low + f_high = z0  (perfect, lossless decomposition)")
print(f"\nKey insight: Zero learnable parameters. Pure math.")
print(f"  O(n log n) complexity vs O(n²) for spatial attention.")
print(f"\nNext: python step_05_fdaa.py")
print("=" * 60)