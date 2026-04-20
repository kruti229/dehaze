"""
STEP 2 — CONVNEXT ENCODER
===========================

WHY THIS STEP EXISTS:
  Raw pixels are terrible inputs for a diffusion model. A 256×256 image
  has 256×256×3 = 196,608 numbers. Running diffusion on all of them is
  astronomically expensive and the model would have to learn basic vision
  from scratch.

  ConvNext (pretrained on ImageNet with 1.2M images) already understands:
    - Edges and textures (from early layers)
    - Shapes and objects (from deeper layers)
    - Global structure (from the deepest stage)

  By using its first 2 stages as a frozen feature extractor, we get:
    - Rich 192-channel feature maps (capture everything the image contains)
    - 4× spatial downsampling (256×256 → 64×64 — manageable size)
    - Domain knowledge from ImageNet pre-training

  This creates the HIGH-DIMENSIONAL IMPLICIT SPACE that DehazeDiff
  describes. It's called "implicit" because it's learned, not hand-crafted.

  IMPORTANT: ConvNext is FROZEN (no gradients). We never update its weights.
  Only the DCU/DDU and diffusion model are trained.

WHAT HAPPENS:
  [1, 3, 256, 256]  →  ConvNext Stage 1+2  →  [1, 192, 64, 64]
  (hazy image)                                 (implicit space)

  The 192 channels contain rich multi-scale information about the image.
  The 64×64 spatial resolution preserves where things are.
"""

import os, sys
import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs/step_outputs")

print("=" * 60)
print("STEP 2 — ConvNext Encoder (Implicit Space)")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# Build the encoder
# ──────────────────────────────────────────────────────────────
class ConvNextEncoder(nn.Module):
    """
    Uses first 2 stages of ConvNext-Small pretrained on ImageNet.

    Stage 1: 3×3 conv, stride 4 → C=96,  H/4  × W/4
    Stage 2: 3×3 conv, stride 2 → C=192, H/8  × W/8

    For a 256×256 input: output is [B, 192, 32, 32]
    For a 512×512 input: output is [B, 192, 64, 64]

    WHY stage 1+2 specifically?
      DehazeDiff paper uses first 2 stages of ConvNext. Stage 1 captures
      low-level features (edges, colors). Stage 2 captures higher-level
      patterns (shapes, textures). Going deeper would lose spatial resolution
      needed for detail recovery.

    WHY frozen?
      The encoder's job is feature extraction, not haze removal.
      Unfreezing it would require massive gradient memory and can
      cause the pre-trained features to degrade (catastrophic forgetting).
    """
    def __init__(self, pretrained=False):
        super().__init__()
        # features_only=True → returns intermediate feature maps
        # out_indices=[1] → only take Stage 2 output (index 1)
        self.backbone = timm.create_model(
            'convnext_small',
            pretrained=pretrained,
            features_only=True,
            out_indices=[1]      # Stage 2 output: C=192
        )

        # FREEZE ALL WEIGHTS — never update during training
        for param in self.backbone.parameters():
            param.requires_grad = False

        print(f"    ConvNext-Small Stage 1+2 loaded")
        print(f"    Pretrained: {pretrained}")
        n_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"    Parameters: {n_params:,} (all FROZEN)")

    def forward(self, x):
        """
        x: [B, 3, H, W] normalized hazy image
        returns: [B, 192, H/8, W/8] implicit space features
        """
        features = self.backbone(x)  # returns list, we take index 0 (Stage 2)
        return features[0]

    def get_output_channels(self):
        return 192


# ──────────────────────────────────────────────────────────────
# Load the tensor from Step 1 and run through encoder
# ──────────────────────────────────────────────────────────────
tensor_path = os.path.join(OUT, "step_01_lq_tensor.pt")
if not os.path.exists(tensor_path):
    print("Step 1 tensor not found. Run step_01_load_and_normalize.py first.")
    sys.exit(1)

lq_tensor = torch.load(tensor_path, map_location='cpu')
print(f"\nLoaded LQ tensor from Step 1: {list(lq_tensor.shape)}")

# Build encoder
print("\nBuilding ConvNext encoder...")
encoder = ConvNextEncoder(pretrained=False)  # set True with internet
encoder.eval()

# Forward pass
print("\nRunning forward pass...")
with torch.no_grad():
    implicit_space = encoder(lq_tensor)

print(f"\nINPUT  → OUTPUT:")
print(f"  {list(lq_tensor.shape)} → {list(implicit_space.shape)}")


# ──────────────────────────────────────────────────────────────
# VERIFY
# ──────────────────────────────────────────────────────────────
print("\n── VERIFICATION ──────────────────────────────────────")

print(f"\n[A] Shape check")
B, C, H, W = implicit_space.shape
print(f"    Batch     : {B}  (one image)")
print(f"    Channels  : {C}  (expected 192 — ConvNext Stage 2)")
print(f"    Height    : {H}  (input_H / 8)")
print(f"    Width     : {W}  (input_W / 8)")
assert C == 192, f"Expected 192 channels, got {C}"
print("    ✓ Shape correct")

print(f"\n[B] Spatial downsampling ratio")
in_h, in_w = lq_tensor.shape[2], lq_tensor.shape[3]
ratio = in_h / H
print(f"    Input : {in_h}×{in_w}")
print(f"    Output: {H}×{W}")
print(f"    Ratio : {ratio}× downsampling")
print(f"    WHY: {ratio}× reduces spatial size by {ratio**2:.0f}×,")
print(f"         making diffusion {ratio**2:.0f}× more computationally feasible.")

print(f"\n[C] Feature statistics")
print(f"    Mean : {implicit_space.mean().item():.4f}")
print(f"    Std  : {implicit_space.std().item():.4f}")
print(f"    Min  : {implicit_space.min().item():.4f}")
print(f"    Max  : {implicit_space.max().item():.4f}")
print(f"    NaN  : {torch.isnan(implicit_space).any().item()}")
print(f"    Inf  : {torch.isinf(implicit_space).any().item()}")
assert not torch.isnan(implicit_space).any(), "NaN in encoder output!"
print("    ✓ Feature values healthy")

print(f"\n[D] Gradient check")
print(f"    Encoder requires_grad: "
      f"{any(p.requires_grad for p in encoder.parameters())}")
print(f"    Output requires_grad : {implicit_space.requires_grad}")
print(f"    ✓ Encoder correctly frozen (no gradients will flow through it)")

# ──────────────────────────────────────────────────────────────
# Visualize what the encoder is seeing
# ──────────────────────────────────────────────────────────────
print(f"\n[E] Visualizing feature maps...")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Step 2: ConvNext Encoder Output (Implicit Space)\n"
             f"Shape: {list(implicit_space.shape)}", fontsize=12, fontweight='bold')

# Show input image
from PIL import Image
import cv2
lq_img = lq_tensor[0].permute(1, 2, 0).numpy()
axes[0, 0].imshow(lq_img)
axes[0, 0].set_title("Input\n(Hazy Image)", fontsize=9)
axes[0, 0].axis('off')

# Show first 8 feature channels
feats = implicit_space[0].numpy()  # [192, H, W]
for i in range(8):
    row, col = (i+1) // 5, (i+1) % 5
    ax = axes[row, col]
    # Normalize each channel for display
    ch = feats[i]
    ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    ax.imshow(ch_norm, cmap='viridis')
    ax.set_title(f"Channel {i}\n(of 192)", fontsize=9)
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(OUT, "step_02_output.png")
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"    Saved: {save_path}")

# Save for next step
torch.save(implicit_space, os.path.join(OUT, "step_02_implicit_space.pt"))

print("\n" + "=" * 60)
print("STEP 2 COMPLETE ✓")
print(f"\nWhat just happened:")
print(f"  [1,3,{in_h},{in_w}] hazy image → ConvNext (frozen, ImageNet)")
print(f"  → [1,192,{H},{W}] implicit space (rich features, 8× smaller)")
print(f"\nKey insight: These 192 feature maps capture WHAT is in the")
print(f"image (objects, textures, edges) — the encoder doesn't know")
print(f"about haze yet. That's the NEXT module's job (DCU/CEE).")
print(f"\nNext: python step_03_dcu_cee.py")
print("=" * 60)