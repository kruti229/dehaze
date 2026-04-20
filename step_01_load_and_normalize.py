"""
STEP 1 — LOAD IMAGE & NORMALIZE
=================================

WHY THIS STEP EXISTS:
  Neural networks work with floating point tensors in [0, 1] range,
  NOT with uint8 pixel values (0–255). This step converts the raw
  image file into the exact format every downstream module expects.

  Also: we load BOTH the hazy image (LQ = Low Quality) and the
  ground truth clean image (GT). During training, GT is needed to
  compute loss. During inference, only LQ is needed.

WHAT HAPPENS:
  PNG/JPG file  →  numpy uint8 [H, W, 3]  →  float32 [0,1]  →  tensor [1, 3, H, W]
  BGR (OpenCV)  →  RGB (what the network expects)

VERIFICATION:
  - Shape must be [1, 3, H, W]
  - Values must be in [0.0, 1.0]
  - No NaN or Inf values
"""

import os, sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs/step_outputs")

print("=" * 60)
print("STEP 1 — Load Image & Normalize")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# The actual loader function (used by training dataloader too)
# ──────────────────────────────────────────────────────────────
def load_image_as_tensor(image_path: str,
                          patch_size: int = None,
                          device: str = "cpu") -> torch.Tensor:
    """
    Load an image file and return a normalized tensor.

    Args:
        image_path  : path to .png or .jpg
        patch_size  : if set, randomly crop to patch_size × patch_size
                      (used during training — DehazeDiff uses 256×256 for
                       SOTS and 1024×1024 for NH-Haze / Dense-Haze)
        device      : 'cpu' or 'cuda'

    Returns:
        tensor: float32 [1, 3, H, W] in [0, 1]
    """
    # 1. Read with OpenCV (loads as BGR uint8)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # 2. BGR → RGB
    #    WHY: OpenCV reads BGR by default, but PyTorch/timm expect RGB.
    #    Getting this wrong flips red/blue channels — model sees wrong colors.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Optional: crop a random patch
    #    WHY: NH-Haze images are 1200×1600. Training on full images
    #    requires huge VRAM. We crop 1024×1024 patches.
    #    At test time we use the full image.
    if patch_size is not None:
        H, W = img_rgb.shape[:2]
        assert H >= patch_size and W >= patch_size, \
            f"Image {H}×{W} smaller than patch {patch_size}"
        top  = np.random.randint(0, H - patch_size + 1)
        left = np.random.randint(0, W - patch_size + 1)
        img_rgb = img_rgb[top:top+patch_size, left:left+patch_size]

    # 4. uint8 [0,255] → float32 [0,1]
    #    WHY: Neural nets expect values in [0,1] or [-1,1].
    #    [0,1] is standard for image restoration tasks.
    img_float = img_rgb.astype(np.float32) / 255.0

    # 5. numpy [H, W, C] → torch [1, C, H, W]
    #    WHY: PyTorch conv layers expect [Batch, Channels, Height, Width].
    #    The batch dimension (1) is added because everything in the
    #    pipeline processes batches, even during inference with 1 image.
    tensor = torch.from_numpy(img_float)          # [H, W, C]
    tensor = tensor.permute(2, 0, 1)              # [C, H, W]
    tensor = tensor.unsqueeze(0)                   # [1, C, H, W]
    tensor = tensor.to(device)

    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse: tensor [1, 3, H, W] float32 → numpy [H, W, 3] uint8
    Used for saving output images.
    """
    img = tensor.squeeze(0).permute(1, 2, 0)      # [H, W, C]
    img = img.clamp(0, 1)                          # clip to valid range
    img = (img.cpu().numpy() * 255).astype(np.uint8)
    return img


# ──────────────────────────────────────────────────────────────
# Run the loader on our test image
# ──────────────────────────────────────────────────────────────
hazy_path = os.path.join(OUT, "test_hazy.png")
if not os.path.exists(hazy_path):
    print("Test image not found. Run step_00_verify_env.py first.")
    sys.exit(1)

print(f"\nLoading: {hazy_path}")
lq_tensor = load_image_as_tensor(hazy_path, patch_size=None, device="cpu")

# ──────────────────────────────────────────────────────────────
# VERIFY everything about the tensor
# ──────────────────────────────────────────────────────────────
print("\n── VERIFICATION ──────────────────────────────────────")

print(f"\n[A] Shape check")
print(f"    tensor.shape = {list(lq_tensor.shape)}")
print(f"    Expected     : [1, 3, H, W]")
assert lq_tensor.ndim == 4,      "Must be 4D tensor [B, C, H, W]"
assert lq_tensor.shape[0] == 1,  "Batch size should be 1 for single image"
assert lq_tensor.shape[1] == 3,  "Must have 3 channels (RGB)"
print("    ✓ Shape correct")

print(f"\n[B] Value range check")
print(f"    Min value : {lq_tensor.min().item():.4f}  (expected ≥ 0.0)")
print(f"    Max value : {lq_tensor.max().item():.4f}  (expected ≤ 1.0)")
print(f"    Mean value: {lq_tensor.mean().item():.4f}")
assert lq_tensor.min() >= 0.0, "Values below 0 found!"
assert lq_tensor.max() <= 1.0, "Values above 1 found!"
print("    ✓ Values in [0, 1]")

print(f"\n[C] NaN / Inf check")
assert not torch.isnan(lq_tensor).any(), "NaN values found!"
assert not torch.isinf(lq_tensor).any(), "Inf values found!"
print("    ✓ No NaN or Inf")

print(f"\n[D] dtype check")
print(f"    dtype : {lq_tensor.dtype}  (expected torch.float32)")
assert lq_tensor.dtype == torch.float32, "Must be float32!"
print("    ✓ dtype correct")

print(f"\n[E] Channel statistics (R, G, B separately)")
for i, ch in enumerate(['R', 'G', 'B']):
    ch_data = lq_tensor[0, i]
    print(f"    {ch}: mean={ch_data.mean():.3f}  std={ch_data.std():.3f}  "
          f"min={ch_data.min():.3f}  max={ch_data.max():.3f}")

# ──────────────────────────────────────────────────────────────
# Simulate a TRAINING BATCH (random patch crop)
# ──────────────────────────────────────────────────────────────
print(f"\n[F] Training patch simulation (patch_size=128)")
patch_tensor = load_image_as_tensor(hazy_path, patch_size=128, device="cpu")
print(f"    Full image tensor : {list(lq_tensor.shape)}")
print(f"    Cropped patch     : {list(patch_tensor.shape)}")
print("    WHY: Training always uses fixed patch sizes for memory")
print("         efficiency and batch consistency.")
print("    ✓ Patch cropping works")

# ──────────────────────────────────────────────────────────────
# Save a visualization
# ──────────────────────────────────────────────────────────────
print(f"\n[G] Saving visualization...")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Step 1: Image Loading & Normalization", fontsize=13, fontweight='bold')

# Original hazy
img_orig = tensor_to_image(lq_tensor)
axes[0].imshow(img_orig)
axes[0].set_title("Hazy Input (LQ)")
axes[0].axis('off')

# Individual channels
for idx, (ch_name, color) in enumerate(zip(['Red', 'Green', 'Blue'],
                                            ['Reds', 'Greens', 'Blues'])):
    ch = lq_tensor[0, idx].numpy()
    axes[1].hist(ch.flatten(), bins=50, alpha=0.5, label=ch_name,
                 color=['red', 'green', 'blue'][idx])
axes[1].set_title("Pixel Value Distribution\n(all 3 channels)")
axes[1].set_xlabel("Value (0=black, 1=white)")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].set_xlim(0, 1)

# Annotate what goes next
axes[2].text(0.5, 0.7, "Tensor shape:\n[1, 3, H, W]",
             ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='lightblue'))
axes[2].text(0.5, 0.35, "dtype: float32\nrange: [0, 1]\ndevice: cpu/cuda",
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow'))
axes[2].text(0.5, 0.1, "→ Next: ConvNext Encoder",
             ha='center', va='center', fontsize=10, color='darkgreen')
axes[2].axis('off')
axes[2].set_title("Ready for Network")

plt.tight_layout()
save_path = os.path.join(OUT, "step_01_output.png")
plt.savefig(save_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"    Saved: {save_path}")

# Save tensor for next step
torch.save(lq_tensor, os.path.join(OUT, "step_01_lq_tensor.pt"))
print(f"    Tensor saved for next step.")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE ✓")
print(f"\nWhat just happened:")
print(f"  PNG file  →  [1, 3, {lq_tensor.shape[2]}, {lq_tensor.shape[3]}] float32 tensor")
print(f"  BGR → RGB conversion applied")
print(f"  Values normalized from [0,255] → [0.0, 1.0]")
print(f"\nNext: python step_02_convnext_encoder.py")
print("=" * 60)