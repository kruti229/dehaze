"""
STEP 0 — VERIFY ENVIRONMENT
============================
Run this FIRST before anything else.
Checks every package, every path, every import.
If this passes 100%, the rest of the pipeline will run.
"""

import sys, os

print("=" * 60)
print("AILD-Freq Pipeline — Step 0: Environment Verification")
print("=" * 60)

# ── 1. Python version ──────────────────────────────────────────
print("\n[1] Python version")
print(f"    Python {sys.version}")
assert sys.version_info >= (3, 8), "Need Python 3.8+"
print("    ✓ OK")

# ── 2. PyTorch + CUDA ─────────────────────────────────────────
print("\n[2] PyTorch + device")
import torch
print(f"    PyTorch version : {torch.__version__}")
if torch.cuda.is_available():
    print(f"    GPU             : {torch.cuda.get_device_name(0)}")
    print(f"    VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = "cuda"
else:
    print("    GPU             : NOT AVAILABLE — running on CPU")
    print("    WARNING: Training will be extremely slow on CPU.")
    print("    For testing the pipeline structure this is fine.")
    DEVICE = "cpu"
print(f"    Using device    : {DEVICE}")
print("    ✓ OK")

# ── 3. Core packages ──────────────────────────────────────────
print("\n[3] Required packages")
packages = {
    "torch.fft"    : "FFT decomposition in FDAA",
    "timm"         : "ConvNext encoder (ImageNet pretrained)",
    "einops"       : "Tensor reshaping in attention blocks",
    "lpips"        : "Perceptual loss evaluation metric",
    "cv2"          : "Image loading/saving (opencv-python)",
    "PIL"          : "Image processing (Pillow)",
    "numpy"        : "Array operations",
    "matplotlib"   : "Visualization of intermediate features",
    "tqdm"         : "Progress bars during training",
    "yaml"         : "Config file parsing",
}

failed = []
for pkg, reason in packages.items():
    try:
        __import__(pkg)
        print(f"    ✓ {pkg:<20} ({reason})")
    except ImportError:
        print(f"    ✗ {pkg:<20} MISSING — needed for: {reason}")
        failed.append(pkg)

if failed:
    print(f"\n  INSTALL MISSING: pip install {' '.join(failed)}")
    print("  For opencv: pip install opencv-python")
    sys.exit(1)

# ── 4. torch.fft specific test ───────────────────────────────
print("\n[4] FFT functionality test")
# This is critical — FDAA depends on torch.fft.fft2
x = torch.randn(2, 16, 32, 32)                 # [B, C, H, W]
fft_out = torch.fft.fft2(x, norm='ortho')       # complex output
shifted = torch.fft.fftshift(fft_out)           # shift zero-freq to center
back = torch.fft.ifft2(
    torch.fft.ifftshift(shifted), norm='ortho') # back to spatial
diff = torch.abs(x - back.real).max().item()
print(f"    FFT → shift → ishift → IFFT max reconstruction error: {diff:.2e}")
assert diff < 1e-5, "FFT round-trip error too large"
print("    ✓ FFT round-trip verified")

# ── 5. timm ConvNext test ─────────────────────────────────────
print("\n[5] timm ConvNext encoder test")
import timm
# List available convnext models
models = [m for m in timm.list_models() if 'convnext' in m.lower()]
print(f"    Available ConvNext variants: {len(models)}")
# We use convnext_small — takes ImageNet pretrained weights
# first 2 stages output feature map at stride 8 (C=192)
model = timm.create_model(
    'convnext_small',
    pretrained=False,   # set True when you have internet
    features_only=True,
    out_indices=[1]     # Stage 2 output (stride 8, C=192)
)
dummy = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    feats = model(dummy)
print(f"    Input shape      : {list(dummy.shape)}")
print(f"    Encoder output   : {list(feats[0].shape)}")
print(f"    WHY: ConvNext was pretrained on ImageNet — its features")
print(f"         capture rich texture/structure priors that help the")
print(f"         diffusion model focus on haze removal, not learning")
print(f"         basic image features from scratch.")
print("    ✓ ConvNext encoder works")

# ── 6. Directory structure ────────────────────────────────────
print("\n[6] Project directory structure")
dirs = [
    "models/modules",
    "data",
    "checkpoints",
    "outputs/step_outputs",
    "configs",
    "steps",
]
base = os.path.dirname(os.path.abspath(__file__))
for d in dirs:
    path = os.path.join(base, d)
    os.makedirs(path, exist_ok=True)
    print(f"    ✓ {path}")

# ── 7. Create a synthetic hazy test image ─────────────────────
print("\n[7] Create synthetic test image (if no real data yet)")
import numpy as np
from PIL import Image

test_img_path = os.path.join(base, "outputs/step_outputs/test_hazy.png")
if not os.path.exists(test_img_path):
    # Create a synthetic 'hazy' image: sharp image + white haze overlay
    np.random.seed(42)
    # Base image: scene with structure
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Background gradient (sky-like)
    for i in range(256):
        img[i, :] = [max(50, 200 - i//2), max(80, 180 - i//3), 200]
    # Add some objects
    img[80:140, 60:120] = [100, 80, 60]    # building 1
    img[90:150, 140:200] = [120, 100, 80]  # building 2
    # Add haze: blend toward white
    haze = np.ones_like(img) * 220
    haze_density = 0.5
    hazy_img = (img * (1 - haze_density) + haze * haze_density).astype(np.uint8)
    Image.fromarray(hazy_img).save(test_img_path)
    print(f"    Created synthetic hazy test image: {test_img_path}")
else:
    print(f"    Test image already exists: {test_img_path}")

# ── 8. Write device config for all steps ─────────────────────
config_path = os.path.join(base, "outputs/step_outputs/device.txt")
with open(config_path, "w") as f:
    f.write(DEVICE)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED ✓")
print(f"Device selected: {DEVICE}")
print("\nNext step: python step_01_load_and_normalize.py")
print("=" * 60)