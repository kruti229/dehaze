import os
import argparse
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import glob

from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")



def pick_first_image(folder):
    if folder is None or (not os.path.isdir(folder)):
        return None
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)
    return files[0] if files else None
def auto_pick_paths_from_config(config_path, input_path=None, gt_path=None):
    opt = yaml.safe_load(open(config_path, "r"))
    # prefer test split, fallback to val, then train
    split_order = ["test", "val", "train"]
    if input_path is None:
        for s in split_order:
            lq_root = opt.get("datasets", {}).get(s, {}).get("dataroot_LQ", None)
            input_path = pick_first_image(lq_root)
            if input_path:
                print(f"[auto] picked input from {s}: {input_path}")
                break
    if gt_path is None:
        for s in split_order:
            gt_root = opt.get("datasets", {}).get(s, {}).get("dataroot_GT", None)
            gt_path = pick_first_image(gt_root)
            if gt_path:
                print(f"[auto] picked gt from {s}: {gt_path}")
                break
    return input_path, gt_path

def load_img_as_tensor(path, device):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return t


def tensor_to_rgb_uint8(t):
    x = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0).round().astype(np.uint8)


def save_rgb(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def psnr_torch(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_torch(pred, target):
    # Simple global SSIM (fast, dependency-free)
    C1, C2 = 0.01**2, 0.03**2
    x = pred.clamp(0, 1)
    y = target.clamp(0, 1)

    mu_x = x.mean(dim=[2, 3], keepdim=True)
    mu_y = y.mean(dim=[2, 3], keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=[2, 3], keepdim=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean().item()


@torch.no_grad()
def run_stage1_infer(config_path, ckpt_path, input_path, output_path, compare_path, gt_path, device):
    _ = yaml.safe_load(open(config_path, "r"))  # kept for consistency/repro

    encoder = ConvNextEncoder(pretrained=True).to(device).eval()
    codec = DehazeDiffCodec(192, 16, 64).to(device).eval()
    ddu = DimensionalDecompressionUnit(16, 192, 64).to(device).eval()
    decoder = PixelShuffleDecoder(in_channels=192, upscale_factor=8).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    codec.load_state_dict(ckpt["codec"])
    ddu.load_state_dict(ckpt["ddu"])
    decoder.load_state_dict(ckpt["decoder"])

    lq = load_img_as_tensor(input_path, device)

    # Stage-1 forward
    impl = encoder(lq)
    z, cee = codec.encode(impl)
    impl_rec = ddu(z, cee)
    pred = decoder(impl_rec).clamp(0, 1)

    # Save result image
    pred_rgb = tensor_to_rgb_uint8(pred)
    save_rgb(output_path, pred_rgb)

    # Metrics / comparison
    gt = None
    if gt_path:
        gt = load_img_as_tensor(gt_path, device)
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)

    # A and B clarification:
    # A (important): Stage-1 branch A reconstruction loss proxy at inference => L1(pred, input_hazy)
    # B (important only if GT exists): Stage-1 branch B style proxy => L1(pred, GT)
    A_l1 = F.l1_loss(pred, lq).item()
    print(f"A_l1 (pred vs input_hazy): {A_l1:.6f}")

    if gt is not None:
        B_l1 = F.l1_loss(pred, gt).item()
        psnr = psnr_torch(pred, gt)
        ssim = ssim_torch(pred, gt)
        print(f"B_l1 (pred vs GT): {B_l1:.6f}")
        print(f"PSNR: {psnr:.4f} dB")
        print(f"SSIM: {ssim:.4f}")
    else:
        print("GT not provided -> PSNR/SSIM and B_l1 skipped.")

    # Build comparison image
    in_rgb = tensor_to_rgb_uint8(lq)
    if gt is None:
        # [Input | Output]
        comp = np.concatenate([in_rgb, pred_rgb], axis=1)
        cv2.putText(comp, "Input", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(comp, "Output", (in_rgb.shape[1] + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        gt_rgb = tensor_to_rgb_uint8(gt)
        # [Input | Output | GT]
        comp = np.concatenate([in_rgb, pred_rgb, gt_rgb], axis=1)
        w = in_rgb.shape[1]
        cv2.putText(comp, "Input", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(comp, "Output", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(comp, "GT", (2 * w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    save_rgb(compare_path, comp)

    print("=" * 60)
    print("Stage-1 inference done")
    print(f"Input      : {input_path}")
    print(f"Output     : {output_path}")
    print(f"Comparison : {compare_path}")
    print(f"Checkpoint : {ckpt_path}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", default=None, help="Optional; auto-picked from config if missing")
    ap.add_argument("--output", default="outputs/stage1_result.png")
    ap.add_argument("--compare", default="outputs/stage1_compare.png")
    ap.add_argument("--gt", default=None, help="Optional; auto-picked from config if missing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_stage1_infer(
        config_path=args.config,
        ckpt_path=args.ckpt,
        input_path=args.input,
        output_path=args.output,
        compare_path=args.compare,
        gt_path=args.gt,
        device=args.device,
    )


if __name__ == "__main__":
    main()