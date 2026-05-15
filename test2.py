import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.lqgt_dataset import LQGTDataset

from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.fdaa import FDAA
from models.modules.hag import HAG
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder
from models.modules.physical_guidance import TransmissionEstimator, PhysicalContextFusion

from utils.checkpoint_utils import load_checkpoint
from utils.img_utils import normalize_for_encoder
from utils.metrics import compute_psnr, compute_ssim

from models.modules.latent_translator import LatentTranslator



class LatentTranslator(nn.Module):
    """
    Same Stage-2 translator used in train_stage2.py.

    Input:
        z_h + c_h_guided + t_h

    Output:
        z_pred = clean-like latent
    """

    def __init__(
        self,
        latent_ch=16,
        cee_ch=64,
        width=64,
        num_blocks=6,
        residual_scale=0.5,
    ):
        super().__init__()

        self.residual_scale = residual_scale
        in_ch = latent_ch + cee_ch + 1

        self.intro = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GroupNorm(8, width),
                )
                for _ in range(num_blocks)
            ]
        )

        self.out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(width, latent_ch, 3, padding=1),
        )

    def forward(self, z_h, c_h, t_h):
        if c_h.shape[-2:] != z_h.shape[-2:]:
            c_h = F.interpolate(
                c_h,
                size=z_h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if t_h.shape[-2:] != z_h.shape[-2:]:
            t_h = F.interpolate(
                t_h,
                size=z_h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([z_h, c_h, t_h], dim=1)
        x = self.intro(x)

        for block in self.blocks:
            x = x + block(x)

        delta = self.out(x)
        z_pred = z_h + self.residual_scale * torch.tanh(delta)

        return z_pred, delta


def build_dataloader(opt, split="test"):
    ds = LQGTDataset(opt, split)

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(opt["datasets"][split].get("n_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return ds, dl


def build_modules(opt, device):
    encoder = ConvNextEncoder(
        pretrained=bool(opt["network_encoder"].get("pretrained", True)),
        backbone=opt["network_encoder"].get("backbone", "convnext_small"),
    ).to(device)

    impl_ch = int(opt["network_L"]["implicit_channels"])
    lat_ch = int(opt["network_L"]["latent_channels"])
    cee_ch = int(opt["network_L"]["cee_channels"])

    codec = DehazeDiffCodec(
        impl_ch=impl_ch,
        lat_ch=lat_ch,
        cee_ch=cee_ch,
    ).to(device)

    ddu = DimensionalDecompressionUnit(
        in_ch=lat_ch,
        out_ch=impl_ch,
        cee_ch=cee_ch,
    ).to(device)

    decoder = PixelShuffleDecoder(
        in_channels=impl_ch,
        upscale_factor=8,
    ).to(device)

    fdaa_cfg = opt.get("fdaa", {})
    fdaa = FDAA(
        channels=int(fdaa_cfg.get("channels", lat_ch)),
        num_heads=int(fdaa_cfg.get("num_heads", 4)),
        low_freq_ratio=float(fdaa_cfg.get("low_freq_ratio", 0.25)),
    ).to(device)

    hag_cfg = opt.get("hag", {})
    hag = HAG(
        t_max=int(hag_cfg.get("t_max", 16)),
        min_steps=int(hag_cfg.get("min_steps", 4)),
    ).to(device)

    physical_opt = opt.get("physical", {})
    phys_t = TransmissionEstimator(
        base_ch=int(physical_opt.get("base_ch", 32)),
        t_min=float(physical_opt.get("t_min", 0.05)),
    ).to(device)

    phys_fusion = PhysicalContextFusion(
        cee_ch=cee_ch,
    ).to(device)

    st2 = opt.get("stage2_translator", {})
    stage2_mapper = LatentTranslator(
        latent_ch=lat_ch,
        cee_ch=cee_ch,
        width=int(st2.get("width", 64)),
        num_blocks=int(st2.get("num_blocks", 6)),
        residual_scale=float(st2.get("residual_scale", 0.5)),
    ).to(device)

    return encoder, codec, fdaa, hag, ddu, decoder, phys_t, phys_fusion, stage2_mapper


def match_spatial(pred, target):
    if pred.shape[-2:] == target.shape[-2:]:
        return pred, target

    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])

    return pred[..., :h, :w], target[..., :h, :w]


def reconstruct_from_latent(z, c, ddu, decoder):
    return decoder(ddu(z, c)).clamp(0.0, 1.0)


def tensor_to_rgb_uint8(t):
    if t.dim() == 4:
        t = t[0]

    if t.shape[0] != 3:
        raise ValueError(f"Expected 3-channel tensor, got shape {tuple(t.shape)}")

    t = t.detach().float().clamp(0.0, 1.0).cpu()
    rgb = t.permute(1, 2, 0).numpy()
    rgb = np.round(rgb * 255.0).astype(np.uint8)

    return rgb


def save_rgb(path, rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {rgb.shape}")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr)

    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def crop_np_to_match(*imgs):
    h = min(img.shape[0] for img in imgs)
    w = min(img.shape[1] for img in imgs)

    return [img[:h, :w] for img in imgs]


def color_diagnostic(pred_eval):
    r = pred_eval[:, 0:1]
    g = pred_eval[:, 1:2]
    b = pred_eval[:, 2:3]

    rg_diff = float((r - g).abs().mean().item())
    rb_diff = float((r - b).abs().mean().item())
    gb_diff = float((g - b).abs().mean().item())

    chroma = (rg_diff + rb_diff + gb_diff) / 3.0
    is_gray_like = rg_diff < 0.01 and rb_diff < 0.01 and gb_diff < 0.01

    return rg_diff, rb_diff, gb_diff, chroma, is_gray_like


def restore_color_ycbcr_tensor(
    pred_eval,
    lq_eval,
    min_chroma=0.08,
    chroma_blend=0.70,
    luma_blend=0.10,
):
    """
    Visual fallback only.
    Does not use GT.
    Keeps predicted luminance mostly intact and borrows chroma from hazy input
    only when predicted chroma is too weak.
    """

    pred = pred_eval[0].detach().float().clamp(0.0, 1.0)
    hazy = lq_eval[0].detach().float().clamp(0.0, 1.0)

    def rgb_to_ycbcr_img(x):
        r = x[0:1]
        g = x[1:2]
        b = x[2:3]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

        return torch.cat([y, cb, cr], dim=0)

    def ycbcr_to_rgb_img(x):
        y = x[0:1]
        cb = x[1:2] - 0.5
        cr = x[2:3] - 0.5

        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb

        return torch.cat([r, g, b], dim=0).clamp(0.0, 1.0)

    def chroma_strength(x):
        r = x[0:1]
        g = x[1:2]
        b = x[2:3]
        return ((r - g).abs().mean() + (g - b).abs().mean() + (b - r).abs().mean()) / 3.0

    pred_chroma = float(chroma_strength(pred).item())

    if pred_chroma >= min_chroma:
        return pred_eval, False, pred_chroma

    pred_ycc = rgb_to_ycbcr_img(pred)
    hazy_ycc = rgb_to_ycbcr_img(hazy)

    y = (1.0 - luma_blend) * pred_ycc[0:1] + luma_blend * hazy_ycc[0:1]
    cb = chroma_blend * hazy_ycc[1:2] + (1.0 - chroma_blend) * pred_ycc[1:2]
    cr = chroma_blend * hazy_ycc[2:3] + (1.0 - chroma_blend) * pred_ycc[2:3]

    out_ycc = torch.cat([y, cb, cr], dim=0)
    out_rgb = ycbcr_to_rgb_img(out_ycc)

    out = out_rgb.unsqueeze(0).to(pred_eval.device)
    return out.clamp(0.0, 1.0), True, pred_chroma


def make_compare_panel(
    hazy_rgb,
    pred_rgb,
    gt_rgb,
    filename,
    psnr,
    ssim,
    score,
    input_psnr,
    input_ssim,
    mode,
    rg_diff,
    rb_diff,
    gb_diff,
    pred_chroma,
    is_gray_like,
    color_fix_used,
    color_fix_applied,
    delta_abs,
):
    hazy_rgb, pred_rgb, gt_rgb = crop_np_to_match(
        hazy_rgb,
        pred_rgb,
        gt_rgb,
    )

    h, w, _ = pred_rgb.shape

    body = np.concatenate(
        [hazy_rgb, pred_rgb, gt_rgb],
        axis=1,
    )

    header_h = max(150, int(0.10 * h))
    header = np.ones((header_h, body.shape[1], 3), dtype=np.uint8) * 255

    panel = np.concatenate([header, body], axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.70, w / 950.0)
    small_scale = scale * 0.56
    thickness = max(2, int(w / 750))

    black = (0, 0, 0)
    blue = (30, 60, 180)
    red = (180, 40, 40)

    y1 = int(header_h * 0.22)
    y2 = int(header_h * 0.45)
    y3 = int(header_h * 0.66)
    y4 = int(header_h * 0.86)

    x_hazy = 20
    x_pred = w + 20
    x_gt = 2 * w + 20

    cv2.putText(panel, "Hazy Input", (x_hazy, y1), font, scale, black, thickness, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"Input PSNR: {input_psnr:.3f} | SSIM: {input_ssim:.4f}",
        (x_hazy, y2),
        font,
        small_scale,
        black,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )

    cv2.putText(panel, "Dehazed Output", (x_pred, y1), font, scale, blue, thickness, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"PSNR: {psnr:.3f} | SSIM: {ssim:.4f} | Score: {score:.3f}",
        (x_pred, y2),
        font,
        small_scale,
        blue,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"RGBdiff: {rg_diff:.4f}/{rb_diff:.4f}/{gb_diff:.4f} | Chroma: {pred_chroma:.4f}",
        (x_pred, y3),
        font,
        small_scale,
        blue,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"GrayLike: {is_gray_like} | ColorFix: {color_fix_used}/{color_fix_applied} | Delta: {delta_abs:.5f}",
        (x_pred, y4),
        font,
        small_scale,
        blue,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )

    cv2.putText(panel, "Ground Truth", (x_gt, y1), font, scale, red, thickness, cv2.LINE_AA)
    cv2.putText(
        panel,
        filename,
        (x_gt, y2),
        font,
        small_scale,
        red,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        mode,
        (x_gt, y3),
        font,
        small_scale,
        red,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )

    return panel


@torch.no_grad()
def run_test(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = opt["name"]
    ckpt_dir = os.path.join("checkpoints", dataset_name)

    final_path = os.path.join(ckpt_dir, "final_best.pth")
    stage1_path = os.path.join(ckpt_dir, "stage1_best.pth")

    if os.path.exists(final_path):
        ckpt_path = final_path
    else:
        ckpt_path = stage1_path

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found. Expected either {final_path} or {stage1_path}"
        )

    _, test_dl = build_dataloader(opt, "test")

    (
        encoder,
        codec,
        fdaa,
        hag,
        ddu,
        decoder,
        phys_t,
        phys_fusion,
        stage2_mapper,
    ) = build_modules(opt, device)

    ckpt = load_checkpoint(ckpt_path, device)

    codec.load_state_dict(ckpt["codec"])
    ddu.load_state_dict(ckpt["ddu"])
    decoder.load_state_dict(ckpt["decoder"])

    if "fdaa" in ckpt:
        fdaa.load_state_dict(ckpt["fdaa"])

    if "hag" in ckpt:
        hag.load_state_dict(ckpt["hag"])

    if "phys_t" in ckpt:
        phys_t.load_state_dict(ckpt["phys_t"])

    if "phys_fusion" in ckpt:
        missing, unexpected = phys_fusion.load_state_dict(ckpt["phys_fusion"], strict=False)
        if missing:
            print(f"[Load] phys_fusion missing keys initialized randomly: {missing}")
        if unexpected:
            print(f"[Load] phys_fusion unexpected keys ignored: {unexpected}")

    if "stage2_mapper" in ckpt:
        stage2_mapper.load_state_dict(ckpt["stage2_mapper"])

    for m in [
        encoder,
        codec,
        fdaa,
        hag,
        ddu,
        decoder,
        phys_t,
        phys_fusion,
        stage2_mapper,
    ]:
        m.eval()

    phase = ckpt.get("phase", "AILD_FREQ_STAGE1")
    mode = ckpt.get("stage2_mode", "stage1_direct_dehaze")

    uses_physical = bool(ckpt.get("uses_physical_guidance", False))
    uses_stage2_mapper = bool(ckpt.get("uses_stage2_mapper", False)) and "stage2_mapper" in ckpt

    test_cfg = opt.get("test", {})
    use_color_fix = bool(test_cfg.get("use_color_fix", False))
    min_chroma = float(test_cfg.get("min_chroma", 0.08))
    chroma_blend = float(test_cfg.get("chroma_blend", 0.70))
    luma_blend = float(test_cfg.get("luma_blend", 0.10))

    out_root = os.path.join("results_test2_fullres", dataset_name)
    dehazed_dir = os.path.join(out_root, "dehazed")
    compare_dir = os.path.join(out_root, "compare")

    os.makedirs(dehazed_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    csv_path = os.path.join(out_root, "metrics.csv")
    summary_path = os.path.join(out_root, "summary.txt")

    print("")
    print("===============================================")
    print(f"Dataset              : {dataset_name}")
    print(f"Checkpoint           : {ckpt_path}")
    print(f"Loaded phase         : {phase}")
    print(f"Mode                 : {mode}")
    print(f"Uses physical        : {uses_physical}")
    print(f"Uses Stage2 mapper   : {uses_stage2_mapper}")
    print(f"Color fix            : {use_color_fix}")
    print(f"Output folder        : {out_root}")
    print("===============================================")
    print("")

    rows = []

    psnr_sum = 0.0
    ssim_sum = 0.0
    score_sum = 0.0
    input_psnr_sum = 0.0
    input_ssim_sum = 0.0
    delta_abs_sum = 0.0
    rg_sum = 0.0
    rb_sum = 0.0
    gb_sum = 0.0
    chroma_sum = 0.0
    gray_count = 0
    colorfix_count = 0
    count = 0

    for idx, batch in enumerate(tqdm(test_dl), start=1):
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        lq_path = batch["LQ_path"][0]
        filename = os.path.basename(lq_path)
        stem, _ = os.path.splitext(filename)

        impl_h = encoder(normalize_for_encoder(lq))
        z_h, c_h = codec.encode(impl_h)

        t_h = None

        if uses_physical and "phys_t" in ckpt and "phys_fusion" in ckpt:
            t_h, _ = phys_t(lq)
            c_h = phys_fusion(c_h, t_h)

        if uses_stage2_mapper:
            if t_h is None:
                t_h, _ = phys_t(lq)
            z_pred, delta = stage2_mapper(z_h, c_h, t_h)
            inference_mode = "Stage2 latent translator physical"
            delta_abs = float(delta.abs().mean().item())
        else:
            z_pred = z_h
            inference_mode = "Stage1 direct dehaze physical" if uses_physical else "Stage1 direct dehaze"
            delta_abs = 0.0

        pred = reconstruct_from_latent(z_pred, c_h, ddu, decoder)

        pred_eval, gt_eval = match_spatial(pred, gt)
        lq_eval, gt_input = match_spatial(lq, gt_eval)

        colorfix_applied = False
        colorfix_chroma_before = 0.0

        if use_color_fix:
            pred_eval, colorfix_applied, colorfix_chroma_before = restore_color_ycbcr_tensor(
                pred_eval=pred_eval,
                lq_eval=lq_eval,
                min_chroma=min_chroma,
                chroma_blend=chroma_blend,
                luma_blend=luma_blend,
            )

        psnr = compute_psnr(pred_eval.float(), gt_eval.float())
        ssim = compute_ssim(pred_eval.float(), gt_eval.float())
        score = psnr + 10.0 * ssim

        input_psnr = compute_psnr(lq_eval.float(), gt_input.float())
        input_ssim = compute_ssim(lq_eval.float(), gt_input.float())

        rg_diff, rb_diff, gb_diff, pred_chroma, is_gray_like = color_diagnostic(pred_eval)

        pred_rgb = tensor_to_rgb_uint8(pred_eval)
        lq_rgb = tensor_to_rgb_uint8(lq_eval)
        gt_rgb = tensor_to_rgb_uint8(gt_eval)

        save_rgb(os.path.join(dehazed_dir, f"{stem}_dehazed.png"), pred_rgb)

        panel = make_compare_panel(
            hazy_rgb=lq_rgb,
            pred_rgb=pred_rgb,
            gt_rgb=gt_rgb,
            filename=filename,
            psnr=psnr,
            ssim=ssim,
            score=score,
            input_psnr=input_psnr,
            input_ssim=input_ssim,
            mode=inference_mode,
            rg_diff=rg_diff,
            rb_diff=rb_diff,
            gb_diff=gb_diff,
            pred_chroma=pred_chroma,
            is_gray_like=is_gray_like,
            color_fix_used=use_color_fix,
            color_fix_applied=colorfix_applied,
            delta_abs=delta_abs,
        )

        save_rgb(os.path.join(compare_dir, f"{stem}_compare.png"), panel)

        h, w = pred_rgb.shape[:2]

        print(
            f"[{idx:03d}/{len(test_dl):03d}] "
            f"{filename} | "
            f"PSNR={psnr:.3f} "
            f"SSIM={ssim:.4f} "
            f"Score={score:.3f} "
            f"Input={input_psnr:.3f}/{input_ssim:.4f} "
            f"DeltaAbs={delta_abs:.5f} "
            f"RGBdiff=({rg_diff:.4f},{rb_diff:.4f},{gb_diff:.4f}) "
            f"Chroma={pred_chroma:.4f} "
            f"GrayLike={is_gray_like} "
            f"ColorFix={use_color_fix} "
            f"Applied={colorfix_applied} "
            f"FullRes={h}x{w}"
        )

        rows.append(
            {
                "index": idx,
                "filename": filename,
                "height": h,
                "width": w,
                "psnr": psnr,
                "ssim": ssim,
                "score": score,
                "input_psnr": input_psnr,
                "input_ssim": input_ssim,
                "delta_abs": delta_abs,
                "rg_diff": rg_diff,
                "rb_diff": rb_diff,
                "gb_diff": gb_diff,
                "pred_chroma": pred_chroma,
                "is_gray_like": is_gray_like,
                "color_fix": use_color_fix,
                "colorfix_applied": colorfix_applied,
                "colorfix_chroma_before": colorfix_chroma_before,
                "phase": phase,
                "mode": inference_mode,
            }
        )

        psnr_sum += psnr
        ssim_sum += ssim
        score_sum += score
        input_psnr_sum += input_psnr
        input_ssim_sum += input_ssim
        delta_abs_sum += delta_abs
        rg_sum += rg_diff
        rb_sum += rb_diff
        gb_sum += gb_diff
        chroma_sum += pred_chroma

        if is_gray_like:
            gray_count += 1

        if colorfix_applied:
            colorfix_count += 1

        count += 1

    avg_psnr = psnr_sum / count
    avg_ssim = ssim_sum / count
    avg_score = score_sum / count
    avg_input_psnr = input_psnr_sum / count
    avg_input_ssim = input_ssim_sum / count
    avg_delta_abs = delta_abs_sum / count
    avg_rg = rg_sum / count
    avg_rb = rb_sum / count
    avg_gb = gb_sum / count
    avg_chroma = chroma_sum / count
    gray_ratio = gray_count / count

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "index",
            "filename",
            "height",
            "width",
            "psnr",
            "ssim",
            "score",
            "input_psnr",
            "input_ssim",
            "delta_abs",
            "rg_diff",
            "rb_diff",
            "gb_diff",
            "pred_chroma",
            "is_gray_like",
            "color_fix",
            "colorfix_applied",
            "colorfix_chroma_before",
            "phase",
            "mode",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Phase: {phase}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Uses Physical: {uses_physical}\n")
        f.write(f"Uses Stage2 Mapper: {uses_stage2_mapper}\n")
        f.write(f"Color Fix: {use_color_fix}\n")
        f.write(f"Samples: {count}\n")
        f.write(f"Average PSNR: {avg_psnr:.6f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.6f}\n")
        f.write(f"Average Score: {avg_score:.6f}\n")
        f.write(f"Average Input PSNR: {avg_input_psnr:.6f} dB\n")
        f.write(f"Average Input SSIM: {avg_input_ssim:.6f}\n")
        f.write(f"Average DeltaAbs: {avg_delta_abs:.6f}\n")
        f.write(f"Average RGBdiff RG: {avg_rg:.6f}\n")
        f.write(f"Average RGBdiff RB: {avg_rb:.6f}\n")
        f.write(f"Average RGBdiff GB: {avg_gb:.6f}\n")
        f.write(f"Average Pred Chroma: {avg_chroma:.6f}\n")
        f.write(f"GrayLike Count: {gray_count}\n")
        f.write(f"GrayLike Ratio: {gray_ratio:.6f}\n")
        f.write(f"ColorFix Applied Count: {colorfix_count}\n")
        f.write(f"Dehazed folder: {dehazed_dir}\n")
        f.write(f"Compare folder: {compare_dir}\n")
        f.write(f"CSV: {csv_path}\n")

    print("")
    print(f"Full-resolution Test2 results on {dataset_name}:")
    print(f"  Average PSNR       : {avg_psnr:.3f} dB")
    print(f"  Average SSIM       : {avg_ssim:.4f}")
    print(f"  Average Score      : {avg_score:.3f}")
    print(f"  Average Input PSNR : {avg_input_psnr:.3f} dB")
    print(f"  Average Input SSIM : {avg_input_ssim:.4f}")
    print(f"  Average DeltaAbs   : {avg_delta_abs:.5f}")
    print(f"  Average RGBdiff    : RG={avg_rg:.4f}, RB={avg_rb:.4f}, GB={avg_gb:.4f}")
    print(f"  Average Chroma     : {avg_chroma:.4f}")
    print(f"  GrayLike Count     : {gray_count}/{count}")
    print(f"  ColorFix Applied   : {colorfix_count}/{count}")
    print(f"  Dehazed images     : {dehazed_dir}")
    print(f"  Compare panels     : {compare_dir}")
    print(f"  Metrics CSV        : {csv_path}")
    print(f"  Summary TXT        : {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")

    args = parser.parse_args()

    run_test(args.config)