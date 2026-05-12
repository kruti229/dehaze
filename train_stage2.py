import os
import random

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.lqgt_dataset import LQGTDataset

from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.fdaa import FDAA
from models.modules.hag import HAG
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder

from utils.checkpoint_utils import load_checkpoint, save_checkpoint
from utils.img_utils import normalize_for_encoder
from utils.metrics import compute_psnr, compute_ssim


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(opt, split):
    ds = LQGTDataset(opt, split)

    dl = DataLoader(
        ds,
        batch_size=int(opt["datasets"][split]["batch_size"]),
        shuffle=False,
        num_workers=int(opt["datasets"][split].get("n_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return ds, dl


def match_spatial(pred, target):
    if pred.shape[-2:] == target.shape[-2:]:
        return pred, target

    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])

    return pred[..., :h, :w], target[..., :h, :w]


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def build_modules(opt, device):
    encoder = ConvNextEncoder(
        pretrained=bool(opt["network_encoder"]["pretrained"]),
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

    return encoder, codec, fdaa, hag, ddu, decoder


def reconstruct_from_latent(z, c, ddu, decoder):
    return decoder(ddu(z, c)).clamp(0, 1)


def fdaa_calibrated_latent(fdaa, z_h, eta):
    z_f, f_low, f_high, alpha_map = fdaa(z_h)

    # identity-safe FDAA residual
    # eta=0.0 gives exact Stage1 baseline
    z_pred = z_h + eta * torch.tanh(z_f - z_h)

    return z_pred, z_f, alpha_map


@torch.no_grad()
def evaluate_eta(
    encoder,
    codec,
    fdaa,
    ddu,
    decoder,
    loader,
    eta,
    opt,
    device,
):
    for m in [encoder, codec, fdaa, ddu, decoder]:
        m.eval()

    sums = {
        "psnr": 0.0,
        "ssim": 0.0,
        "score": 0.0,

        "baseline_psnr": 0.0,
        "baseline_ssim": 0.0,
        "baseline_score": 0.0,

        "raw_fdaa_psnr": 0.0,
        "raw_fdaa_ssim": 0.0,

        "cross_psnr": 0.0,
        "cross_ssim": 0.0,

        "zgap": 0.0,
        "target_gap": 0.0,
        "fdaa_gap": 0.0,
        "alpha": 0.0,
    }

    count = 0

    for batch in loader:
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        bs = lq.size(0)

        impl_h = encoder(normalize_for_encoder(lq))
        impl_c = encoder(normalize_for_encoder(gt))

        z_h, c_h = codec.encode(impl_h)
        z_c, _ = codec.encode(impl_c)

        img_base = reconstruct_from_latent(z_h, c_h, ddu, decoder)

        z_pred, z_f, alpha_map = fdaa_calibrated_latent(
            fdaa=fdaa,
            z_h=z_h,
            eta=eta,
        )

        img_pred = reconstruct_from_latent(z_pred, c_h, ddu, decoder)
        img_raw_fdaa = reconstruct_from_latent(z_f, c_h, ddu, decoder)
        img_cross = reconstruct_from_latent(z_c, c_h, ddu, decoder)

        img_pred, gt_eval = match_spatial(img_pred, gt)
        img_base, gt_base = match_spatial(img_base, gt)
        img_raw_fdaa, gt_fdaa = match_spatial(img_raw_fdaa, gt)
        img_cross, gt_cross = match_spatial(img_cross, gt)

        psnr = compute_psnr(img_pred.float(), gt_eval.float())
        ssim = compute_ssim(img_pred.float(), gt_eval.float())
        score = psnr + 10.0 * ssim

        base_psnr = compute_psnr(img_base.float(), gt_base.float())
        base_ssim = compute_ssim(img_base.float(), gt_base.float())
        base_score = base_psnr + 10.0 * base_ssim

        raw_fdaa_psnr = compute_psnr(img_raw_fdaa.float(), gt_fdaa.float())
        raw_fdaa_ssim = compute_ssim(img_raw_fdaa.float(), gt_fdaa.float())

        cross_psnr = compute_psnr(img_cross.float(), gt_cross.float())
        cross_ssim = compute_ssim(img_cross.float(), gt_cross.float())

        sums["psnr"] += psnr * bs
        sums["ssim"] += ssim * bs
        sums["score"] += score * bs

        sums["baseline_psnr"] += base_psnr * bs
        sums["baseline_ssim"] += base_ssim * bs
        sums["baseline_score"] += base_score * bs

        sums["raw_fdaa_psnr"] += raw_fdaa_psnr * bs
        sums["raw_fdaa_ssim"] += raw_fdaa_ssim * bs

        sums["cross_psnr"] += cross_psnr * bs
        sums["cross_ssim"] += cross_ssim * bs

        sums["zgap"] += float(F.l1_loss(z_pred, z_c.detach()).item()) * bs
        sums["target_gap"] += float(F.l1_loss(z_h, z_c.detach()).item()) * bs
        sums["fdaa_gap"] += float(F.l1_loss(z_f, z_h.detach()).item()) * bs
        sums["alpha"] += float(alpha_map.mean().item()) * bs

        count += bs

    return {k: v / count for k, v in sums.items()}


def save_stage2_checkpoint(
    path,
    epoch,
    codec,
    fdaa,
    hag,
    ddu,
    decoder,
    metrics,
    eta,
):
    state = {
        "epoch": epoch,

        "codec": codec.state_dict(),
        "fdaa": fdaa.state_dict(),
        "hag": hag.state_dict(),
        "ddu": ddu.state_dict(),
        "decoder": decoder.state_dict(),

        "phase": "AILD_FREQ_STAGE2_FDAA_CALIBRATED",
        "stage2_mode": "fdaa_eta_calibration",

        "metrics": metrics,

        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "score": metrics.get("score"),

        "baseline_psnr": metrics.get("baseline_psnr"),
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_score": metrics.get("baseline_score"),

        "eta": eta,

        "uses_fdaa": True,
        "uses_hag": False,
        "uses_net": False,
        "residual_stage2": True,
    }

    save_checkpoint(path, state)


def train_stage2(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    set_seed(int(opt["train"].get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = os.path.join("checkpoints", opt["name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    encoder, codec, fdaa, hag, ddu, decoder = build_modules(opt, device)

    stage1_path = os.path.join(ckpt_dir, "stage1_best.pth")
    s1_ckpt = load_checkpoint(stage1_path, device)

    codec.load_state_dict(s1_ckpt["codec"])
    ddu.load_state_dict(s1_ckpt["ddu"])
    decoder.load_state_dict(s1_ckpt["decoder"])

    print(f"[Stage2-Calib] Loaded Stage1 checkpoint: {stage1_path}")
    print(
        f"[Stage2-Calib] Stage1 metrics | "
        f"selection={s1_ckpt.get('selection_metric', 'unknown')} "
        f"dehaze_psnr={s1_ckpt.get('dehaze_psnr', 'unknown')} "
        f"dehaze_ssim={s1_ckpt.get('dehaze_ssim', 'unknown')} "
        f"dehaze_score={s1_ckpt.get('dehaze_score', 'unknown')} "
        f"cross_psnr={s1_ckpt.get('cross_psnr', 'unknown')} "
        f"cross_ssim={s1_ckpt.get('cross_ssim', 'unknown')}"
    )

    freeze_module(encoder)
    freeze_module(codec)
    freeze_module(ddu)
    freeze_module(decoder)
    freeze_module(hag)
    freeze_module(fdaa)

    _, val_dl = build_dataloader(opt, "val")

    eta_candidates = opt["train"].get(
        "stage2_eta_candidates",
        [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0],
    )

    eta_candidates = [float(x) for x in eta_candidates]
    save_margin = float(opt["train"].get("stage2_save_margin", 0.0))

    print(f"[Stage2-Calib] ETA candidates: {eta_candidates}")
    print("[Stage2-Calib] No training. Validation calibration only.")

    best_eta = None
    best_stats = None
    best_score = -1.0

    for eta in eta_candidates:
        stats = evaluate_eta(
            encoder=encoder,
            codec=codec,
            fdaa=fdaa,
            ddu=ddu,
            decoder=decoder,
            loader=val_dl,
            eta=eta,
            opt=opt,
            device=device,
        )

        improvement = stats["score"] - stats["baseline_score"]

        print(
            f"[ETA={eta:.3f}] "
            f"val={stats['psnr']:.3f}/{stats['ssim']:.4f} "
            f"score={stats['score']:.3f} | "
            f"base={stats['baseline_psnr']:.3f}/{stats['baseline_ssim']:.4f} "
            f"base_score={stats['baseline_score']:.3f} "
            f"improve={improvement:.4f} | "
            f"raw_fdaa={stats['raw_fdaa_psnr']:.3f}/{stats['raw_fdaa_ssim']:.4f} "
            f"cross={stats['cross_psnr']:.3f}/{stats['cross_ssim']:.4f} "
            f"zgap={stats['zgap']:.5f} "
            f"target_gap={stats['target_gap']:.5f} "
            f"fdaa_gap={stats['fdaa_gap']:.5f} "
            f"alpha={stats['alpha']:.4f}"
        )

        if stats["score"] > best_score:
            best_score = stats["score"]
            best_eta = eta
            best_stats = stats

    if best_stats is None:
        raise RuntimeError("No ETA candidate was evaluated.")

    improvement = best_stats["score"] - best_stats["baseline_score"]

    metrics = {
        "psnr": best_stats["psnr"],
        "ssim": best_stats["ssim"],
        "score": best_stats["score"],

        "baseline_psnr": best_stats["baseline_psnr"],
        "baseline_ssim": best_stats["baseline_ssim"],
        "baseline_score": best_stats["baseline_score"],
        "improvement": improvement,

        "raw_fdaa_psnr": best_stats["raw_fdaa_psnr"],
        "raw_fdaa_ssim": best_stats["raw_fdaa_ssim"],

        "cross_psnr": best_stats["cross_psnr"],
        "cross_ssim": best_stats["cross_ssim"],

        "zgap": best_stats["zgap"],
        "target_gap": best_stats["target_gap"],
        "fdaa_gap": best_stats["fdaa_gap"],
        "alpha": best_stats["alpha"],

        "eta": best_eta,
        "mode": "fdaa_eta_calibration",
    }

    print("")
    print("========== Stage2 Calibration Result ==========")
    print(f"Best eta       : {best_eta}")
    print(f"Best val PSNR  : {best_stats['psnr']:.3f}")
    print(f"Best val SSIM  : {best_stats['ssim']:.4f}")
    print(f"Best score     : {best_stats['score']:.3f}")
    print(f"Stage1 score   : {best_stats['baseline_score']:.3f}")
    print(f"Improvement    : {improvement:.4f}")
    print("===============================================")

    should_save = best_stats["score"] >= best_stats["baseline_score"] + save_margin

    if should_save:
        save_stage2_checkpoint(
            os.path.join(ckpt_dir, "stage2_best.pth"),
            0,
            codec,
            fdaa,
            hag,
            ddu,
            decoder,
            metrics,
            best_eta,
        )

        save_stage2_checkpoint(
            os.path.join(ckpt_dir, opt["train"].get("merged_checkpoint_name", "final_best.pth")),
            0,
            codec,
            fdaa,
            hag,
            ddu,
            decoder,
            metrics,
            best_eta,
        )

        print(
            f"Saved Stage2 calibrated FDAA checkpoint "
            f"(eta={best_eta}, improvement={improvement:.4f})"
        )
    else:
        print(
            "Stage2 not saved because no eta candidate matched or beat Stage1 baseline."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")

    args = parser.parse_args()

    train_stage2(args.config)