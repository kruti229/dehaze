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
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder

from utils.checkpoint_utils import save_checkpoint
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
        shuffle=bool(opt["datasets"][split].get("use_shuffle", False)),
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

    return encoder, codec, ddu, decoder


def reconstruct_from_latent(z, c, ddu, decoder):
    return decoder(ddu(z, c)).clamp(0, 1)


def color_loss(pred, target):
    pred_mean = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))

    pred_std = pred.std(dim=(2, 3), unbiased=False)
    target_std = target.std(dim=(2, 3), unbiased=False)

    return F.l1_loss(pred_mean, target_mean) + 0.5 * F.l1_loss(pred_std, target_std)


def sobel_edges(x):
    _, c, _, _ = x.shape

    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)

    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)

    return torch.sqrt(gx * gx + gy * gy + 1.0e-6)


def edge_loss(pred, target):
    return F.l1_loss(
        sobel_edges(pred.float()),
        sobel_edges(target.float()),
    )


def tv_loss(img):
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw


def ssim_loss(pred, target):
    try:
        from pytorch_msssim import ssim

        return 1.0 - ssim(
            pred.float(),
            target.float(),
            data_range=1.0,
            size_average=True,
        )
    except Exception:
        return torch.tensor(
            0.0,
            device=pred.device,
            dtype=pred.dtype,
        )


def frequency_loss(pred, target):
    pred_fft = torch.fft.fft2(pred.float(), norm="ortho")
    target_fft = torch.fft.fft2(target.float(), norm="ortho")

    pred_amp = torch.log1p(torch.abs(pred_fft))
    target_amp = torch.log1p(torch.abs(target_fft))

    return F.l1_loss(pred_amp, target_amp)


def estimate_atmospheric_light(hazy, top_percent=0.001):
    b, c, _, _ = hazy.shape

    flat = hazy.view(b, c, -1)
    intensity = flat.mean(dim=1)

    k = max(1, int(intensity.shape[1] * top_percent))
    idx = torch.topk(intensity, k=k, dim=1).indices

    a_list = []
    for i in range(b):
        pixels = flat[i, :, idx[i]]
        a = pixels.mean(dim=1)
        a_list.append(a)

    A = torch.stack(a_list, dim=0).view(b, c, 1, 1)
    return A.clamp(0.05, 1.0)


def estimate_transmission_pseudo(hazy, omega=0.95, min_t=0.10):
    dark = hazy.min(dim=1, keepdim=True)[0]
    t = 1.0 - omega * dark
    return t.clamp(min_t, 1.0)


def asm_reconstruction_loss(pred_clean, hazy):
    pred_clean, hazy = match_spatial(pred_clean, hazy)

    with torch.no_grad():
        A = estimate_atmospheric_light(hazy)
        t = estimate_transmission_pseudo(hazy)

    hazy_recon = pred_clean * t + A * (1.0 - t)

    return F.l1_loss(hazy_recon.float(), hazy.float())


def recon_loss(pred, target, opt):
    pred, target = match_spatial(pred, target)

    lambda_ssim = float(opt["loss"].get("lambda_ssim", 0.35))
    lambda_edge = float(opt["loss"].get("lambda_edge", 0.12))
    lambda_color = float(opt["loss"].get("lambda_color", 0.08))
    lambda_tv = float(opt["loss"].get("lambda_tv", 0.0005))
    lambda_freq = float(opt["loss"].get("lambda_freq", 0.10))

    l1 = F.l1_loss(pred.float(), target.float())
    ssim_l = ssim_loss(pred, target)
    edge_l = edge_loss(pred, target)
    color_l = color_loss(pred, target)
    tv_l = tv_loss(pred)
    freq_l = frequency_loss(pred, target)

    total = (
        l1
        + lambda_ssim * ssim_l
        + lambda_edge * edge_l
        + lambda_color * color_l
        + lambda_tv * tv_l
        + lambda_freq * freq_l
    )

    return total, {
        "l1": l1.detach(),
        "ssim_loss": ssim_l.detach(),
        "edge": edge_l.detach(),
        "color": color_l.detach(),
        "tv": tv_l.detach(),
        "freq": freq_l.detach(),
    }


def latent_alignment_loss(z_h, z_c):
    return F.l1_loss(z_h, z_c.detach())


def context_alignment_loss(c_h, c_c):
    return F.l1_loss(c_h, c_c.detach())


@torch.no_grad()
def validate_stage1(
    encoder,
    codec,
    ddu,
    decoder,
    val_loader,
    opt,
    device,
):
    for m in [encoder, codec, ddu, decoder]:
        m.eval()

    lambda_dehaze = float(opt["loss"].get("lambda_dehaze_recon", 2.0))
    lambda_hazy = float(opt["loss"].get("lambda_hazy_recon", 0.0))
    lambda_clean = float(opt["loss"].get("lambda_clean_recon", 0.5))
    lambda_cross = float(opt["loss"].get("lambda_cross", 0.5))
    lambda_z = float(opt["loss"].get("lambda_z_align", 0.05))
    lambda_cee = float(opt["loss"].get("lambda_cee_align", 0.0))
    lambda_asm = float(opt["loss"].get("lambda_asm", 0.03))

    sums = {
        "loss": 0.0,

        "dehaze_psnr": 0.0,
        "dehaze_ssim": 0.0,
        "dehaze_score": 0.0,

        "clean_psnr": 0.0,
        "clean_ssim": 0.0,

        "cross_psnr": 0.0,
        "cross_ssim": 0.0,

        "hazy_recon_psnr": 0.0,
        "hazy_recon_ssim": 0.0,

        "z_align": 0.0,
        "cee_align": 0.0,
        "asm_loss": 0.0,
    }

    count = 0

    for batch in val_loader:
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        bs = lq.size(0)

        impl_h = encoder(normalize_for_encoder(lq))
        impl_c = encoder(normalize_for_encoder(gt))

        z_h, c_h = codec.encode(impl_h)
        z_c, c_c = codec.encode(impl_c)

        pred_dehaze = reconstruct_from_latent(z_h, c_h, ddu, decoder)
        pred_clean = reconstruct_from_latent(z_c, c_c, ddu, decoder)
        pred_cross = reconstruct_from_latent(z_c, c_h, ddu, decoder)

        pred_dehaze_gt, gt_eval = match_spatial(pred_dehaze, gt)
        pred_dehaze_lq, lq_eval = match_spatial(pred_dehaze, lq)

        pred_clean, gt_clean = match_spatial(pred_clean, gt)
        pred_cross, gt_cross = match_spatial(pred_cross, gt)

        loss_dehaze, _ = recon_loss(pred_dehaze_gt, gt_eval, opt)
        loss_hazy, _ = recon_loss(pred_dehaze_lq, lq_eval, opt)
        loss_clean, _ = recon_loss(pred_clean, gt_clean, opt)
        loss_cross, _ = recon_loss(pred_cross, gt_cross, opt)

        z_align = latent_alignment_loss(z_h, z_c)
        cee_align = context_alignment_loss(c_h, c_c)
        loss_asm = asm_reconstruction_loss(pred_dehaze_gt, lq_eval)

        loss = (
            lambda_dehaze * loss_dehaze
            + lambda_hazy * loss_hazy
            + lambda_clean * loss_clean
            + lambda_cross * loss_cross
            + lambda_z * z_align
            + lambda_cee * cee_align
            + lambda_asm * loss_asm
        )

        dehaze_psnr = compute_psnr(pred_dehaze_gt.float(), gt_eval.float())
        dehaze_ssim = compute_ssim(pred_dehaze_gt.float(), gt_eval.float())
        dehaze_score = dehaze_psnr + 10.0 * dehaze_ssim

        clean_psnr = compute_psnr(pred_clean.float(), gt_clean.float())
        clean_ssim = compute_ssim(pred_clean.float(), gt_clean.float())

        cross_psnr = compute_psnr(pred_cross.float(), gt_cross.float())
        cross_ssim = compute_ssim(pred_cross.float(), gt_cross.float())

        hazy_psnr = compute_psnr(pred_dehaze_lq.float(), lq_eval.float())
        hazy_ssim = compute_ssim(pred_dehaze_lq.float(), lq_eval.float())

        sums["loss"] += float(loss.item()) * bs

        sums["dehaze_psnr"] += dehaze_psnr * bs
        sums["dehaze_ssim"] += dehaze_ssim * bs
        sums["dehaze_score"] += dehaze_score * bs

        sums["clean_psnr"] += clean_psnr * bs
        sums["clean_ssim"] += clean_ssim * bs

        sums["cross_psnr"] += cross_psnr * bs
        sums["cross_ssim"] += cross_ssim * bs

        sums["hazy_recon_psnr"] += hazy_psnr * bs
        sums["hazy_recon_ssim"] += hazy_ssim * bs

        sums["z_align"] += float(z_align.item()) * bs
        sums["cee_align"] += float(cee_align.item()) * bs
        sums["asm_loss"] += float(loss_asm.item()) * bs

        count += bs

    return {k: v / count for k, v in sums.items()}


def save_stage1_checkpoint(
    path,
    epoch,
    codec,
    ddu,
    decoder,
    metrics,
):
    state = {
        "epoch": epoch,

        "codec": codec.state_dict(),
        "ddu": ddu.state_dict(),
        "decoder": decoder.state_dict(),

        "phase": "AILD_FREQ_STAGE1_DEHAZE_ALIGNED_ASM",
        "selection_metric": "dehaze_score",

        "metrics": metrics,

        "dehaze_psnr": metrics["dehaze_psnr"],
        "dehaze_ssim": metrics["dehaze_ssim"],
        "dehaze_score": metrics["dehaze_score"],

        "clean_psnr": metrics["clean_psnr"],
        "clean_ssim": metrics["clean_ssim"],

        "cross_psnr": metrics["cross_psnr"],
        "cross_ssim": metrics["cross_ssim"],

        "hazy_psnr": metrics["hazy_recon_psnr"],
        "hazy_ssim": metrics["hazy_recon_ssim"],

        "asm_loss": metrics.get("asm_loss", None),
    }

    save_checkpoint(path, state)


def train_stage1(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    set_seed(int(opt["train"].get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = os.path.join("checkpoints", opt["name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    encoder, codec, ddu, decoder = build_modules(opt, device)

    freeze_module(encoder)

    _, train_dl = build_dataloader(opt, "train")
    _, val_dl = build_dataloader(opt, "val")

    params = list(codec.parameters()) + list(ddu.parameters()) + list(decoder.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=float(opt["train"].get("lr_stage1", 1.0e-4)),
        betas=(
            float(opt["train"].get("beta1", 0.9)),
            float(opt["train"].get("beta2", 0.99)),
        ),
        weight_decay=float(opt["train"].get("weight_decay", 1.0e-4)),
    )

    total_epochs = int(opt["train"].get("stage1_epochs", 500))
    val_freq = int(opt["train"].get("val_freq", 25))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=float(opt["train"].get("min_lr_stage1", 5.0e-6)),
    )

    lambda_dehaze = float(opt["loss"].get("lambda_dehaze_recon", 2.0))
    lambda_hazy = float(opt["loss"].get("lambda_hazy_recon", 0.0))
    lambda_clean = float(opt["loss"].get("lambda_clean_recon", 0.5))
    lambda_cross = float(opt["loss"].get("lambda_cross", 0.5))
    lambda_z = float(opt["loss"].get("lambda_z_align", 0.05))
    lambda_cee = float(opt["loss"].get("lambda_cee_align", 0.0))
    lambda_asm = float(opt["loss"].get("lambda_asm", 0.03))

    print(
        f"[Stage1-Dehaze-ASM] Training {opt['name']} | "
        f"epochs={total_epochs} | "
        f"lambda_dehaze={lambda_dehaze} | "
        f"lambda_hazy={lambda_hazy} | "
        f"lambda_clean={lambda_clean} | "
        f"lambda_cross={lambda_cross} | "
        f"lambda_z={lambda_z} | "
        f"lambda_cee={lambda_cee} | "
        f"lambda_asm={lambda_asm}"
    )

    best_score = -1.0
    best_psnr = -1.0
    best_ssim = -1.0

    for epoch in range(1, total_epochs + 1):
        encoder.eval()
        codec.train()
        ddu.train()
        decoder.train()

        sums = {
            "loss": 0.0,
            "dehaze": 0.0,
            "hazy": 0.0,
            "clean": 0.0,
            "cross": 0.0,
            "z": 0.0,
            "cee": 0.0,
            "asm": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "score": 0.0,
        }

        count = 0

        for batch in train_dl:
            lq = batch["LQ"].to(device)
            gt = batch["GT"].to(device)

            bs = lq.size(0)

            with torch.no_grad():
                impl_h = encoder(normalize_for_encoder(lq))
                impl_c = encoder(normalize_for_encoder(gt))

            z_h, c_h = codec.encode(impl_h)
            z_c, c_c = codec.encode(impl_c)

            pred_dehaze = reconstruct_from_latent(z_h, c_h, ddu, decoder)
            pred_clean = reconstruct_from_latent(z_c, c_c, ddu, decoder)
            pred_cross = reconstruct_from_latent(z_c, c_h, ddu, decoder)

            pred_dehaze_gt, gt_eval = match_spatial(pred_dehaze, gt)
            pred_dehaze_lq, lq_eval = match_spatial(pred_dehaze, lq)

            pred_clean, gt_clean = match_spatial(pred_clean, gt)
            pred_cross, gt_cross = match_spatial(pred_cross, gt)

            loss_dehaze, _ = recon_loss(pred_dehaze_gt, gt_eval, opt)
            loss_hazy, _ = recon_loss(pred_dehaze_lq, lq_eval, opt)
            loss_clean, _ = recon_loss(pred_clean, gt_clean, opt)
            loss_cross, _ = recon_loss(pred_cross, gt_cross, opt)

            z_align = latent_alignment_loss(z_h, z_c)
            cee_align = context_alignment_loss(c_h, c_c)
            loss_asm = asm_reconstruction_loss(pred_dehaze_gt, lq_eval)

            loss = (
                lambda_dehaze * loss_dehaze
                + lambda_hazy * loss_hazy
                + lambda_clean * loss_clean
                + lambda_cross * loss_cross
                + lambda_z * z_align
                + lambda_cee * cee_align
                + lambda_asm * loss_asm
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params,
                float(opt["train"].get("grad_clip", 1.0)),
            )

            optimizer.step()

            psnr = compute_psnr(pred_dehaze_gt.detach().float(), gt_eval.float())
            ssim = compute_ssim(pred_dehaze_gt.detach().float(), gt_eval.float())
            score = psnr + 10.0 * ssim

            sums["loss"] += float(loss.item()) * bs
            sums["dehaze"] += float(loss_dehaze.item()) * bs
            sums["hazy"] += float(loss_hazy.item()) * bs
            sums["clean"] += float(loss_clean.item()) * bs
            sums["cross"] += float(loss_cross.item()) * bs
            sums["z"] += float(z_align.item()) * bs
            sums["cee"] += float(cee_align.item()) * bs
            sums["asm"] += float(loss_asm.item()) * bs

            sums["psnr"] += psnr * bs
            sums["ssim"] += ssim * bs
            sums["score"] += score * bs

            count += bs

        scheduler.step()

        if epoch % val_freq == 0 or epoch == 1:
            val_stats = validate_stage1(
                encoder=encoder,
                codec=codec,
                ddu=ddu,
                decoder=decoder,
                val_loader=val_dl,
                opt=opt,
                device=device,
            )

            lr_now = optimizer.param_groups[0]["lr"]

            print(
                f"[S1-Dehaze-ASM][{epoch:04d}/{total_epochs}] "
                f"lr={lr_now:.2e} "
                f"train_loss={sums['loss'] / count:.4f} "
                f"dehaze={sums['dehaze'] / count:.4f} "
                f"hazy={sums['hazy'] / count:.4f} "
                f"clean={sums['clean'] / count:.4f} "
                f"cross={sums['cross'] / count:.4f} "
                f"z={sums['z'] / count:.5f} "
                f"cee={sums['cee'] / count:.5f} "
                f"asm={sums['asm'] / count:.5f} "
                f"train_psnr={sums['psnr'] / count:.3f} "
                f"train_ssim={sums['ssim'] / count:.4f} | "
                f"val_dehaze={val_stats['dehaze_psnr']:.3f}/"
                f"{val_stats['dehaze_ssim']:.4f} "
                f"val_score={val_stats['dehaze_score']:.3f} "
                f"val_clean={val_stats['clean_psnr']:.3f}/"
                f"{val_stats['clean_ssim']:.4f} "
                f"val_cross={val_stats['cross_psnr']:.3f}/"
                f"{val_stats['cross_ssim']:.4f} "
                f"val_hazy_diag={val_stats['hazy_recon_psnr']:.3f}/"
                f"{val_stats['hazy_recon_ssim']:.4f} "
                f"val_z={val_stats['z_align']:.5f} "
                f"val_cee={val_stats['cee_align']:.5f} "
                f"val_asm={val_stats['asm_loss']:.5f}"
            )

            if val_stats["dehaze_score"] > best_score:
                best_score = val_stats["dehaze_score"]
                best_psnr = val_stats["dehaze_psnr"]
                best_ssim = val_stats["dehaze_ssim"]

                save_stage1_checkpoint(
                    os.path.join(ckpt_dir, "stage1_best.pth"),
                    epoch,
                    codec,
                    ddu,
                    decoder,
                    val_stats,
                )

                print(
                    f"Saved best Stage1-Dehaze-ASM checkpoint "
                    f"(score={best_score:.3f}, "
                    f"PSNR={best_psnr:.3f}, "
                    f"SSIM={best_ssim:.4f})"
                )

            save_stage1_checkpoint(
                os.path.join(ckpt_dir, "stage1_last.pth"),
                epoch,
                codec,
                ddu,
                decoder,
                val_stats,
            )

    print(
        f"Best Stage1-Dehaze-ASM: "
        f"score={best_score:.3f}, "
        f"PSNR={best_psnr:.3f}, "
        f"SSIM={best_ssim:.4f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")

    args = parser.parse_args()

    train_stage1(args.config)