import os

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
from models.modules.loss import AILDFreqLoss
from models.modules.nafnet import ConditionalNAFNet
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder
from utils.checkpoint_utils import load_checkpoint, save_checkpoint
from utils.img_utils import normalize_for_encoder
from utils.metrics import compute_psnr, compute_ssim


def build_dataloader(opt, split):
    ds = LQGTDataset(opt, split)
    dl = DataLoader(
        ds,
        batch_size=opt["datasets"][split]["batch_size"],
        shuffle=opt["datasets"][split].get("use_shuffle", False),
        num_workers=opt["datasets"][split].get("n_workers", 0),
        pin_memory=True,
    )
    return ds, dl


def get_context_alpha(epoch, opt, total_epochs):
    sched = opt["train"].get("alpha_schedule", None)
    if sched is None:
        progress = (epoch - 1) / max(1, total_epochs - 1)
        alpha_start = float(opt["train"].get("alpha_start", 1.0))
        alpha_end = float(opt["train"].get("alpha_end", 0.5))
        return alpha_start + (alpha_end - alpha_start) * progress

    boundaries = sched.get("boundaries", [])
    values = sched.get("values", [1.0, 0.75, 0.5])

    if len(values) != len(boundaries) + 1:
        raise ValueError("alpha_schedule.values must have exactly len(boundaries) + 1 elements")

    for boundary, value in zip(boundaries, values):
        if epoch <= int(boundary):
            return float(value)
    return float(values[-1])


def build_scheduler(optimizer, opt, total_epochs):
    sched_cfg = opt["train"].get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine").lower()

    if sched_type == "cosine":
        min_lr = float(sched_cfg.get("min_lr", 5.0e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=min_lr,
        )

    if sched_type == "step":
        step_size = int(sched_cfg.get("step_size", max(1, total_epochs // 3)))
        gamma = float(sched_cfg.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    return None


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def build_stage2_modules(opt, device):
    encoder = ConvNextEncoder(
        pretrained=opt["network_encoder"]["pretrained"],
        backbone=opt["network_encoder"].get("backbone", "convnext_small"),
    ).to(device)
    encoder.eval()

    impl_ch = opt["network_L"]["implicit_channels"]
    lat_ch = opt["network_L"]["latent_channels"]
    cee_ch = opt["network_L"]["cee_channels"]

    codec = DehazeDiffCodec(impl_ch, lat_ch, cee_ch).to(device)
    ddu = DimensionalDecompressionUnit(lat_ch, impl_ch, cee_ch).to(device)
    decoder = PixelShuffleDecoder(in_channels=impl_ch, upscale_factor=8).to(device)
    fdaa = FDAA(**opt["fdaa"]).to(device)
    hag = HAG(t_max=opt["hag"]["t_max"], min_steps=opt["hag"]["min_steps"]).to(device)
    net = ConditionalNAFNet(**opt["network_G"]["setting"]).to(device)
    return encoder, codec, fdaa, hag, net, ddu, decoder


def reconstruct_from_latent(z_pred, c_ctx, ddu, decoder):
    return decoder(ddu(z_pred, c_ctx)).clamp(0, 1)


def save_merged_checkpoint(path, epoch, codec, fdaa, hag, net, ddu, decoder, metrics, alpha, phase):
    state = {
        "epoch": epoch,
        "codec": codec.state_dict(),
        "fdaa": fdaa.state_dict(),
        "hag": hag.state_dict(),
        "net": net.state_dict(),
        "ddu": ddu.state_dict(),
        "decoder": decoder.state_dict(),
        "alpha": alpha,
        "phase": phase,
        "metrics": metrics,
    }
    save_checkpoint(path, state)


def build_condition(z_h, fdaa, hag):
    z_f, f_low, f_high, alpha_map = fdaa(z_h)
    _, hag_gate = hag(f_low, f_high)
    g = hag_gate.view(-1, 1, 1, 1)
    z_cond = g * z_f + (1.0 - g) * z_h
    return z_cond, z_f, alpha_map, hag_gate


@torch.no_grad()
def validate_stage2(encoder, codec, fdaa, hag, net, ddu, decoder, val_loader, loss_fn, opt, device, alpha):
    for m in [encoder, codec, fdaa, hag, net, ddu, decoder]:
        m.eval()

    latent_clip = float(opt["train"]["latent_clip"])
    lambda_ctx = float(opt["loss"].get("lambda_ctx", 0.0))
    loss_sum, psnr_sum, ssim_sum, ctx_sum, count = 0.0, 0.0, 0.0, 0.0, 0

    for batch in val_loader:
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)
        bs = lq.size(0)

        impl_h = encoder(normalize_for_encoder(lq))
        impl_c = encoder(normalize_for_encoder(gt))
        z_h, c_h = codec.encode(impl_h)
        z_c, c_c = codec.encode(impl_c)

        c_mix = alpha * c_c + (1.0 - alpha) * c_h
        z_cond, _, _, _ = build_condition(z_h, fdaa, hag)

        t = torch.zeros((bs,), device=device, dtype=torch.long)
        z0_pred = net(z_cond, z_cond, t).clamp(-latent_clip, latent_clip)
        img_pred = reconstruct_from_latent(z0_pred, c_mix, ddu, decoder)

        main_loss = loss_fn.forward_stage2(img_pred, gt, z0_pred, z_c)
        ctx_loss = F.l1_loss(c_h, c_c)
        loss = main_loss + lambda_ctx * ctx_loss

        loss_sum += float(loss.item()) * bs
        psnr_sum += compute_psnr(img_pred, gt) * bs
        ssim_sum += compute_ssim(img_pred, gt) * bs
        ctx_sum += float(ctx_loss.item()) * bs
        count += bs

    return {
        "loss": loss_sum / count,
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
        "ctx": ctx_sum / count,
    }


def train_stage2(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join("checkpoints", opt["name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    encoder, codec, fdaa, hag, net, ddu, decoder = build_stage2_modules(opt, device)

    s1_ckpt = load_checkpoint(os.path.join(ckpt_dir, "stage1_best.pth"), device)
    codec.load_state_dict(s1_ckpt["codec"])
    ddu.load_state_dict(s1_ckpt["ddu"])
    decoder.load_state_dict(s1_ckpt["decoder"])

    for p in codec.parameters():
        p.requires_grad = False

    loss_kwargs = {
        "lambda_img": opt["loss"].get("lambda_img", 1.0),
        "lambda_lat": opt["loss"].get("lambda_lat", 0.5),
        "lambda_freq": opt["loss"].get("lambda_freq", 0.2),
        "lambda_freq_prior": opt["loss"].get("lambda_freq_prior", 0.01),
        "lambda_perc": opt["loss"].get("lambda_perc", 0.0),
        "lambda_ssim": opt["loss"].get("lambda_ssim", 0.15),
        "lambda_edge": opt["loss"].get("lambda_edge", 0.05),
    }
    loss_fn = AILDFreqLoss(**loss_kwargs).to(device)

    _, train_dl = build_dataloader(opt, "train")
    _, val_dl = build_dataloader(opt, "val")

    optimizer = torch.optim.AdamW(
        [
            {"params": fdaa.parameters(), "lr": opt["train"]["lr_stage2"]},
            {"params": hag.parameters(), "lr": opt["train"]["lr_stage2"]},
            {"params": net.parameters(), "lr": opt["train"]["lr_stage2"]},
            {"params": ddu.parameters(), "lr": 1e-5},
            {"params": decoder.parameters(), "lr": 1e-5},
        ],
        betas=(opt["train"]["beta1"], opt["train"]["beta2"]),
    )

    total_epochs = int(opt["train"]["stage2_warmup_epochs"])
    scheduler = build_scheduler(optimizer, opt, total_epochs)

    latent_clip = float(opt["train"]["latent_clip"])
    lambda_ctx = float(opt["loss"].get("lambda_ctx", 0.0))
    merged_name = opt["train"].get("merged_checkpoint_name", "final_best.pth")

    best_psnr = -1.0

    for epoch in range(1, total_epochs + 1):
        fdaa.train()
        hag.train()
        net.train()
        ddu.train()
        decoder.train()

        alpha = get_context_alpha(epoch, opt, total_epochs)
        loss_sum, psnr_sum, count = 0.0, 0.0, 0

        for batch in train_dl:
            lq = batch["LQ"].to(device)
            gt = batch["GT"].to(device)
            bs = lq.size(0)

            with torch.no_grad():
                impl_h = encoder(normalize_for_encoder(lq))
                impl_c = encoder(normalize_for_encoder(gt))
                z_h, c_h = codec.encode(impl_h)
                z_c, c_c = codec.encode(impl_c)

            c_mix = alpha * c_c + (1.0 - alpha) * c_h
            z_cond, _, _, _ = build_condition(z_h, fdaa, hag)

            t = torch.zeros((bs,), device=device, dtype=torch.long)
            z0_pred = net(z_cond, z_cond, t).clamp(-latent_clip, latent_clip)
            img_pred = reconstruct_from_latent(z0_pred, c_mix, ddu, decoder)

            main_loss = loss_fn.forward_stage2(img_pred, gt, z0_pred, z_c)
            ctx_loss = F.l1_loss(c_h, c_c)
            loss = main_loss + lambda_ctx * ctx_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fdaa.parameters()) + list(hag.parameters()) + list(net.parameters()) + list(ddu.parameters()) + list(decoder.parameters()),
                1.0,
            )
            optimizer.step()

            loss_sum += float(loss.item()) * bs
            psnr_sum += compute_psnr(img_pred.detach(), gt) * bs
            count += bs

        if scheduler is not None:
            scheduler.step()

        if epoch % opt["train"]["val_freq"] == 0 or epoch == 1:
            val_stats = validate_stage2(
                encoder, codec, fdaa, hag, net, ddu, decoder, val_dl, loss_fn, opt, device, alpha
            )
            print(
                f"[S2-PRETRAINED][{epoch:04d}/{total_epochs}] alpha={alpha:.3f} lr={get_current_lr(optimizer):.2e} "
                f"train_loss={loss_sum / count:.4f} train_psnr={psnr_sum / count:.3f} "
                f"val_loss={val_stats['loss']:.4f} val_psnr={val_stats['psnr']:.3f} "
                f"val_ssim={val_stats['ssim']:.4f} val_ctx={val_stats['ctx']:.4f}"
            )

            if val_stats["psnr"] > best_psnr:
                best_psnr = val_stats["psnr"]
                metrics = {
                    "psnr": val_stats["psnr"],
                    "ssim": val_stats["ssim"],
                    "loss": val_stats["loss"],
                    "ctx": val_stats["ctx"],
                }
                stage2_state = {
                    "epoch": epoch,
                    "codec": codec.state_dict(),
                    "fdaa": fdaa.state_dict(),
                    "hag": hag.state_dict(),
                    "net": net.state_dict(),
                    "ddu": ddu.state_dict(),
                    "decoder": decoder.state_dict(),
                    "psnr": val_stats["psnr"],
                    "ssim": val_stats["ssim"],
                    "alpha": alpha,
                    "phase": "AILD_FREQ_PRETRAINED_UNIFIED",
                }
                save_checkpoint(os.path.join(ckpt_dir, "stage2_best.pth"), stage2_state)
                save_merged_checkpoint(
                    os.path.join(ckpt_dir, merged_name),
                    epoch,
                    codec,
                    fdaa,
                    hag,
                    net,
                    ddu,
                    decoder,
                    metrics,
                    alpha,
                    "AILD_FREQ_PRETRAINED_UNIFIED",
                )

    print(f"Best Stage2 PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    args = parser.parse_args()
    train_stage2(args.config)
