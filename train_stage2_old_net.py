import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.lqgt_dataset import LQGTDataset

from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.fdaa import FDAA
from models.modules.hag import HAG
from models.modules.nafnet import ConditionalNAFNet
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

    net_cfg = opt.get("network_G", {}).get("setting", {})
    net = ConditionalNAFNet(
        img_channel=int(net_cfg.get("img_channel", lat_ch)),
        width=int(net_cfg.get("width", 64)),
        middle_blk_num=int(net_cfg.get("middle_blk_num", 8)),
        enc_blk_nums=tuple(net_cfg.get("enc_blk_nums", [1, 1, 1])),
        dec_blk_nums=tuple(net_cfg.get("dec_blk_nums", [1, 1, 1])),
    ).to(device)

    return encoder, codec, fdaa, hag, net, ddu, decoder


def zero_init_net_output(net):
    """
    Critical stabilization:
    Start Stage 2 as almost identity residual mapping.
    If net output starts near zero, then z_pred ~= z_h at epoch 1.
    """
    if hasattr(net, "ending") and isinstance(net.ending, nn.Conv2d):
        nn.init.zeros_(net.ending.weight)
        if net.ending.bias is not None:
            nn.init.zeros_(net.ending.bias)
        print("[Stage2] Zero-initialized net.ending for residual-safe start.")
    else:
        print("[Stage2] Warning: could not find net.ending Conv2d for zero init.")


def reconstruct_from_latent(z, c, ddu, decoder):
    return decoder(ddu(z, c)).clamp(0, 1)


def build_hazy_condition(fdaa, hag, z_h):
    z_f, f_low, f_high, alpha_map = fdaa(z_h)

    _, hag_gate = hag(f_low, f_high)
    g = hag_gate.view(-1, 1, 1, 1)

    z_cond = g * z_f + (1.0 - g) * z_h

    return z_cond, z_f, f_low, f_high, alpha_map, hag_gate


def predict_stage2(fdaa, hag, net, z_h, latent_clip, residual_scale):
    z_cond, z_f, f_low, f_high, alpha_map, hag_gate = build_hazy_condition(
        fdaa=fdaa,
        hag=hag,
        z_h=z_h,
    )

    bs = z_h.size(0)
    t = torch.zeros(
        (bs,),
        device=z_h.device,
        dtype=torch.long,
    )

    delta_raw = net(
        z_cond,
        z_cond,
        t,
    )

    delta = residual_scale * torch.tanh(delta_raw)

    z_pred = (z_h + delta).clamp(
        -latent_clip,
        latent_clip,
    )

    return {
        "z_pred": z_pred,
        "delta": delta,
        "z_cond": z_cond,
        "z_f": z_f,
        "alpha_map": alpha_map,
        "hag_gate": hag_gate,
    }


def color_loss(pred, target):
    pred_mean = pred.mean(dim=(2, 3))
    target_mean = target.mean(dim=(2, 3))

    pred_std = pred.std(dim=(2, 3), unbiased=False)
    target_std = target.std(dim=(2, 3), unbiased=False)

    return F.l1_loss(pred_mean, target_mean) + 0.5 * F.l1_loss(pred_std, target_std)


def tv_loss(img):
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw


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


def compute_losses(img_pred, gt, z_pred, z_h, z_c, delta, z_cond, opt):
    img_pred, gt_eval = match_spatial(img_pred, gt)

    l_img = F.l1_loss(img_pred.float(), gt_eval.float())
    l_ssim = ssim_loss(img_pred, gt_eval)
    l_edge = edge_loss(img_pred, gt_eval)
    l_color = color_loss(img_pred, gt_eval)
    l_tv = tv_loss(img_pred)

    target_delta = (z_c - z_h).detach()

    l_z = F.l1_loss(z_pred, z_c.detach())
    l_delta = F.l1_loss(delta, target_delta)

    # Prevent FDAA/HAG from moving z_h too aggressively.
    l_cond_identity = F.l1_loss(z_cond, z_h.detach())

    lambda_img = float(opt["loss"].get("lambda_stage2_img", 1.0))
    lambda_ssim = float(opt["loss"].get("lambda_stage2_ssim", 0.35))
    lambda_edge = float(opt["loss"].get("lambda_stage2_edge", 0.10))
    lambda_color = float(opt["loss"].get("lambda_stage2_color", 0.05))
    lambda_tv = float(opt["loss"].get("lambda_stage2_tv", 0.001))

    lambda_z = float(opt["loss"].get("lambda_stage2_z", 0.20))
    lambda_delta = float(opt["loss"].get("lambda_stage2_delta", 0.05))
    lambda_cond_identity = float(opt["loss"].get("lambda_stage2_cond_identity", 0.02))

    total = (
        lambda_img * l_img
        + lambda_ssim * l_ssim
        + lambda_edge * l_edge
        + lambda_color * l_color
        + lambda_tv * l_tv
        + lambda_z * l_z
        + lambda_delta * l_delta
        + lambda_cond_identity * l_cond_identity
    )

    return {
        "total": total,
        "img": l_img,
        "ssim_loss": l_ssim,
        "edge": l_edge,
        "color": l_color,
        "tv": l_tv,
        "z": l_z,
        "delta": l_delta,
        "cond_identity": l_cond_identity,
        "img_pred": img_pred,
        "gt_eval": gt_eval,
    }


@torch.no_grad()
def validate_stage2(
    encoder,
    codec,
    fdaa,
    hag,
    net,
    ddu,
    decoder,
    val_loader,
    opt,
    device,
):
    for m in [encoder, codec, fdaa, hag, net, ddu, decoder]:
        m.eval()

    latent_clip = float(opt["train"].get("latent_clip", 3.0))
    residual_scale = float(opt["train"].get("stage2_residual_scale", 1.0))

    sums = {
        "loss": 0.0,
        "img": 0.0,
        "ssim_loss": 0.0,
        "edge": 0.0,
        "color": 0.0,
        "tv": 0.0,
        "z": 0.0,
        "delta": 0.0,
        "cond_identity": 0.0,

        "psnr": 0.0,
        "ssim": 0.0,
        "score": 0.0,

        "baseline_psnr": 0.0,
        "baseline_ssim": 0.0,
        "baseline_score": 0.0,

        "fdaa_only_psnr": 0.0,
        "zcond_only_psnr": 0.0,
        "cross_psnr": 0.0,
        "cross_ssim": 0.0,

        "zgap": 0.0,
        "target_gap": 0.0,
        "cond_gap": 0.0,
        "fdaa_alpha": 0.0,
        "hag_gate": 0.0,
    }

    count = 0

    for batch in val_loader:
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        bs = lq.size(0)

        impl_h = encoder(normalize_for_encoder(lq))
        impl_c = encoder(normalize_for_encoder(gt))

        z_h, c_h = codec.encode(impl_h)
        z_c, _ = codec.encode(impl_c)

        baseline = reconstruct_from_latent(z_h, c_h, ddu, decoder)

        pred_pack = predict_stage2(
            fdaa=fdaa,
            hag=hag,
            net=net,
            z_h=z_h,
            latent_clip=latent_clip,
            residual_scale=residual_scale,
        )

        z_pred = pred_pack["z_pred"]
        delta = pred_pack["delta"]
        z_cond = pred_pack["z_cond"]
        z_f = pred_pack["z_f"]

        pred = reconstruct_from_latent(z_pred, c_h, ddu, decoder)
        fdaa_only = reconstruct_from_latent(z_f, c_h, ddu, decoder)
        zcond_only = reconstruct_from_latent(z_cond, c_h, ddu, decoder)
        cross = reconstruct_from_latent(z_c, c_h, ddu, decoder)

        losses = compute_losses(
            img_pred=pred,
            gt=gt,
            z_pred=z_pred,
            z_h=z_h,
            z_c=z_c,
            delta=delta,
            z_cond=z_cond,
            opt=opt,
        )

        pred_eval = losses["img_pred"]
        gt_eval = losses["gt_eval"]

        baseline_eval, gt_base = match_spatial(baseline, gt)
        fdaa_only, gt_fdaa = match_spatial(fdaa_only, gt)
        zcond_only, gt_cond = match_spatial(zcond_only, gt)
        cross, gt_cross = match_spatial(cross, gt)

        psnr = compute_psnr(pred_eval.float(), gt_eval.float())
        ssim = compute_ssim(pred_eval.float(), gt_eval.float())
        score = psnr + 10.0 * ssim

        base_psnr = compute_psnr(baseline_eval.float(), gt_base.float())
        base_ssim = compute_ssim(baseline_eval.float(), gt_base.float())
        base_score = base_psnr + 10.0 * base_ssim

        fdaa_psnr = compute_psnr(fdaa_only.float(), gt_fdaa.float())
        cond_psnr = compute_psnr(zcond_only.float(), gt_cond.float())

        cross_psnr = compute_psnr(cross.float(), gt_cross.float())
        cross_ssim = compute_ssim(cross.float(), gt_cross.float())

        sums["loss"] += float(losses["total"].item()) * bs
        sums["img"] += float(losses["img"].item()) * bs
        sums["ssim_loss"] += float(losses["ssim_loss"].item()) * bs
        sums["edge"] += float(losses["edge"].item()) * bs
        sums["color"] += float(losses["color"].item()) * bs
        sums["tv"] += float(losses["tv"].item()) * bs
        sums["z"] += float(losses["z"].item()) * bs
        sums["delta"] += float(losses["delta"].item()) * bs
        sums["cond_identity"] += float(losses["cond_identity"].item()) * bs

        sums["psnr"] += psnr * bs
        sums["ssim"] += ssim * bs
        sums["score"] += score * bs

        sums["baseline_psnr"] += base_psnr * bs
        sums["baseline_ssim"] += base_ssim * bs
        sums["baseline_score"] += base_score * bs

        sums["fdaa_only_psnr"] += fdaa_psnr * bs
        sums["zcond_only_psnr"] += cond_psnr * bs
        sums["cross_psnr"] += cross_psnr * bs
        sums["cross_ssim"] += cross_ssim * bs

        sums["zgap"] += float(F.l1_loss(z_pred, z_c.detach()).item()) * bs
        sums["target_gap"] += float(F.l1_loss(z_h, z_c.detach()).item()) * bs
        sums["cond_gap"] += float(F.l1_loss(z_cond, z_h.detach()).item()) * bs
        sums["fdaa_alpha"] += float(pred_pack["alpha_map"].mean().item()) * bs
        sums["hag_gate"] += float(pred_pack["hag_gate"].mean().item()) * bs

        count += bs

    return {k: v / count for k, v in sums.items()}


def save_stage2_checkpoint(
    path,
    epoch,
    codec,
    fdaa,
    hag,
    net,
    ddu,
    decoder,
    metrics,
):
    state = {
        "epoch": epoch,

        "codec": codec.state_dict(),
        "fdaa": fdaa.state_dict(),
        "hag": hag.state_dict(),
        "net": net.state_dict(),
        "ddu": ddu.state_dict(),
        "decoder": decoder.state_dict(),

        "phase": "AILD_FREQ_STAGE2_FDAA_HAG_RESIDUAL_SAFE",
        "stage2_mode": "fdaa_hag_residual_safe",

        "metrics": metrics,
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "score": metrics.get("score"),

        "baseline_psnr": metrics.get("baseline_psnr"),
        "baseline_ssim": metrics.get("baseline_ssim"),
        "baseline_score": metrics.get("baseline_score"),

        "uses_fdaa": True,
        "uses_hag": True,
        "uses_net": True,
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

    encoder, codec, fdaa, hag, net, ddu, decoder = build_modules(opt, device)

    stage1_path = os.path.join(ckpt_dir, "stage1_best.pth")
    s1_ckpt = load_checkpoint(stage1_path, device)

    codec.load_state_dict(s1_ckpt["codec"])
    ddu.load_state_dict(s1_ckpt["ddu"])
    decoder.load_state_dict(s1_ckpt["decoder"])

    print(f"[Stage2-FDAA-HAG] Loaded Stage1 checkpoint: {stage1_path}")
    print(
        f"[Stage2-FDAA-HAG] Stage1 metrics | "
        f"selection={s1_ckpt.get('selection_metric', 'unknown')} "
        f"dehaze_psnr={s1_ckpt.get('dehaze_psnr', 'unknown')} "
        f"dehaze_ssim={s1_ckpt.get('dehaze_ssim', 'unknown')} "
        f"cross_psnr={s1_ckpt.get('cross_psnr', 'unknown')} "
        f"cross_ssim={s1_ckpt.get('cross_ssim', 'unknown')}"
    )

    freeze_module(encoder)
    freeze_module(codec)
    freeze_module(ddu)
    freeze_module(decoder)

    zero_init_net_output(net)

    _, train_dl = build_dataloader(opt, "train")
    _, val_dl = build_dataloader(opt, "val")

    net_lr = float(opt["train"].get("lr_stage2", 3.0e-5))
    fdaa_lr = float(opt["train"].get("fdaa_lr", 1.0e-5))
    hag_lr = float(opt["train"].get("hag_lr", 1.0e-5))

    optimizer = torch.optim.AdamW(
        [
            {"params": net.parameters(), "lr": net_lr},
            {"params": fdaa.parameters(), "lr": fdaa_lr},
            {"params": hag.parameters(), "lr": hag_lr},
        ],
        betas=(
            float(opt["train"].get("beta1", 0.9)),
            float(opt["train"].get("beta2", 0.99)),
        ),
        weight_decay=float(opt["train"].get("weight_decay", 1.0e-4)),
    )

    total_epochs = int(opt["train"].get("stage2_warmup_epochs", 300))
    val_freq = int(opt["train"].get("val_freq", 25))
    latent_clip = float(opt["train"].get("latent_clip", 3.0))
    residual_scale = float(opt["train"].get("stage2_residual_scale", 1.0))
    save_margin = float(opt["train"].get("stage2_save_margin", 0.02))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=float(opt["train"].get("min_lr_stage2", 1.0e-6)),
    )

    merged_name = opt["train"].get("merged_checkpoint_name", "final_best.pth")

    best_score = -1.0
    best_psnr = -1.0
    best_ssim = -1.0

    print(
        f"[Stage2-FDAA-HAG] Active residual Stage2 | "
        f"epochs={total_epochs} | net_lr={net_lr} | "
        f"fdaa_lr={fdaa_lr} | hag_lr={hag_lr} | "
        f"residual_scale={residual_scale}"
    )

    for epoch in range(1, total_epochs + 1):
        encoder.eval()
        codec.eval()
        ddu.eval()
        decoder.eval()

        fdaa.train()
        hag.train()
        net.train()

        sums = {
            "loss": 0.0,
            "img": 0.0,
            "ssim_loss": 0.0,
            "edge": 0.0,
            "color": 0.0,
            "z": 0.0,
            "delta": 0.0,
            "cond_identity": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "score": 0.0,
            "zgap": 0.0,
            "target_gap": 0.0,
            "cond_gap": 0.0,
            "fdaa_alpha": 0.0,
            "hag_gate": 0.0,
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
                z_c, _ = codec.encode(impl_c)

            pred_pack = predict_stage2(
                fdaa=fdaa,
                hag=hag,
                net=net,
                z_h=z_h,
                latent_clip=latent_clip,
                residual_scale=residual_scale,
            )

            z_pred = pred_pack["z_pred"]
            delta = pred_pack["delta"]
            z_cond = pred_pack["z_cond"]

            img_pred = reconstruct_from_latent(z_pred, c_h, ddu, decoder)

            losses = compute_losses(
                img_pred=img_pred,
                gt=gt,
                z_pred=z_pred,
                z_h=z_h,
                z_c=z_c,
                delta=delta,
                z_cond=z_cond,
                opt=opt,
            )

            loss = losses["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(fdaa.parameters()) + list(hag.parameters()),
                float(opt["train"].get("grad_clip", 1.0)),
            )

            optimizer.step()

            img_eval = losses["img_pred"]
            gt_eval = losses["gt_eval"]

            psnr = compute_psnr(img_eval.detach().float(), gt_eval.float())
            ssim = compute_ssim(img_eval.detach().float(), gt_eval.float())
            score = psnr + 10.0 * ssim

            sums["loss"] += float(loss.item()) * bs
            sums["img"] += float(losses["img"].item()) * bs
            sums["ssim_loss"] += float(losses["ssim_loss"].item()) * bs
            sums["edge"] += float(losses["edge"].item()) * bs
            sums["color"] += float(losses["color"].item()) * bs
            sums["z"] += float(losses["z"].item()) * bs
            sums["delta"] += float(losses["delta"].item()) * bs
            sums["cond_identity"] += float(losses["cond_identity"].item()) * bs

            sums["psnr"] += psnr * bs
            sums["ssim"] += ssim * bs
            sums["score"] += score * bs

            sums["zgap"] += float(F.l1_loss(z_pred.detach(), z_c.detach()).item()) * bs
            sums["target_gap"] += float(F.l1_loss(z_h.detach(), z_c.detach()).item()) * bs
            sums["cond_gap"] += float(F.l1_loss(z_cond.detach(), z_h.detach()).item()) * bs
            sums["fdaa_alpha"] += float(pred_pack["alpha_map"].detach().mean().item()) * bs
            sums["hag_gate"] += float(pred_pack["hag_gate"].detach().mean().item()) * bs

            count += bs

        scheduler.step()

        if epoch % val_freq == 0 or epoch == 1:
            val_stats = validate_stage2(
                encoder=encoder,
                codec=codec,
                fdaa=fdaa,
                hag=hag,
                net=net,
                ddu=ddu,
                decoder=decoder,
                val_loader=val_dl,
                opt=opt,
                device=device,
            )

            train_stats = {k: v / count for k, v in sums.items()}
            improvement = val_stats["score"] - val_stats["baseline_score"]

            print(
                f"[S2-FDAA-HAG][{epoch:04d}/{total_epochs}] "
                f"train_loss={train_stats['loss']:.4f} "
                f"img={train_stats['img']:.4f} "
                f"ssim_loss={train_stats['ssim_loss']:.4f} "
                f"z={train_stats['z']:.5f} "
                f"delta={train_stats['delta']:.5f} "
                f"cond_id={train_stats['cond_identity']:.5f} "
                f"train_psnr={train_stats['psnr']:.3f} "
                f"train_ssim={train_stats['ssim']:.4f} | "
                f"val={val_stats['psnr']:.3f}/"
                f"{val_stats['ssim']:.4f} "
                f"score={val_stats['score']:.3f} | "
                f"base={val_stats['baseline_psnr']:.3f}/"
                f"{val_stats['baseline_ssim']:.4f} "
                f"base_score={val_stats['baseline_score']:.3f} "
                f"improve={improvement:.3f} "
                f"fdaa_only={val_stats['fdaa_only_psnr']:.3f} "
                f"zcond_only={val_stats['zcond_only_psnr']:.3f} "
                f"cross={val_stats['cross_psnr']:.3f}/"
                f"{val_stats['cross_ssim']:.4f} "
                f"zgap={val_stats['zgap']:.5f} "
                f"target_gap={val_stats['target_gap']:.5f} "
                f"cond_gap={val_stats['cond_gap']:.5f} "
                f"fdaa={val_stats['fdaa_alpha']:.4f} "
                f"hag={val_stats['hag_gate']:.4f}"
            )

            if val_stats["score"] > best_score:
                best_score = val_stats["score"]
                best_psnr = val_stats["psnr"]
                best_ssim = val_stats["ssim"]

            if val_stats["score"] > val_stats["baseline_score"] + save_margin:
                metrics = {
                    "psnr": val_stats["psnr"],
                    "ssim": val_stats["ssim"],
                    "score": val_stats["score"],

                    "baseline_psnr": val_stats["baseline_psnr"],
                    "baseline_ssim": val_stats["baseline_ssim"],
                    "baseline_score": val_stats["baseline_score"],
                    "improvement": improvement,

                    "loss": val_stats["loss"],
                    "img_loss": val_stats["img"],
                    "ssim_loss": val_stats["ssim_loss"],
                    "edge_loss": val_stats["edge"],
                    "color_loss": val_stats["color"],
                    "tv_loss": val_stats["tv"],
                    "z_loss": val_stats["z"],
                    "delta_loss": val_stats["delta"],
                    "cond_identity": val_stats["cond_identity"],

                    "fdaa_only_psnr": val_stats["fdaa_only_psnr"],
                    "zcond_only_psnr": val_stats["zcond_only_psnr"],
                    "cross_psnr": val_stats["cross_psnr"],
                    "cross_ssim": val_stats["cross_ssim"],

                    "zgap": val_stats["zgap"],
                    "target_gap": val_stats["target_gap"],
                    "cond_gap": val_stats["cond_gap"],
                    "fdaa_alpha": val_stats["fdaa_alpha"],
                    "hag_gate": val_stats["hag_gate"],

                    "mode": "fdaa_hag_residual_safe",
                }

                save_stage2_checkpoint(
                    os.path.join(ckpt_dir, "stage2_best.pth"),
                    epoch,
                    codec,
                    fdaa,
                    hag,
                    net,
                    ddu,
                    decoder,
                    metrics,
                )

                save_stage2_checkpoint(
                    os.path.join(ckpt_dir, merged_name),
                    epoch,
                    codec,
                    fdaa,
                    hag,
                    net,
                    ddu,
                    decoder,
                    metrics,
                )

                print(
                    f"Saved Stage2 FDAA-HAG checkpoint "
                    f"(PSNR={val_stats['psnr']:.3f}, "
                    f"SSIM={val_stats['ssim']:.4f}, "
                    f"improvement={improvement:.3f})"
                )
            else:
                print("Stage2 not saved because it has not beaten Stage1 baseline.")

    print(
        f"Best observed Stage2 FDAA-HAG: "
        f"score={best_score:.3f}, PSNR={best_psnr:.3f}, SSIM={best_ssim:.4f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")

    args = parser.parse_args()

    train_stage2(args.config)