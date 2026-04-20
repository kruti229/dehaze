import os
import csv
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.lqgt_dataset import LQGTDataset
from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.fdaa import FDAA
from models.modules.hag import HAG
from models.modules.loss import AILDFreqLoss
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder
from models.modules.nafnet import ConditionalNAFNet
from utils.sde_utils import IRSDE


def diffusion_noise_loss(pred_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_noise, true_noise)


def append_csv_row(csv_path, row_dict):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)


def _compute_ssim_simple(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Simple global SSIM fallback (dependency-free), pred/gt in [0,1].
    pred, gt: [B,3,H,W], uses first item.
    """
    x = pred[0].clamp(0, 1).detach().cpu()
    y = gt[0].clamp(0, 1).detach().cpu()

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(x.shape[0]):
        xc = x[c]
        yc = y[c]
        mux = xc.mean()
        muy = yc.mean()
        sigx = ((xc - mux) ** 2).mean()
        sigy = ((yc - muy) ** 2).mean()
        sigxy = ((xc - mux) * (yc - muy)).mean()

        num = (2 * mux * muy + C1) * (2 * sigxy + C2)
        den = (mux * mux + muy * muy + C1) * (sigx + sigy + C2)
        vals.append((num / (den + 1e-12)).item())
    return float(sum(vals) / len(vals))


@torch.no_grad()
def validate(encoder, codec, fdaa, hag, net, ddu, decoder, sde, opt, device):
    """
    Returns:
      val_psnr, val_ssim, val_t_eff_mean
    """
    from utils.metrics import compute_psnr  # keep your existing metric util

    encoder.eval()
    codec.eval()
    fdaa.eval()
    hag.eval()
    net.eval()
    ddu.eval()
    decoder.eval()

    val_ds = LQGTDataset(opt, "val")
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    psnrs, ssims, steps_list = [], [], []

    for batch in val_dl:
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        # Encode
        impl_lq = encoder(lq)
        z_lq, cee = codec.encode(impl_lq)

        # FDAA + HAG
        z_fdaa, f_low, f_high, _ = fdaa(z_lq)
        t_eff, _ = hag(f_low, f_high)
        steps = int(t_eff.item())
        steps_list.append(steps)

        # Reverse SDE
        sde.set_mu(z_fdaa)
        z_noisy = sde.noise_state(z_lq)
        z_clean = sde.reverse_sde(z_noisy, t_eff=steps, z_fdaa=z_fdaa)

        # Decode
        impl_clean = ddu(z_clean, cee)
        img_clean = decoder(impl_clean).clamp(0, 1)

        # Metrics
        psnrs.append(float(compute_psnr(img_clean, gt)))
        ssims.append(_compute_ssim_simple(img_clean, gt))

    val_psnr = float(sum(psnrs) / max(1, len(psnrs)))
    val_ssim = float(sum(ssims) / max(1, len(ssims)))
    val_t_eff_mean = float(sum(steps_list) / max(1, len(steps_list)))
    return val_psnr, val_ssim, val_t_eff_mean


def train_stage2(config_path: str):
    opt = yaml.safe_load(open(config_path, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Directories / logging
    # -----------------------------
    ckpt_dir = f"checkpoints/{opt['name']}"
    os.makedirs(ckpt_dir, exist_ok=True)

    log_dir = os.path.join("logs", opt["name"])
    os.makedirs(log_dir, exist_ok=True)
    metrics_csv = os.path.join(log_dir, "stage2_metrics.csv")

    # -----------------------------
    # Load stage1 checkpoint
    # -----------------------------
    stage1_path = f"{ckpt_dir}/stage1_best.pth"
    if not os.path.exists(stage1_path):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {stage1_path}")

    stage1 = torch.load(stage1_path, map_location=device)

    # -----------------------------
    # Models
    # -----------------------------
    encoder = ConvNextEncoder(pretrained=True).to(device)
    codec = DehazeDiffCodec(192, 16, 64).to(device)
    codec.load_state_dict(stage1["codec"])

    ddu = DimensionalDecompressionUnit(16, 192, 64).to(device)
    ddu.load_state_dict(stage1["ddu"])

    decoder = PixelShuffleDecoder(in_channels=192, upscale_factor=8).to(device)
    if "decoder" in stage1:
        decoder.load_state_dict(stage1["decoder"])

    # Freeze encoder + codec in stage2
    for p in encoder.parameters():
        p.requires_grad = False
    for p in codec.parameters():
        p.requires_grad = False

    # Trainable modules
    net_cfg = opt["network_G"]["setting"]
    net = ConditionalNAFNet(**net_cfg).to(device)
    fdaa = FDAA(
        channels=opt["fdaa"]["channels"],
        num_heads=opt["fdaa"]["num_heads"],
        low_freq_ratio=opt["fdaa"]["low_freq_ratio"],
    ).to(device)
    hag = HAG(
        t_max=opt["hag"]["t_max"],
        min_steps=opt["hag"]["min_steps"],
    ).to(device)

    # -----------------------------
    # Optimizer
    # -----------------------------
    from lion_pytorch import Lion

    trainable = (
        list(net.parameters())
        + list(fdaa.parameters())
        + list(hag.parameters())
        + list(ddu.parameters())
        + list(decoder.parameters())
    )
    optim = Lion(
        trainable,
        lr=opt["train"]["lr_G"],
        betas=(opt["train"]["beta1"], opt["train"]["beta2"]),
    )

    # -----------------------------
    # SDE + losses
    # -----------------------------
    sde = IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], device=device)
    sde.set_model(net)

    image_loss_fn = AILDFreqLoss(
        l1=opt["loss"]["lambda1"],
        l2=opt["loss"]["lambda2"],
        l3=opt["loss"]["lambda3"],
    ).to(device)

    # -----------------------------
    # Data
    # -----------------------------
    train_ds = LQGTDataset(opt, "train")
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check dataroot_LQ / dataroot_GT in config.")

    train_dl = DataLoader(
        train_ds,
        batch_size=opt["datasets"]["train"]["batch_size"],
        shuffle=True,
        num_workers=opt["datasets"]["train"].get("n_workers", 4),
        pin_memory=True,
    )

    epochs = int(opt["train"]["stage2_epochs"])
    val_freq = int(opt["train"]["val_freq"])
    best_psnr = -1.0

    print(f"Stage 2 training: {epochs} epochs, {len(train_ds)} samples")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        net.train()
        fdaa.train()
        hag.train()
        ddu.train()
        decoder.train()

        running_total = 0.0
        running_diff = 0.0
        running_img = 0.0
        running_steps = 0

        for batch in train_dl:
            lq = batch["LQ"].to(device)
            gt = batch["GT"].to(device)

            # Frozen encode
            with torch.no_grad():
                impl_lq = encoder(lq)
                impl_gt = encoder(gt)
                z_lq, cee = codec.encode(impl_lq)
                z_gt, _ = codec.encode(impl_gt)

            # FDAA + HAG
            z_fdaa, f_low, f_high, _ = fdaa(z_lq)
            t_eff, _ = hag(f_low, f_high)  # [B]

            # Sample timestep constrained by haze-adaptive steps
            t_upper = max(2, int(t_eff.min().item()))
            t = torch.randint(1, t_upper + 1, (1,), device=z_lq.device).item()

            # Diffusion train step
            sde.set_mu(z_fdaa)
            z_noisy = sde.noise_state(z_gt)
            pred_noise = net(z_noisy, z_fdaa, t)

            sigma_t = sde.sigmas[t].to(z_gt.device)
            true_noise = (z_noisy - z_gt) / (sigma_t + 1e-8)

            # 1) score/noise loss
            loss_diff = diffusion_noise_loss(pred_noise, true_noise)

            # 2) image-space loss through decode path
            z_est = z_noisy - sigma_t * pred_noise
            impl_est = ddu(z_est, cee)
            img_est = decoder(impl_est).clamp(0, 1)
            loss_img = image_loss_fn(img_est, gt, stage=2)

            # total
            loss = loss_diff + loss_img

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_total += float(loss.item())
            running_diff += float(loss_diff.item())
            running_img += float(loss_img.item())
            running_steps += 1

        avg_total = running_total / max(1, running_steps)
        avg_diff = running_diff / max(1, running_steps)
        avg_img = running_img / max(1, running_steps)
        epoch_sec = time.time() - epoch_start

        # quick train print every epoch
        print(
            f"Epoch {epoch:6d}/{epochs} | "
            f"total={avg_total:.4f} diff={avg_diff:.4f} img={avg_img:.4f} "
            f"| {epoch_sec:.1f}s"
        )

        # validation
        if epoch % val_freq == 0:
            val_psnr, val_ssim, val_t_eff_mean = validate(
                encoder, codec, fdaa, hag, net, ddu, decoder, sde, opt, device
            )

            print(
                f"  [val] PSNR={val_psnr:.3f} dB | "
                f"SSIM={val_ssim:.4f} | "
                f"mean_t_eff={val_t_eff_mean:.2f}"
            )

            append_csv_row(
                metrics_csv,
                {
                    "epoch": epoch,
                    "train_total": avg_total,
                    "train_diff": avg_diff,
                    "train_img": avg_img,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "val_t_eff_mean": val_t_eff_mean,
                    "epoch_sec": epoch_sec,
                },
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    {
                        "epoch": epoch,
                        "net": net.state_dict(),
                        "fdaa": fdaa.state_dict(),
                        "hag": hag.state_dict(),
                        "ddu": ddu.state_dict(),
                        "decoder": decoder.state_dict(),
                        "psnr": val_psnr,
                        "ssim": val_ssim,
                        "t_eff_mean": val_t_eff_mean,
                    },
                    f"{ckpt_dir}/stage2_best.pth",
                )
                print(f"  [save] best checkpoint -> {ckpt_dir}/stage2_best.pth")

    print(f"Stage 2 complete. Best PSNR: {best_psnr:.3f} dB")
    print(f"Metrics CSV: {metrics_csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nh_haze.yml")
    args = ap.parse_args()
    train_stage2(args.config)