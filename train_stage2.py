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


def get_context_alpha(epoch, total_epochs):
    progress = (epoch - 1) / max(1, total_epochs - 1)
    return 1.0 - 0.5 * progress


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


def reconstruct_from_latent(z_pred, c_mix, ddu, decoder):
    return decoder(ddu(z_pred, c_mix)).clamp(0, 1)


def save_merged_checkpoint(path, epoch, codec, fdaa, hag, net, ddu, decoder, predehazer_path, metrics, alpha, phase):
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
    if predehazer_path and os.path.exists(predehazer_path):
        pre_ckpt = load_checkpoint(predehazer_path, "cpu")
        if "model" in pre_ckpt:
            state["predehazer"] = pre_ckpt["model"]
    save_checkpoint(path, state)


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
        z_f, _, _, _ = fdaa(z_h)
        t = torch.zeros((bs,), device=device, dtype=torch.long)
        z0_pred = net(z_f, z_f, t).clamp(-latent_clip, latent_clip)
        img_pred = reconstruct_from_latent(z0_pred, c_mix, ddu, decoder)
        main_loss = loss_fn.forward_stage2(img_pred, gt, z0_pred, z_c)
        ctx_loss = F.l1_loss(c_h, c_c)
        loss = main_loss + lambda_ctx * ctx_loss
        loss_sum += float(loss.item()) * bs
        psnr_sum += compute_psnr(img_pred, gt) * bs
        ssim_sum += compute_ssim(img_pred, gt) * bs
        ctx_sum += float(ctx_loss.item()) * bs
        count += bs
    return loss_sum / count, psnr_sum / count, ssim_sum / count, ctx_sum / count


def train_stage2(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join("checkpoints", opt["name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    encoder, codec, fdaa, hag, net, ddu, decoder = build_stage2_modules(opt, device)
    predehazer_ckpt = os.path.join(ckpt_dir, "predehazer_best.pth")

    s1_ckpt = load_checkpoint(os.path.join(ckpt_dir, "stage1_best.pth"), device)
    codec.load_state_dict(s1_ckpt["codec"])
    ddu.load_state_dict(s1_ckpt["ddu"])
    decoder.load_state_dict(s1_ckpt["decoder"])
    for p in codec.parameters():
        p.requires_grad = False

    loss_fn = AILDFreqLoss(
        **{
            k: opt["loss"][k]
            for k in ["lambda_img", "lambda_lat", "lambda_freq", "lambda_freq_prior", "lambda_perc", "lambda_ssim"]
        }
    ).to(device)
    _, train_dl = build_dataloader(opt, "train")
    _, val_dl = build_dataloader(opt, "val")
    optimizer = torch.optim.AdamW(
        [
            {"params": fdaa.parameters(), "lr": opt["train"]["lr_stage2"]},
            {"params": net.parameters(), "lr": opt["train"]["lr_stage2"]},
            {"params": ddu.parameters(), "lr": 1e-5},
            {"params": decoder.parameters(), "lr": 1e-5},
        ],
        betas=(opt["train"]["beta1"], opt["train"]["beta2"]),
    )

    latent_clip = float(opt["train"]["latent_clip"])
    total_epochs = int(opt["train"]["stage2_warmup_epochs"])
    lambda_ctx = float(opt["loss"].get("lambda_ctx", 0.0))
    merged_name = opt["train"].get("merged_checkpoint_name", "final_best.pth")
    best_psnr = -1.0

    for epoch in range(1, total_epochs + 1):
        fdaa.train()
        hag.train()
        net.train()
        ddu.train()
        decoder.train()
        alpha = get_context_alpha(epoch, total_epochs)
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
            z_f, _, _, _ = fdaa(z_h)
            t = torch.zeros((bs,), device=device, dtype=torch.long)
            z0_pred = net(z_f, z_f, t).clamp(-latent_clip, latent_clip)
            img_pred = reconstruct_from_latent(z0_pred, c_mix, ddu, decoder)
            main_loss = loss_fn.forward_stage2(img_pred, gt, z0_pred, z_c)
            ctx_loss = F.l1_loss(c_h, c_c)
            loss = main_loss + lambda_ctx * ctx_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(fdaa.parameters()) + list(net.parameters()) + list(ddu.parameters()) + list(decoder.parameters()),
                1.0,
            )
            optimizer.step()
            loss_sum += float(loss.item()) * bs
            psnr_sum += compute_psnr(img_pred.detach(), gt) * bs
            count += bs

        if epoch % opt["train"]["val_freq"] == 0 or epoch == 1:
            val_loss, val_psnr, val_ssim, val_ctx = validate_stage2(
                encoder, codec, fdaa, hag, net, ddu, decoder, val_dl, loss_fn, opt, device, alpha
            )
            print(
                f"[S2][{epoch:04d}/{total_epochs}] alpha={alpha:.3f} "
                f"train_loss={loss_sum / count:.4f} train_psnr={psnr_sum / count:.3f} "
                f"val_loss={val_loss:.4f} val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f} val_ctx={val_ctx:.4f}"
            )
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                metrics = {"psnr": val_psnr, "ssim": val_ssim, "loss": val_loss, "ctx": val_ctx}
                stage2_state = {
                    "epoch": epoch,
                    "codec": codec.state_dict(),
                    "fdaa": fdaa.state_dict(),
                    "hag": hag.state_dict(),
                    "net": net.state_dict(),
                    "ddu": ddu.state_dict(),
                    "decoder": decoder.state_dict(),
                    "psnr": val_psnr,
                    "ssim": val_ssim,
                    "alpha": alpha,
                    "phase": "FINAL_WARM_ONLY_SSIM",
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
                    predehazer_ckpt,
                    metrics,
                    alpha,
                    "FINAL_WARM_ONLY_SSIM",
                )
    print(f"Best Stage2 PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    args = parser.parse_args()
    train_stage2(args.config)
