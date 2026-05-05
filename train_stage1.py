import os
import yaml
import torch
from torch.utils.data import DataLoader

from data.lqgt_dataset import LQGTDataset
from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder
from models.modules.loss import AILDFreqLoss
from utils.img_utils import normalize_for_encoder
from utils.metrics import compute_psnr, compute_ssim
from utils.checkpoint_utils import save_checkpoint


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


@torch.no_grad()
def validate_stage1(encoder, codec, ddu, decoder, val_loader, loss_fn, device):
    encoder.eval()
    codec.eval()
    ddu.eval()
    decoder.eval()

    hazy_loss_sum, clean_loss_sum = 0.0, 0.0
    hazy_psnr_sum, clean_psnr_sum = 0.0, 0.0
    hazy_ssim_sum, clean_ssim_sum = 0.0, 0.0
    count = 0

    for batch in val_loader:
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)
        bs = lq.size(0)

        # Hazy branch
        impl_h = encoder(normalize_for_encoder(lq))
        z_h, c_h = codec.encode(impl_h)
        rec_h = decoder(ddu(z_h, c_h)).clamp(0, 1)
        loss_h = loss_fn.forward_stage1(rec_h, lq)

        # Clean branch
        impl_c = encoder(normalize_for_encoder(gt))
        z_c, c_c = codec.encode(impl_c)
        rec_c = decoder(ddu(z_c, c_c)).clamp(0, 1)
        loss_c = loss_fn.forward_stage1(rec_c, gt)

        hazy_loss_sum += float(loss_h.item()) * bs
        clean_loss_sum += float(loss_c.item()) * bs

        hazy_psnr_sum += compute_psnr(rec_h, lq) * bs
        clean_psnr_sum += compute_psnr(rec_c, gt) * bs

        hazy_ssim_sum += compute_ssim(rec_h, lq) * bs
        clean_ssim_sum += compute_ssim(rec_c, gt) * bs

        count += bs

    return {
        "hazy_loss": hazy_loss_sum / count,
        "clean_loss": clean_loss_sum / count,
        "hazy_psnr": hazy_psnr_sum / count,
        "clean_psnr": clean_psnr_sum / count,
        "hazy_ssim": hazy_ssim_sum / count,
        "clean_ssim": clean_ssim_sum / count,
    }


def train_stage1(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join("checkpoints", opt["name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    # encoder = ConvNextEncoder(pretrained=opt["network_encoder"]["pretrained"]).to(device)
    encoder = ConvNextEncoder(
        pretrained=opt["network_encoder"]["pretrained"],
        backbone=opt["network_encoder"].get("backbone", "convnext_small"),
    ).to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    impl_ch = opt["network_L"]["implicit_channels"]
    lat_ch = opt["network_L"]["latent_channels"]
    cee_ch = opt["network_L"]["cee_channels"]

    codec = DehazeDiffCodec(impl_ch, lat_ch, cee_ch).to(device)
    ddu = DimensionalDecompressionUnit(lat_ch, impl_ch, cee_ch).to(device)
    decoder = PixelShuffleDecoder(in_channels=impl_ch, upscale_factor=8).to(device)

    loss_fn = AILDFreqLoss(
        lambda_img=opt["loss"]["lambda_img"],
        lambda_lat=opt["loss"]["lambda_lat"],
        lambda_freq=opt["loss"]["lambda_freq"],
        lambda_freq_prior=opt["loss"]["lambda_freq_prior"],
        lambda_perc=opt["loss"]["lambda_perc"],
    ).to(device)

    _, train_dl = build_dataloader(opt, "train")
    _, val_dl = build_dataloader(opt, "val")

    params = list(codec.parameters()) + list(ddu.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=opt["train"]["lr_stage1"],
        betas=(opt["train"]["beta1"], opt["train"]["beta2"]),
    )

    best_clean_psnr = -1.0
    epochs = opt["train"]["stage1_epochs"]

    for epoch in range(1, epochs + 1):
        codec.train()
        ddu.train()
        decoder.train()

        hazy_loss_sum, clean_loss_sum = 0.0, 0.0
        hazy_psnr_sum, clean_psnr_sum = 0.0, 0.0
        hazy_ssim_sum, clean_ssim_sum = 0.0, 0.0
        count = 0

        for batch in train_dl:
            lq = batch["LQ"].to(device)
            gt = batch["GT"].to(device)
            bs = lq.size(0)

            with torch.no_grad():
                impl_h = encoder(normalize_for_encoder(lq))
                impl_c = encoder(normalize_for_encoder(gt))

            # Hazy autoencoding branch
            z_h, c_h = codec.encode(impl_h)
            rec_h = decoder(ddu(z_h, c_h)).clamp(0, 1)
            loss_h = loss_fn.forward_stage1(rec_h, lq)

            # Clean autoencoding branch
            z_c, c_c = codec.encode(impl_c)
            rec_c = decoder(ddu(z_c, c_c)).clamp(0, 1)
            loss_c = loss_fn.forward_stage1(rec_c, gt)

            loss = loss_h + loss_c

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            hazy_loss_sum += float(loss_h.item()) * bs
            clean_loss_sum += float(loss_c.item()) * bs

            hazy_psnr_sum += compute_psnr(rec_h.detach(), lq) * bs
            clean_psnr_sum += compute_psnr(rec_c.detach(), gt) * bs

            hazy_ssim_sum += compute_ssim(rec_h.detach(), lq) * bs
            clean_ssim_sum += compute_ssim(rec_c.detach(), gt) * bs

            count += bs

        train_hazy_loss = hazy_loss_sum / count
        train_clean_loss = clean_loss_sum / count
        train_hazy_psnr = hazy_psnr_sum / count
        train_clean_psnr = clean_psnr_sum / count
        train_hazy_ssim = hazy_ssim_sum / count
        train_clean_ssim = clean_ssim_sum / count

        if epoch % opt["train"]["val_freq"] == 0 or epoch == 1:
            val_stats = validate_stage1(encoder, codec, ddu, decoder, val_dl, loss_fn, device)

            print(
                f"[S1][{epoch:04d}/{epochs}] "
                f"train_hazy_psnr={train_hazy_psnr:.3f} train_clean_psnr={train_clean_psnr:.3f} | "
                f"val_hazy_psnr={val_stats['hazy_psnr']:.3f} val_clean_psnr={val_stats['clean_psnr']:.3f}"
            )

            if val_stats["clean_psnr"] > best_clean_psnr:
                best_clean_psnr = val_stats["clean_psnr"]
                save_checkpoint(
                    os.path.join(ckpt_dir, "stage1_best.pth"),
                    {
                        "epoch": epoch,
                        "codec": codec.state_dict(),
                        "ddu": ddu.state_dict(),
                        "decoder": decoder.state_dict(),
                        "clean_psnr": val_stats["clean_psnr"],
                        "hazy_psnr": val_stats["hazy_psnr"],
                    },
                )

    print(f"Best Stage1 clean PSNR: {best_clean_psnr:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    args = parser.parse_args()
    train_stage1(args.config)
