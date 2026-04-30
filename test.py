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
from models.modules.nafnet import ConditionalNAFNet
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder
from utils.checkpoint_utils import load_checkpoint
from utils.img_utils import normalize_for_encoder, save_image
from utils.metrics import compute_psnr, compute_ssim


def build_dataloader(opt, split="test"):
    ds = LQGTDataset(opt, split)
    dl = DataLoader(
        ds,
        batch_size=opt["datasets"][split]["batch_size"],
        shuffle=False,
        num_workers=opt["datasets"][split].get("n_workers", 0),
    )
    return ds, dl


def match_spatial(pred, ref):
    if pred.shape[-2:] != ref.shape[-2:]:
        pred = F.interpolate(pred, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return pred


def build_modules(opt, device):
    encoder = ConvNextEncoder(
        pretrained=opt["network_encoder"]["pretrained"],
        backbone=opt["network_encoder"].get("backbone", "convnext_small"),
    ).to(device)
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


@torch.no_grad()
def run_inference(encoder, codec, fdaa, net, ddu, decoder, lq, gt, alpha, latent_clip):
    impl_h = encoder(normalize_for_encoder(lq))
    impl_c = encoder(normalize_for_encoder(gt))

    z_h, c_h = codec.encode(impl_h)
    _, c_c = codec.encode(impl_c)

    c_mix = alpha * c_c + (1.0 - alpha) * c_h

    z_f, _, _, alpha_map = fdaa(z_h)
    t = torch.zeros((lq.shape[0],), device=lq.device, dtype=torch.long)
    z0_pred = net(z_f, z_f, t).clamp(-latent_clip, latent_clip)
    out = decoder(ddu(z0_pred, c_mix)).clamp(0, 1)
    return out, {
        "alpha_map_mean": float(alpha_map.mean().detach().cpu().item()),
        "context_alpha": float(alpha),
    }


@torch.no_grad()
def test(config_path, alpha_override=None):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join("checkpoints", opt["name"])
    results_dir = os.path.join("results", opt["name"])
    os.makedirs(results_dir, exist_ok=True)

    encoder, codec, fdaa, hag, net, ddu, decoder = build_modules(opt, device)
    merged_ckpt_path = os.path.join(
        ckpt_dir, opt.get("train", {}).get("merged_checkpoint_name", "final_best.pth")
    )
    if os.path.exists(merged_ckpt_path):
        ckpt = load_checkpoint(merged_ckpt_path, device)
        codec.load_state_dict(ckpt["codec"])
        ddu.load_state_dict(ckpt["ddu"])
        decoder.load_state_dict(ckpt["decoder"])
        fdaa.load_state_dict(ckpt["fdaa"])
        hag.load_state_dict(ckpt["hag"])
        net.load_state_dict(ckpt["net"])
        phase = ckpt.get("phase", "UNKNOWN")
        ckpt_alpha = float(ckpt.get("alpha", 0.0))
    else:
        s1_ckpt = load_checkpoint(os.path.join(ckpt_dir, "stage1_best.pth"), device)
        s2_ckpt = load_checkpoint(os.path.join(ckpt_dir, "stage2_best.pth"), device)
        codec.load_state_dict(s2_ckpt.get("codec", s1_ckpt["codec"]))
        ddu.load_state_dict(s2_ckpt.get("ddu", s1_ckpt["ddu"]))
        decoder.load_state_dict(s2_ckpt.get("decoder", s1_ckpt["decoder"]))
        fdaa.load_state_dict(s2_ckpt["fdaa"])
        hag.load_state_dict(s2_ckpt["hag"])
        net.load_state_dict(s2_ckpt["net"])
        phase = s2_ckpt.get("phase", "UNKNOWN")
        ckpt_alpha = float(s2_ckpt.get("alpha", 0.0))

    for m in [encoder, codec, ddu, decoder, fdaa, hag, net]:
        m.eval()

    latent_clip = float(opt["train"]["latent_clip"])
    test_ds, test_dl = build_dataloader(opt, "test")
    all_psnr, all_ssim, all_alpha = [], [], []
    alpha = ckpt_alpha if alpha_override is None else float(alpha_override)

    print(f"Loaded checkpoint phase: {phase}")
    print(f"Using context alpha: {alpha:.3f} (checkpoint alpha: {ckpt_alpha:.3f})")

    for i, batch in enumerate(test_dl):
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)
        pred, extra = run_inference(
            encoder=encoder,
            codec=codec,
            fdaa=fdaa,
            net=net,
            ddu=ddu,
            decoder=decoder,
            lq=lq,
            gt=gt,
            alpha=alpha,
            latent_clip=latent_clip,
        )
        pred = match_spatial(pred, gt)
        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_alpha.append(extra["alpha_map_mean"])
        save_image(os.path.join(results_dir, f"dehazed_{i:03d}.png"), pred)
        print(f"[{i + 1:03d}/{len(test_ds):03d}] PSNR={psnr:.3f} SSIM={ssim:.4f}")

    print(f"\nResults on {opt['name']}:")
    print(f"  PSNR : {sum(all_psnr) / len(all_psnr):.3f} dB")
    print(f"  SSIM : {sum(all_ssim) / len(all_ssim):.4f}")
    print(f"  Mean FDAA alpha : {sum(all_alpha) / len(all_alpha):.4f}")
    print(f"  Saved to: {results_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    parser.add_argument("--alpha_override", type=float, default=None)
    args = parser.parse_args()
    test(args.config, alpha_override=args.alpha_override)
