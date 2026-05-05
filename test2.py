import csv
import os

import cv2
import numpy as np
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
from utils.img_utils import normalize_for_encoder, save_image, tensor_to_uint8
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
def run_inference(encoder, codec, fdaa, hag, net, ddu, decoder, lq, latent_clip):
    impl_h = encoder(normalize_for_encoder(lq))
    z_h, c_h = codec.encode(impl_h)

    z_f, f_low, f_high, alpha_map = fdaa(z_h)
    _, hag_gate = hag(f_low, f_high)
    g = hag_gate.view(-1, 1, 1, 1)
    z_cond = g * z_f + (1.0 - g) * z_h

    t = torch.zeros((lq.shape[0],), device=lq.device, dtype=torch.long)
    z0_pred = net(z_cond, z_cond, t).clamp(-latent_clip, latent_clip)
    out = decoder(ddu(z0_pred, c_h)).clamp(0, 1)

    return out, {
        "alpha_map_mean": float(alpha_map.mean().detach().cpu().item()),
        "hag_gate_mean": float(hag_gate.mean().detach().cpu().item()),
    }


def draw_label(img, text):
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_comparison_panel(lq_tensor, pred_tensor, gt_tensor, psnr, ssim, alpha_mean, hag_mean):
    lq_img = tensor_to_uint8(lq_tensor)
    pred_img = tensor_to_uint8(pred_tensor)
    gt_img = tensor_to_uint8(gt_tensor)

    lq_panel = draw_label(lq_img, "Hazy")
    pred_panel = draw_label(
        pred_img,
        f"Dehazed  PSNR:{psnr:.2f}  SSIM:{ssim:.4f}  A:{alpha_mean:.3f}  H:{hag_mean:.3f}",
    )
    gt_panel = draw_label(gt_img, "Clear")

    spacer = np.full((lq_panel.shape[0], 12, 3), 255, dtype=np.uint8)
    return np.concatenate([lq_panel, spacer, pred_panel, spacer, gt_panel], axis=1)


def write_csv(csv_path, rows, summary):
    fieldnames = [
        "index",
        "lq_path",
        "gt_path",
        "output_path",
        "compare_path",
        "psnr",
        "ssim",
        "fdaa_alpha_mean",
        "hag_gate_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = os.path.splitext(csv_path)[0] + "_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


@torch.no_grad()
def test(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join("checkpoints", opt["name"])
    results_dir = os.path.join("results", opt["name"])
    compare_dir = os.path.join("results", f"{opt['name']}_compare")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

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

    for m in [encoder, codec, ddu, decoder, fdaa, hag, net]:
        m.eval()

    latent_clip = float(opt["train"]["latent_clip"])
    test_ds, test_dl = build_dataloader(opt, "test")
    all_psnr, all_ssim, all_alpha, all_hag = [], [], [], []
    rows = []

    print(f"Loaded checkpoint phase: {phase}")
    print("Inference mode: hazy-only with comparison panels")

    for i, batch in enumerate(test_dl):
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        pred, extra = run_inference(
            encoder=encoder,
            codec=codec,
            fdaa=fdaa,
            hag=hag,
            net=net,
            ddu=ddu,
            decoder=decoder,
            lq=lq,
            latent_clip=latent_clip,
        )
        pred = match_spatial(pred, gt)

        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)

        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_alpha.append(extra["alpha_map_mean"])
        all_hag.append(extra["hag_gate_mean"])

        output_path = os.path.join(results_dir, f"dehazed_{i:03d}.png")
        compare_path = os.path.join(compare_dir, f"compare_{i:03d}_psnr_{psnr:.2f}_ssim_{ssim:.4f}.png")

        save_image(output_path, pred)

        panel = make_comparison_panel(
            lq.cpu(),
            pred.cpu(),
            gt.cpu(),
            psnr,
            ssim,
            extra["alpha_map_mean"],
            extra["hag_gate_mean"],
        )
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.imwrite(compare_path, panel_bgr)

        lq_path = batch["LQ_path"][0] if isinstance(batch["LQ_path"], list) else batch["LQ_path"]
        gt_path = batch["GT_path"][0] if isinstance(batch["GT_path"], list) else batch["GT_path"]

        rows.append(
            {
                "index": i,
                "lq_path": lq_path,
                "gt_path": gt_path,
                "output_path": output_path,
                "compare_path": compare_path,
                "psnr": f"{psnr:.6f}",
                "ssim": f"{ssim:.6f}",
                "fdaa_alpha_mean": f"{extra['alpha_map_mean']:.6f}",
                "hag_gate_mean": f"{extra['hag_gate_mean']:.6f}",
            }
        )

        print(f"[{i + 1:03d}/{len(test_ds):03d}] PSNR={psnr:.3f} SSIM={ssim:.4f}")

    mean_psnr = sum(all_psnr) / len(all_psnr)
    mean_ssim = sum(all_ssim) / len(all_ssim)
    mean_alpha = sum(all_alpha) / len(all_alpha)
    mean_hag = sum(all_hag) / len(all_hag)

    summary = {
        "dataset": opt["name"],
        "mode": "hazy_only_compare",
        "phase": phase,
        "mean_psnr": f"{mean_psnr:.6f}",
        "mean_ssim": f"{mean_ssim:.6f}",
        "mean_fdaa_alpha": f"{mean_alpha:.6f}",
        "mean_hag_gate": f"{mean_hag:.6f}",
        "num_images": len(rows),
    }

    csv_path = os.path.join(compare_dir, f"{opt['name']}_compare_metrics.csv")
    write_csv(csv_path, rows, summary)

    print(f"\nResults on {opt['name']}:")
    print(f"  PSNR : {mean_psnr:.3f} dB")
    print(f"  SSIM : {mean_ssim:.4f}")
    print(f"  Mean FDAA alpha : {mean_alpha:.4f}")
    print(f"  Mean HAG gate : {mean_hag:.4f}")
    print(f"  Dehazed images : {results_dir}/")
    print(f"  Compare panels : {compare_dir}/")
    print(f"  CSV : {csv_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    args = parser.parse_args()
    test(args.config)
