import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.lqgt_dataset import LQGTDataset
from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder

from utils.checkpoint_utils import load_checkpoint
from utils.img_utils import normalize_for_encoder, save_image
from utils.metrics import compute_psnr, compute_ssim


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


def pad_to_multiple(x, multiple=32):
    _, _, h, w = x.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, (h, w)

    x = F.pad(
        x,
        pad=(0, pad_w, 0, pad_h),
        mode="reflect",
    )

    return x, (h, w)


def crop_to_original(x, original_hw):
    h, w = original_hw
    return x[..., :h, :w]


def match_spatial_pair(pred, target):
    if pred.shape[-2:] == target.shape[-2:]:
        return pred, target

    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])

    return pred[..., :h, :w], target[..., :h, :w]


def build_modules(opt, device):
    encoder = ConvNextEncoder(
        pretrained=bool(opt["network_encoder"]["pretrained"]),
        backbone=opt["network_encoder"].get("backbone", "convnext_small"),
    ).to(device)

    impl_ch = int(opt["network_L"]["implicit_channels"])
    lat_ch = int(opt["network_L"]["latent_channels"])
    cee_ch = int(opt["network_L"]["cee_channels"])

    codec = DehazeDiffCodec(
        impl_ch,
        lat_ch,
        cee_ch,
    ).to(device)

    ddu = DimensionalDecompressionUnit(
        lat_ch,
        impl_ch,
        cee_ch,
    ).to(device)

    decoder = PixelShuffleDecoder(
        in_channels=impl_ch,
        upscale_factor=8,
    ).to(device)

    return encoder, codec, ddu, decoder


def reconstruct_from_latent(z, c, ddu, decoder):
    return decoder(ddu(z, c)).clamp(0, 1)


def save_compare_panel(save_path, lq, dehaze, gt):
    lq, dehaze = match_spatial_pair(lq, dehaze)
    gt, dehaze = match_spatial_pair(gt, dehaze)

    h = min(lq.shape[-2], dehaze.shape[-2], gt.shape[-2])
    w = min(lq.shape[-1], dehaze.shape[-1], gt.shape[-1])

    lq = lq[..., :h, :w]
    dehaze = dehaze[..., :h, :w]
    gt = gt[..., :h, :w]

    panel = torch.cat([lq, dehaze, gt], dim=-1)
    save_image(save_path, panel)


@torch.no_grad()
def test_stage1(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = os.path.join(
        "checkpoints",
        opt["name"],
    )

    ckpt_path = os.path.join(
        ckpt_dir,
        "stage1_best.pth",
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing Stage1 checkpoint: {ckpt_path}")

    results_root = os.path.join(
        "results_stage1",
        opt["name"],
    )

    dehaze_dir = os.path.join(results_root, "dehaze_direct")
    clean_dir = os.path.join(results_root, "clean_recon")
    cross_dir = os.path.join(results_root, "cross_oracle")
    compare_dir = os.path.join(results_root, "compare")

    os.makedirs(dehaze_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(cross_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    encoder, codec, ddu, decoder = build_modules(opt, device)

    ckpt = load_checkpoint(ckpt_path, device)

    codec.load_state_dict(ckpt["codec"])
    ddu.load_state_dict(ckpt["ddu"])
    decoder.load_state_dict(ckpt["decoder"])

    for m in [encoder, codec, ddu, decoder]:
        m.eval()

    print(f"Loaded Stage1 checkpoint: {ckpt_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Selection metric: {ckpt.get('selection_metric', 'unknown')}")
    print(f"Checkpoint dehaze PSNR: {ckpt.get('dehaze_psnr', 'unknown')}")
    print(f"Checkpoint dehaze SSIM: {ckpt.get('dehaze_ssim', 'unknown')}")
    print(f"Checkpoint clean PSNR: {ckpt.get('clean_psnr', 'unknown')}")
    print(f"Checkpoint cross PSNR: {ckpt.get('cross_psnr', 'unknown')}")

    _, test_dl = build_dataloader(opt, "test")

    dehaze_psnr_all = []
    dehaze_ssim_all = []

    clean_psnr_all = []
    clean_ssim_all = []

    cross_psnr_all = []
    cross_ssim_all = []

    for i, batch in enumerate(tqdm(test_dl)):
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        lq_pad, original_hw = pad_to_multiple(lq, multiple=32)
        gt_pad, _ = pad_to_multiple(gt, multiple=32)

        impl_h = encoder(normalize_for_encoder(lq_pad))
        impl_c = encoder(normalize_for_encoder(gt_pad))

        z_h, c_h = codec.encode(impl_h)
        z_c, c_c = codec.encode(impl_c)

        # 1. Deployable dehazing path:
        # hazy latent + hazy context -> clean image
        dehaze = reconstruct_from_latent(
            z_h,
            c_h,
            ddu,
            decoder,
        )
        dehaze = crop_to_original(dehaze, original_hw)

        # 2. Clean reconstruction diagnostic:
        # clean latent + clean context -> clean image
        clean_rec = reconstruct_from_latent(
            z_c,
            c_c,
            ddu,
            decoder,
        )
        clean_rec = crop_to_original(clean_rec, original_hw)

        # 3. Oracle/cross diagnostic:
        # clean latent + hazy context -> clean image
        cross_rec = reconstruct_from_latent(
            z_c,
            c_h,
            ddu,
            decoder,
        )
        cross_rec = crop_to_original(cross_rec, original_hw)

        dehaze, gt_eval = match_spatial_pair(dehaze, gt)
        clean_rec, gt_clean = match_spatial_pair(clean_rec, gt)
        cross_rec, gt_cross = match_spatial_pair(cross_rec, gt)
        lq_eval, _ = match_spatial_pair(lq, gt_eval)

        dehaze_psnr = compute_psnr(dehaze.float(), gt_eval.float())
        dehaze_ssim = compute_ssim(dehaze.float(), gt_eval.float())

        clean_psnr = compute_psnr(clean_rec.float(), gt_clean.float())
        clean_ssim = compute_ssim(clean_rec.float(), gt_clean.float())

        cross_psnr = compute_psnr(cross_rec.float(), gt_cross.float())
        cross_ssim = compute_ssim(cross_rec.float(), gt_cross.float())

        dehaze_psnr_all.append(dehaze_psnr)
        dehaze_ssim_all.append(dehaze_ssim)

        clean_psnr_all.append(clean_psnr)
        clean_ssim_all.append(clean_ssim)

        cross_psnr_all.append(cross_psnr)
        cross_ssim_all.append(cross_ssim)

        name = os.path.basename(batch["LQ_path"][0])
        stem = os.path.splitext(name)[0]

        save_image(
            os.path.join(dehaze_dir, f"{stem}_dehaze.png"),
            dehaze,
        )

        save_image(
            os.path.join(clean_dir, f"{stem}_clean_recon.png"),
            clean_rec,
        )

        save_image(
            os.path.join(cross_dir, f"{stem}_cross_oracle.png"),
            cross_rec,
        )

        save_compare_panel(
            os.path.join(compare_dir, f"{stem}_compare.png"),
            lq_eval,
            dehaze,
            gt_eval,
        )

        print(
            f"[{i + 1:03d}/{len(test_dl):03d}] "
            f"{name} | "
            f"dehaze={dehaze_psnr:.3f}/{dehaze_ssim:.4f} | "
            f"clean={clean_psnr:.3f}/{clean_ssim:.4f} | "
            f"cross_oracle={cross_psnr:.3f}/{cross_ssim:.4f} | "
            f"fullres={original_hw[0]}x{original_hw[1]}"
        )

    print(f"\nStage1 Full-Resolution Test Results on {opt['name']}:")

    print(
        f"  Deployable dehaze PSNR / SSIM : "
        f"{sum(dehaze_psnr_all) / len(dehaze_psnr_all):.3f} / "
        f"{sum(dehaze_ssim_all) / len(dehaze_ssim_all):.4f}"
    )

    print(
        f"  Clean recon PSNR / SSIM       : "
        f"{sum(clean_psnr_all) / len(clean_psnr_all):.3f} / "
        f"{sum(clean_ssim_all) / len(clean_ssim_all):.4f}"
    )

    print(
        f"  Cross/oracle PSNR / SSIM      : "
        f"{sum(cross_psnr_all) / len(cross_psnr_all):.3f} / "
        f"{sum(cross_ssim_all) / len(cross_ssim_all):.4f}"
    )

    print(f"  Dehazed images : {dehaze_dir}/")
    print(f"  Compare panels : {compare_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/nh_haze.yml",
    )

    args = parser.parse_args()

    test_stage1(args.config)