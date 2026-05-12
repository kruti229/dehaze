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
from models.modules.medium_latent_mapper import MediumLatentResidualMapper
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder

from utils.checkpoint_utils import load_checkpoint
from utils.img_utils import normalize_for_encoder, save_image
from utils.metrics import compute_psnr, compute_ssim


def build_dataloader(opt, split):
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


def match_spatial(pred, target):
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

    mapper_cfg = opt.get("network_mapper", {})

    mapper = MediumLatentResidualMapper(
        latent_channels=lat_ch,
        width=int(mapper_cfg.get("width", 96)),
        num_blocks=int(mapper_cfg.get("num_blocks", 10)),
        expansion=int(mapper_cfg.get("expansion", 2)),
        use_se=bool(mapper_cfg.get("use_se", True)),
    ).to(device)

    return encoder, codec, mapper, ddu, decoder


def reconstruct_from_latent(z, c, ddu, decoder):
    return decoder(ddu(z, c)).clamp(0, 1)


@torch.no_grad()
def predict_stage2(mapper, z_h, opt):
    latent_clip = float(opt["train"].get("latent_clip", 3.0))
    residual_scale = float(opt["train"].get("stage2_residual_scale", 2.0))

    delta_raw = mapper(z_h, z_h)
    delta = residual_scale * torch.tanh(delta_raw)

    z_pred = (z_h + delta).clamp(
        -latent_clip,
        latent_clip,
    )

    return z_pred, delta


def load_checkpoint_into_modules(ckpt, codec, mapper, ddu, decoder):
    codec.load_state_dict(ckpt["codec"], strict=True)
    ddu.load_state_dict(ckpt["ddu"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    if "mapper" in ckpt:
        mapper.load_state_dict(ckpt["mapper"], strict=True)
        return "stage2_mapper"

    return "stage1_dehaze"


@torch.no_grad()
def run_stage1_dehaze(encoder, codec, ddu, decoder, lq):
    impl_h = encoder(normalize_for_encoder(lq))
    z_h, c_h = codec.encode(impl_h)

    pred = reconstruct_from_latent(
        z_h,
        c_h,
        ddu,
        decoder,
    )

    return pred, None


@torch.no_grad()
def run_stage2_mapper(encoder, codec, mapper, ddu, decoder, lq, opt):
    impl_h = encoder(normalize_for_encoder(lq))
    z_h, c_h = codec.encode(impl_h)

    z_pred, delta = predict_stage2(
        mapper=mapper,
        z_h=z_h,
        opt=opt,
    )

    pred = reconstruct_from_latent(
        z_pred,
        c_h,
        ddu,
        decoder,
    )

    return pred, delta


def save_compare_panel(save_path, lq, pred, gt):
    lq, pred = match_spatial(lq, pred)
    gt, pred = match_spatial(gt, pred)

    h = min(lq.shape[-2], pred.shape[-2], gt.shape[-2])
    w = min(lq.shape[-1], pred.shape[-1], gt.shape[-1])

    lq = lq[..., :h, :w]
    pred = pred[..., :h, :w]
    gt = gt[..., :h, :w]

    panel = torch.cat([lq, pred, gt], dim=-1)
    save_image(save_path, panel)


@torch.no_grad()
def test_fullres(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_dl = build_dataloader(opt, "test")

    encoder, codec, mapper, ddu, decoder = build_modules(opt, device)

    ckpt_dir = os.path.join("checkpoints", opt["name"])

    stage2_path = os.path.join(
        ckpt_dir,
        opt["train"].get("merged_checkpoint_name", "final_best.pth"),
    )

    stage1_path = os.path.join(
        ckpt_dir,
        "stage1_best.pth",
    )

    if os.path.exists(stage2_path):
        ckpt_path = stage2_path
    elif os.path.exists(stage1_path):
        ckpt_path = stage1_path
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {ckpt_dir}. Expected final_best.pth or stage1_best.pth."
        )

    ckpt = load_checkpoint(ckpt_path, device)

    mode = load_checkpoint_into_modules(
        ckpt=ckpt,
        codec=codec,
        mapper=mapper,
        ddu=ddu,
        decoder=decoder,
    )

    for m in [encoder, codec, mapper, ddu, decoder]:
        m.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Loaded phase: {ckpt.get('phase', 'unknown')}")
    print(f"Checkpoint metrics: {ckpt.get('metrics', {})}")

    if mode == "stage2_mapper":
        print("Using full-resolution inference path: Stage2 direct latent mapper")
    else:
        print("Using full-resolution inference path: Stage1 direct dehaze z_h + c_h")

    results_dir = os.path.join("results_fullres", opt["name"], "dehazed")
    compare_dir = os.path.join("results_fullres", opt["name"], "compare")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    psnr_sum = 0.0
    ssim_sum = 0.0
    score_sum = 0.0
    count = 0

    for idx, batch in enumerate(tqdm(test_dl)):
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        lq_pad, original_hw = pad_to_multiple(lq, multiple=32)

        if mode == "stage2_mapper":
            pred_pad, delta = run_stage2_mapper(
                encoder=encoder,
                codec=codec,
                mapper=mapper,
                ddu=ddu,
                decoder=decoder,
                lq=lq_pad,
                opt=opt,
            )
        else:
            pred_pad, delta = run_stage1_dehaze(
                encoder=encoder,
                codec=codec,
                ddu=ddu,
                decoder=decoder,
                lq=lq_pad,
            )

        pred = crop_to_original(pred_pad, original_hw)

        pred, gt_eval = match_spatial(pred, gt)
        lq_eval, _ = match_spatial(lq, gt_eval)

        psnr = compute_psnr(pred.float(), gt_eval.float())
        ssim = compute_ssim(pred.float(), gt_eval.float())
        score = psnr + 10.0 * ssim

        psnr_sum += psnr
        ssim_sum += ssim
        score_sum += score
        count += 1

        name = os.path.basename(batch["LQ_path"][0])
        stem = os.path.splitext(name)[0]

        save_image(
            os.path.join(results_dir, f"{stem}_dehazed.png"),
            pred,
        )

        save_compare_panel(
            os.path.join(compare_dir, f"{stem}_compare.png"),
            lq_eval,
            pred,
            gt_eval,
        )

        if delta is not None:
            delta_abs = float(delta.abs().mean().item())
        else:
            delta_abs = 0.0

        print(
            f"[{idx + 1:03d}/{len(test_dl):03d}] "
            f"{name} | "
            f"PSNR={psnr:.3f} "
            f"SSIM={ssim:.4f} "
            f"Score={score:.3f} "
            f"DeltaAbs={delta_abs:.5f} "
            f"FullRes={original_hw[0]}x{original_hw[1]}"
        )

    mean_psnr = psnr_sum / max(count, 1)
    mean_ssim = ssim_sum / max(count, 1)
    mean_score = score_sum / max(count, 1)

    print("")
    print(f"Full-resolution results on {opt['name']}:")
    print(f"  PSNR  : {mean_psnr:.3f} dB")
    print(f"  SSIM  : {mean_ssim:.4f}")
    print(f"  Score : {mean_score:.3f}")
    print(f"  Dehazed images : {results_dir}/")
    print(f"  Compare panels : {compare_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")

    args = parser.parse_args()

    test_fullres(args.config)