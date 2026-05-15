import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
import numpy as np

from data.lqgt_dataset import LQGTDataset

from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder

from models.modules.physical_guidance import (
    TransmissionEstimator,
    PhysicalContextFusion,
    atmospheric_reconstruction,
    transmission_smoothness_loss,
)

from models.modules.color_losses import (
    RGBHSVColorLoss,
    ChromaConsistencyLoss,
    GrayPenaltyLoss,
)

from utils.checkpoint_utils import save_checkpoint, load_checkpoint
from utils.img_utils import normalize_for_encoder
from utils.metrics import compute_psnr, compute_ssim

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
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
        pretrained=bool(opt["network_encoder"].get("pretrained", True)),
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

    physical_opt = opt.get("physical", {})

    phys_t = TransmissionEstimator(
        base_ch=int(physical_opt.get("base_ch", 32)),
        t_min=float(physical_opt.get("t_min", 0.05)),
    ).to(device)

    phys_fusion = PhysicalContextFusion(
        cee_ch=cee_ch,
    ).to(device)

    return encoder, codec, ddu, decoder, phys_t, phys_fusion


def reconstruct_from_latent(z, c, ddu, decoder):
    return decoder(ddu(z, c)).clamp(0.0, 1.0)


def charbonnier_loss(pred, target, eps=1.0e-3):
    return torch.sqrt((pred - target) * (pred - target) + eps * eps).mean()


def tv_loss(img):
    dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return dh + dw


def frequency_loss(pred, target):
    pred_fft = torch.fft.fft2(pred.float(), norm="ortho")
    target_fft = torch.fft.fft2(target.float(), norm="ortho")

    pred_mag = torch.log1p(torch.abs(pred_fft))
    target_mag = torch.log1p(torch.abs(target_fft))

    return F.l1_loss(pred_mag, target_mag)


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


def structural_loss(pred, target):
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)

    h, w = pred.shape[-2:]

    try:
        from pytorch_msssim import ms_ssim, ssim

        if h >= 160 and w >= 160:
            return 1.0 - ms_ssim(
                pred,
                target,
                data_range=1.0,
                size_average=True,
                win_size=7,
            )

        return 1.0 - ssim(
            pred,
            target,
            data_range=1.0,
            size_average=True,
            win_size=7,
        )

    except Exception:
        return torch.tensor(
            0.0,
            device=pred.device,
            dtype=pred.dtype,
        )


class Stage1Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()

        loss_opt = opt.get("loss", {})

        self.lambda_dehaze = float(loss_opt.get("lambda_dehaze_recon", 2.0))
        self.lambda_hazy = float(loss_opt.get("lambda_hazy_recon", 0.0))
        self.lambda_clean = float(loss_opt.get("lambda_clean_recon", 0.25))
        self.lambda_cross = float(loss_opt.get("lambda_cross", 0.6))

        self.lambda_z = float(loss_opt.get("lambda_z_align", 0.8))
        self.lambda_cee = float(loss_opt.get("lambda_cee_align", 0.0))

        self.lambda_ssim = float(loss_opt.get("lambda_ssim", 0.35))
        self.lambda_edge = float(loss_opt.get("lambda_edge", 0.03))
        self.lambda_hsv = float(loss_opt.get("lambda_hsv", 0.30))
        self.lambda_chroma = float(loss_opt.get("lambda_chroma", 0.40))
        self.lambda_gray = float(loss_opt.get("lambda_gray_penalty", 0.15))
        self.lambda_freq = float(loss_opt.get("lambda_freq", 0.02))
        self.lambda_tv = float(loss_opt.get("lambda_tv", 0.0003))

        self.lambda_phy = float(loss_opt.get("lambda_phy", 0.02))
        self.lambda_t_smooth = float(loss_opt.get("lambda_t_smooth", 0.001))

        self.hsv_color = RGBHSVColorLoss()
        self.chroma_loss = ChromaConsistencyLoss()
        self.gray_loss = GrayPenaltyLoss()

    def image_loss(self, pred, target):
        pred, target = match_spatial(pred, target)

        l_rec = charbonnier_loss(pred, target)
        l_ssim = structural_loss(pred, target)
        l_edge = edge_loss(pred, target)
        l_hsv = self.hsv_color(pred, target)
        l_chroma = self.chroma_loss(pred, target)
        l_gray = self.gray_loss(pred, target)
        l_freq = frequency_loss(pred, target)
        l_tv = tv_loss(pred)

        total = (
            l_rec
            + self.lambda_ssim * l_ssim
            + self.lambda_edge * l_edge
            + self.lambda_hsv * l_hsv
            + self.lambda_chroma * l_chroma
            + self.lambda_gray * l_gray
            + self.lambda_freq * l_freq
            + self.lambda_tv * l_tv
        )

        parts = {
            "rec": l_rec,
            "ssim": l_ssim,
            "edge": l_edge,
            "hsv": l_hsv,
            "chroma": l_chroma,
            "gray": l_gray,
            "freq": l_freq,
            "tv": l_tv,
        }

        return total, parts

    def forward(
        self,
        img_dehaze,
        img_hazy_rec,
        img_clean_rec,
        img_cross,
        hazy,
        clean,
        z_h,
        z_c,
        c_h,
        c_c,
        t_h=None,
        phy_recon=None,
    ):
        dehaze_loss, dehaze_parts = self.image_loss(img_dehaze, clean)
        clean_loss, _ = self.image_loss(img_clean_rec, clean)
        cross_loss, _ = self.image_loss(img_cross, clean)

        if self.lambda_hazy > 0:
            hazy_loss, _ = self.image_loss(img_hazy_rec, hazy)
        else:
            hazy_loss = torch.tensor(0.0, device=clean.device, dtype=clean.dtype)

        z_loss = F.l1_loss(z_h, z_c.detach())

        if self.lambda_cee > 0:
            cee_loss = F.l1_loss(c_h, c_c.detach())
        else:
            cee_loss = torch.tensor(0.0, device=clean.device, dtype=clean.dtype)

        if self.lambda_phy > 0 and phy_recon is not None:
            phy_recon, hazy_phy = match_spatial(phy_recon, hazy)
            phy_loss = charbonnier_loss(phy_recon, hazy_phy)
        else:
            phy_loss = torch.tensor(0.0, device=clean.device, dtype=clean.dtype)

        if self.lambda_t_smooth > 0 and t_h is not None:
            t_smooth = transmission_smoothness_loss(t_h)
        else:
            t_smooth = torch.tensor(0.0, device=clean.device, dtype=clean.dtype)

        total = (
            self.lambda_dehaze * dehaze_loss
            + self.lambda_hazy * hazy_loss
            + self.lambda_clean * clean_loss
            + self.lambda_cross * cross_loss
            + self.lambda_z * z_loss
            + self.lambda_cee * cee_loss
            + self.lambda_phy * phy_loss
            + self.lambda_t_smooth * t_smooth
        )

        losses = {
            "total": total,
            "dehaze": dehaze_loss,
            "hazy": hazy_loss,
            "clean": clean_loss,
            "cross": cross_loss,
            "z": z_loss,
            "cee": cee_loss,
            "phy": phy_loss,
            "t_smooth": t_smooth,
            "ssim": dehaze_parts["ssim"],
            "edge": dehaze_parts["edge"],
            "hsv": dehaze_parts["hsv"],
            "chroma": dehaze_parts["chroma"],
            "gray": dehaze_parts["gray"],
            "freq": dehaze_parts["freq"],
        }

        return total, losses


@torch.no_grad()
def validate(
    encoder,
    codec,
    ddu,
    decoder,
    phys_t,
    phys_fusion,
    val_loader,
    device,
):
    encoder.eval()
    codec.eval()
    ddu.eval()
    decoder.eval()
    phys_t.eval()
    phys_fusion.eval()

    sums = {
        "dehaze_psnr": 0.0,
        "dehaze_ssim": 0.0,
        "dehaze_score": 0.0,
        "hazy_recon_psnr": 0.0,
        "hazy_recon_ssim": 0.0,
        "clean_psnr": 0.0,
        "clean_ssim": 0.0,
        "cross_psnr": 0.0,
        "cross_ssim": 0.0,
        "z_align": 0.0,
        "cee_align": 0.0,
        "t_mean": 0.0,
        "A_mean": 0.0,
    }

    count = 0

    for batch in val_loader:
        hazy = batch["LQ"].to(device)
        clean = batch["GT"].to(device)

        bs = hazy.size(0)

        f_h = encoder(normalize_for_encoder(hazy))
        f_c = encoder(normalize_for_encoder(clean))

        z_h, c_h = codec.encode(f_h)
        z_c, c_c = codec.encode(f_c)

        t_h, A_h = phys_t(hazy)
        c_h_guided = phys_fusion(c_h, t_h)

        if count == 0:
            print(
                "[LatentStats] "
                f"z_h mean/std/abs={z_h.mean().item():.5f}/{z_h.std().item():.5f}/{z_h.abs().mean().item():.5f} | "
                f"z_c mean/std/abs={z_c.mean().item():.5f}/{z_c.std().item():.5f}/{z_c.abs().mean().item():.5f} | "
                f"c_h mean/std/abs={c_h.mean().item():.5f}/{c_h.std().item():.5f}/{c_h.abs().mean().item():.5f} | "
                f"c_guided mean/std/abs={c_h_guided.mean().item():.5f}/{c_h_guided.std().item():.5f}/{c_h_guided.abs().mean().item():.5f} | "
                f"t mean/std={t_h.mean().item():.5f}/{t_h.std().item():.5f}"
            )

        img_dehaze = reconstruct_from_latent(z_h, c_h_guided, ddu, decoder)
        img_hazy = reconstruct_from_latent(z_h, c_h, ddu, decoder)
        img_clean = reconstruct_from_latent(z_c, c_c, ddu, decoder)
        img_cross = reconstruct_from_latent(z_c, c_h_guided, ddu, decoder)

        img_dehaze, clean_eval = match_spatial(img_dehaze, clean)
        img_hazy, hazy_eval = match_spatial(img_hazy, hazy)
        img_clean, clean_eval2 = match_spatial(img_clean, clean)
        img_cross, clean_eval3 = match_spatial(img_cross, clean)

        dehaze_psnr = compute_psnr(img_dehaze.float(), clean_eval.float())
        dehaze_ssim = compute_ssim(img_dehaze.float(), clean_eval.float())
        dehaze_score = dehaze_psnr + 10.0 * dehaze_ssim

        hazy_psnr = compute_psnr(img_hazy.float(), hazy_eval.float())
        hazy_ssim = compute_ssim(img_hazy.float(), hazy_eval.float())

        clean_psnr = compute_psnr(img_clean.float(), clean_eval2.float())
        clean_ssim = compute_ssim(img_clean.float(), clean_eval2.float())

        cross_psnr = compute_psnr(img_cross.float(), clean_eval3.float())
        cross_ssim = compute_ssim(img_cross.float(), clean_eval3.float())

        sums["dehaze_psnr"] += dehaze_psnr * bs
        sums["dehaze_ssim"] += dehaze_ssim * bs
        sums["dehaze_score"] += dehaze_score * bs
        sums["hazy_recon_psnr"] += hazy_psnr * bs
        sums["hazy_recon_ssim"] += hazy_ssim * bs
        sums["clean_psnr"] += clean_psnr * bs
        sums["clean_ssim"] += clean_ssim * bs
        sums["cross_psnr"] += cross_psnr * bs
        sums["cross_ssim"] += cross_ssim * bs
        sums["z_align"] += float(F.l1_loss(z_h, z_c).item()) * bs
        sums["cee_align"] += float(F.l1_loss(c_h, c_c).item()) * bs
        sums["t_mean"] += float(t_h.mean().item()) * bs
        sums["A_mean"] += float(A_h.mean().item()) * bs

        count += bs

    return {k: v / count for k, v in sums.items()}


def save_stage1_checkpoint(
    path,
    epoch,
    codec,
    ddu,
    decoder,
    phys_t,
    phys_fusion,
    metrics,
    selection_metric,
    optimizer=None,
    scheduler=None,
    best_score=None,
):
    state = {
        "epoch": epoch,
        "codec": codec.state_dict(),
        "ddu": ddu.state_dict(),
        "decoder": decoder.state_dict(),
        "phys_t": phys_t.state_dict(),
        "phys_fusion": phys_fusion.state_dict(),
        "uses_physical_guidance": True,
        "phase": "AILD_FREQ_STAGE1_ZALIGN_CHROMA_PHYSICAL",
        "selection_metric": selection_metric,
        "metrics": metrics,
        "best_score": best_score,
        "dehaze_psnr": metrics["dehaze_psnr"],
        "dehaze_ssim": metrics["dehaze_ssim"],
        "dehaze_score": metrics["dehaze_score"],
        "clean_psnr": metrics["clean_psnr"],
        "clean_ssim": metrics["clean_ssim"],
        "cross_psnr": metrics["cross_psnr"],
        "cross_ssim": metrics["cross_ssim"],
        "hazy_recon_psnr": metrics["hazy_recon_psnr"],
        "hazy_recon_ssim": metrics["hazy_recon_ssim"],
        "z_align": metrics["z_align"],
        "cee_align": metrics["cee_align"],
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    save_checkpoint(path, state)


def try_resume(
    ckpt_dir,
    device,
    codec,
    ddu,
    decoder,
    phys_t,
    phys_fusion,
    optimizer,
    scheduler,
):
    resume_path = os.path.join(ckpt_dir, "stage1_last.pth")

    if not os.path.exists(resume_path):
        return 1, -1.0

    ckpt = load_checkpoint(resume_path, device)

    codec.load_state_dict(ckpt["codec"])
    ddu.load_state_dict(ckpt["ddu"])
    decoder.load_state_dict(ckpt["decoder"])

    if "phys_t" in ckpt:
        phys_t.load_state_dict(ckpt["phys_t"])

    if "phys_fusion" in ckpt:
        phys_fusion.load_state_dict(ckpt["phys_fusion"])

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = float(ckpt.get("best_score", ckpt.get("dehaze_score", -1.0)))

    print(f"[Resume] Loaded {resume_path}")
    print(f"[Resume] Continuing from epoch {start_epoch}")
    print(f"[Resume] Best score so far: {best_score:.4f}")

    return start_epoch, best_score


def train_stage1(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    set_seed(int(opt["train"].get("seed", 42)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = os.path.join("checkpoints", opt["name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    encoder, codec, ddu, decoder, phys_t, phys_fusion = build_modules(opt, device)

    freeze_module(encoder)

    _, train_loader = build_dataloader(opt, "train")
    _, val_loader = build_dataloader(opt, "val")

    criterion = Stage1Loss(opt).to(device)

    params = (
        list(codec.parameters())
        + list(ddu.parameters())
        + list(decoder.parameters())
        + list(phys_t.parameters())
        + list(phys_fusion.parameters())
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=float(opt["train"].get("lr_stage1", 1.0e-4)),
        betas=(
            float(opt["train"].get("beta1", 0.9)),
            float(opt["train"].get("beta2", 0.99)),
        ),
        weight_decay=float(opt["train"].get("weight_decay", 1.0e-4)),
    )

    epochs = int(opt["train"].get("stage1_epochs", 700))
    val_freq = int(opt["train"].get("val_freq", 25))
    save_freq = int(opt["train"].get("save_freq", 50))
    grad_clip = float(opt["train"].get("grad_clip", 1.0))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=epochs,
    #     eta_min=float(opt["train"].get("min_lr_stage1", 1.0e-6)),
    # )


    warmup_epochs = int(opt["train"].get("warmup_epochs_stage1", 50))

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=float(opt["train"].get("min_lr_stage1", 5.0e-7)),
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    resume_enabled = bool(opt["train"].get("resume_stage1", True))

    if resume_enabled:
        start_epoch, best_score = try_resume(
            ckpt_dir=ckpt_dir,
            device=device,
            codec=codec,
            ddu=ddu,
            decoder=decoder,
            phys_t=phys_t,
            phys_fusion=phys_fusion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        start_epoch = 1
        best_score = -1.0

    best_psnr = -1.0
    best_ssim = -1.0

    loss_opt = opt.get("loss", {})

    print(
        f"[Stage1-ZAlign-Chroma] Training {opt['name']} | "
        f"epochs={epochs} | "
        f"start_epoch={start_epoch} | "
        f"lambda_dehaze={loss_opt.get('lambda_dehaze_recon')} | "
        f"lambda_cross={loss_opt.get('lambda_cross')} | "
        f"lambda_z={loss_opt.get('lambda_z_align')} | "
        f"lambda_hsv={loss_opt.get('lambda_hsv')} | "
        f"lambda_chroma={loss_opt.get('lambda_chroma')} | "
        f"lambda_gray={loss_opt.get('lambda_gray_penalty')} | "
        f"lambda_phy={loss_opt.get('lambda_phy')}"
    )

    for epoch in range(start_epoch, epochs + 1):
        encoder.eval()
        codec.train()
        ddu.train()
        decoder.train()
        phys_t.train()
        phys_fusion.train()
        criterion.train()

        sums = {
            "loss": 0.0,
            "dehaze": 0.0,
            "hazy": 0.0,
            "clean": 0.0,
            "cross": 0.0,
            "z": 0.0,
            "phy": 0.0,
            "t_smooth": 0.0,
            "ssim": 0.0,
            "edge": 0.0,
            "hsv": 0.0,
            "chroma": 0.0,
            "gray": 0.0,
            "freq": 0.0,
            "psnr": 0.0,
            "ssim_metric": 0.0,
            "t_mean": 0.0,
            "A_mean": 0.0,
        }

        count = 0

        for batch in train_loader:
            hazy = batch["LQ"].to(device)
            clean = batch["GT"].to(device)

            bs = hazy.size(0)

            with torch.no_grad():
                f_h = encoder(normalize_for_encoder(hazy))
                f_c = encoder(normalize_for_encoder(clean))

            z_h, c_h = codec.encode(f_h)
            z_c, c_c = codec.encode(f_c)

            t_h, A_h = phys_t(hazy)
            c_h_guided = phys_fusion(c_h, t_h)

            img_dehaze = reconstruct_from_latent(z_h, c_h_guided, ddu, decoder)
            img_hazy = reconstruct_from_latent(z_h, c_h, ddu, decoder)
            img_clean = reconstruct_from_latent(z_c, c_c, ddu, decoder)
            img_cross = reconstruct_from_latent(z_c, c_h_guided, ddu, decoder)

            phy_recon = atmospheric_reconstruction(img_dehaze, t_h, A_h)

            loss, loss_dict = criterion(
                img_dehaze=img_dehaze,
                img_hazy_rec=img_hazy,
                img_clean_rec=img_clean,
                img_cross=img_cross,
                hazy=hazy,
                clean=clean,
                z_h=z_h,
                z_c=z_c,
                c_h=c_h,
                c_c=c_c,
                t_h=t_h,
                phy_recon=phy_recon,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, grad_clip)

            optimizer.step()

            img_eval, clean_eval = match_spatial(img_dehaze.detach(), clean)

            train_psnr = compute_psnr(img_eval.float(), clean_eval.float())
            train_ssim = compute_ssim(img_eval.float(), clean_eval.float())

            sums["loss"] += float(loss.item()) * bs
            sums["dehaze"] += float(loss_dict["dehaze"].item()) * bs
            sums["hazy"] += float(loss_dict["hazy"].item()) * bs
            sums["clean"] += float(loss_dict["clean"].item()) * bs
            sums["cross"] += float(loss_dict["cross"].item()) * bs
            sums["z"] += float(loss_dict["z"].item()) * bs
            sums["phy"] += float(loss_dict["phy"].item()) * bs
            sums["t_smooth"] += float(loss_dict["t_smooth"].item()) * bs
            sums["ssim"] += float(loss_dict["ssim"].item()) * bs
            sums["edge"] += float(loss_dict["edge"].item()) * bs
            sums["hsv"] += float(loss_dict["hsv"].item()) * bs
            sums["chroma"] += float(loss_dict["chroma"].item()) * bs
            sums["gray"] += float(loss_dict["gray"].item()) * bs
            sums["freq"] += float(loss_dict["freq"].item()) * bs
            sums["psnr"] += train_psnr * bs
            sums["ssim_metric"] += train_ssim * bs
            sums["t_mean"] += float(t_h.detach().mean().item()) * bs
            sums["A_mean"] += float(A_h.detach().mean().item()) * bs

            count += bs

        scheduler.step()

        if epoch % val_freq == 0 or epoch == 1:
            train_stats = {k: v / count for k, v in sums.items()}

            val_stats = validate(
                encoder=encoder,
                codec=codec,
                ddu=ddu,
                decoder=decoder,
                phys_t=phys_t,
                phys_fusion=phys_fusion,
                val_loader=val_loader,
                device=device,
            )

            lr = optimizer.param_groups[0]["lr"]

            print(
                f"[S1-ZAlign-Chroma][{epoch:04d}/{epochs}] "
                f"lr={lr:.2e} "
                f"loss={train_stats['loss']:.4f} "
                f"dehaze={train_stats['dehaze']:.4f} "
                f"z={train_stats['z']:.5f} "
                f"phy={train_stats['phy']:.4f} "
                f"hsv={train_stats['hsv']:.4f} "
                f"chroma={train_stats['chroma']:.4f} "
                f"gray={train_stats['gray']:.4f} "
                f"ssim_loss={train_stats['ssim']:.4f} "
                f"train={train_stats['psnr']:.3f}/{train_stats['ssim_metric']:.4f} "
                f"t={train_stats['t_mean']:.4f} A={train_stats['A_mean']:.4f} | "
                f"val_dehaze={val_stats['dehaze_psnr']:.3f}/{val_stats['dehaze_ssim']:.4f} "
                f"score={val_stats['dehaze_score']:.3f} "
                f"val_clean={val_stats['clean_psnr']:.3f}/{val_stats['clean_ssim']:.4f} "
                f"val_cross={val_stats['cross_psnr']:.3f}/{val_stats['cross_ssim']:.4f} "
                f"val_hazy={val_stats['hazy_recon_psnr']:.3f}/{val_stats['hazy_recon_ssim']:.4f} "
                f"val_z={val_stats['z_align']:.5f} "
                f"val_t={val_stats['t_mean']:.4f}"
            )

            if val_stats["dehaze_score"] > best_score:
                best_score = val_stats["dehaze_score"]
                best_psnr = val_stats["dehaze_psnr"]
                best_ssim = val_stats["dehaze_ssim"]

                save_stage1_checkpoint(
                    path=os.path.join(ckpt_dir, "stage1_best.pth"),
                    epoch=epoch,
                    codec=codec,
                    ddu=ddu,
                    decoder=decoder,
                    phys_t=phys_t,
                    phys_fusion=phys_fusion,
                    metrics=val_stats,
                    selection_metric="dehaze_score",
                    optimizer=None,
                    scheduler=None,
                    best_score=best_score,
                )

                print(
                    f"Saved best Stage1-ZAlign-Chroma checkpoint "
                    f"(score={best_score:.3f}, PSNR={best_psnr:.3f}, SSIM={best_ssim:.4f})"
                )

        if epoch % save_freq == 0 or epoch == epochs:
            last_stats = validate(
                encoder=encoder,
                codec=codec,
                ddu=ddu,
                decoder=decoder,
                phys_t=phys_t,
                phys_fusion=phys_fusion,
                val_loader=val_loader,
                device=device,
            )

            save_stage1_checkpoint(
                path=os.path.join(ckpt_dir, "stage1_last.pth"),
                epoch=epoch,
                codec=codec,
                ddu=ddu,
                decoder=decoder,
                phys_t=phys_t,
                phys_fusion=phys_fusion,
                metrics=last_stats,
                selection_metric="last",
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
            )

            print(f"Saved last checkpoint at epoch {epoch}: {os.path.join(ckpt_dir, 'stage1_last.pth')}")

    print(
        f"Best Stage1-ZAlign-Chroma: score={best_score:.3f}, "
        f"PSNR={best_psnr:.3f}, SSIM={best_ssim:.4f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")

    args = parser.parse_args()

    train_stage1(args.config)
