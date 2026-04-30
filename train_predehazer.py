import os
import yaml
import torch
from torch.utils.data import DataLoader

from data.predehaze_dataset import PreDehazeDataset
from models.modules.predehazer import SimplePreDehazer
from models.modules.loss import AILDFreqLoss
from utils.metrics import compute_psnr, compute_ssim
from utils.checkpoint_utils import save_checkpoint


def build_dataloader(opt, split):
    ds = PreDehazeDataset(opt, split)
    dl = DataLoader(
        ds,
        batch_size=opt["datasets"][split]["batch_size"],
        shuffle=opt["datasets"][split].get("use_shuffle", False),
        num_workers=opt["datasets"][split].get("n_workers", 0),
        pin_memory=True,
    )
    return ds, dl


@torch.no_grad()
def validate_predehazer(model, val_loader, loss_fn, device):
    model.eval()
    loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0

    for batch in val_loader:
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)

        pred = model(lq)
        loss = loss_fn.forward_stage0(pred, gt)

        bs = lq.size(0)
        loss_sum += float(loss.item()) * bs
        psnr_sum += compute_psnr(pred, gt) * bs
        ssim_sum += compute_ssim(pred, gt) * bs
        count += bs

    return loss_sum / count, psnr_sum / count, ssim_sum / count


def train_predehazer(config_path):
    with open(config_path, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimplePreDehazer(ch=opt["network_pre"]["channels"]).to(device)
    loss_fn = AILDFreqLoss().to(device)

    train_ds, train_dl = build_dataloader(opt, "train")
    _, val_dl = build_dataloader(opt, "val")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt["train"]["lr_pre"],
        betas=(opt["train"]["beta1"], opt["train"]["beta2"]),
    )

    ckpt_dir = os.path.join("checkpoints", opt["name"])
    os.makedirs(ckpt_dir, exist_ok=True)

    best_psnr = -1.0
    epochs = opt["train"]["pre_epochs"]

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, psnr_sum, ssim_sum, count = 0.0, 0.0, 0.0, 0

        for batch in train_dl:
            lq = batch["LQ"].to(device)
            gt = batch["GT"].to(device)

            pred = model(lq)
            loss = loss_fn.forward_stage0(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = lq.size(0)
            loss_sum += float(loss.item()) * bs
            psnr_sum += compute_psnr(pred.detach(), gt) * bs
            ssim_sum += compute_ssim(pred.detach(), gt) * bs
            count += bs

        train_loss = loss_sum / count
        train_psnr = psnr_sum / count
        train_ssim = ssim_sum / count

        if epoch % opt["train"]["val_freq"] == 0 or epoch == 1:
            val_loss, val_psnr, val_ssim = validate_predehazer(model, val_dl, loss_fn, device)
            print(
                f"[Pre][{epoch:04d}/{epochs}] "
                f"train_loss={train_loss:.4f} train_psnr={train_psnr:.3f} train_ssim={train_ssim:.4f} | "
                f"val_loss={val_loss:.4f} val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f}"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    os.path.join(ckpt_dir, "predehazer_best.pth"),
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "psnr": val_psnr,
                    },
                )

    print(f"Best predehazer PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    args = parser.parse_args()
    train_predehazer(args.config)
