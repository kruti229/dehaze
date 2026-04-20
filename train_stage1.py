import os
import yaml
import torch
from torch.utils.data import DataLoader

from data.lqgt_dataset import LQGTDataset
from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.loss import AILDFreqLoss
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder


def train_stage1(config_path: str):
    opt = yaml.safe_load(open(config_path, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Models
    # -----------------------------
    encoder = ConvNextEncoder(pretrained=True).to(device)
    codec = DehazeDiffCodec(192, 16, 64).to(device)
    ddu = DimensionalDecompressionUnit(16, 192, 64).to(device)
    decoder = PixelShuffleDecoder(in_channels=192, upscale_factor=8).to(device)

    # Freeze encoder (feature extractor only)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Trainable params in stage1
    params = list(codec.parameters()) + list(ddu.parameters()) + list(decoder.parameters())

    from lion_pytorch import Lion
    optim = Lion(
        params,
        lr=opt["train"]["lr_G"],
        betas=(opt["train"]["beta1"], opt["train"]["beta2"]),
    )

    # Stage1 objective: DehazeDiff-style reconstruction (L1 + DP)
    loss_fn = AILDFreqLoss(
        l1=opt["loss"]["lambda1"],
        l2=opt["loss"]["lambda2"],
        l3=opt["loss"]["lambda3"],
    ).to(device)

    # -----------------------------
    # Data
    # -----------------------------
    train_ds = LQGTDataset(opt, "train")
    train_dl = DataLoader(
        train_ds,
        batch_size=opt["datasets"]["train"]["batch_size"],
        shuffle=True,
        num_workers=opt["datasets"]["train"].get("n_workers", 4),
        pin_memory=True,
    )

    epochs = opt["train"]["stage1_epochs"]
    save_dir = f"checkpoints/{opt['name']}"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    print(f"Stage 1 training: {epochs} epochs, {len(train_ds)} samples")

    for epoch in range(1, epochs + 1):
        codec.train()
        ddu.train()
        decoder.train()

        total_loss = 0.0
        total_a = 0.0
        total_b = 0.0
        total_ctr = 0.0
        n_samples = 0

        for batch in train_dl:
            lq = batch["LQ"].to(device)  # hazy image
            gt = batch["GT"].to(device)  # clean image
            bs = lq.size(0)

            # -----------------------------
            # Phase A: hazy reconstruction branch
            # -----------------------------
            with torch.no_grad():
                impl_hazy = encoder(lq)
            z_hazy, cee_hazy = codec.encode(impl_hazy)
            impl_rec = ddu(z_hazy, cee_hazy)
            pred_a = decoder(impl_rec)
            loss_a = loss_fn(pred_a, lq, stage=1)

            # -----------------------------
            # Phase B: cross-branch alignment branch
            # -----------------------------
            with torch.no_grad():
                impl_gt = encoder(gt)
            z_gt, cee_gt = codec.encode(impl_gt)

            # Decode GT latent with hazy CEE (forces transferable latent/context structure)
            impl_cross = ddu(z_gt, cee_hazy)
            pred_b = decoder(impl_cross)
            loss_b = loss_fn(pred_b, gt, stage=1)

            # -----------------------------
            # Contrastive regularization on CEE
            # -----------------------------
            cos_sim = torch.nn.functional.cosine_similarity(
                cee_gt.flatten(1), cee_hazy.flatten(1), dim=1
            ).mean()
            # push apart => minimize similarity
            loss_ctr = cos_sim

            loss = loss_a + loss_b + 0.1 * loss_ctr

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * bs
            total_a += loss_a.item() * bs
            total_b += loss_b.item() * bs
            total_ctr += loss_ctr.item() * bs
            n_samples += bs

        avg_loss = total_loss / max(1, n_samples)
        avg_a = total_a / max(1, n_samples)
        avg_b = total_b / max(1, n_samples)
        avg_ctr = total_ctr / max(1, n_samples)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:6d}/{epochs} | "
                f"loss={avg_loss:.4f} "
                f"(A={avg_a:.4f}, B={avg_b:.4f}, Ctr={avg_ctr:.4f})"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "codec": codec.state_dict(),
                    "ddu": ddu.state_dict(),
                    "decoder": decoder.state_dict(),
                    "best_loss": best_loss,
                    "config_name": opt["name"],
                },
                f"{save_dir}/stage1_best.pth",
            )

    print(f"Stage 1 complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved: {save_dir}/stage1_best.pth")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nh_haze.yml")
    args = ap.parse_args()
    train_stage1(args.config)