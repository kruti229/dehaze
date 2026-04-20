import torch, yaml, os, cv2, numpy as np
from utils.metrics import compute_psnr, compute_ssim
 
def test(config_path):
    opt    = yaml.safe_load(open(config_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = f"checkpoints/{opt['name']}/stage2_best.pth"
 
    # Load models (same as train_stage2 setup)
    # ... [load encoder, codec, fdaa, hag, net, ddu] ...
    # ... [load state dicts from stage2_best.pth] ...
 
    results_dir = f"results/{opt['name']}"
    os.makedirs(results_dir, exist_ok=True)
 
    from data.lqgt_dataset import LQGTDataset
    from torch.utils.data import DataLoader
    test_ds = LQGTDataset(opt, "test")
    test_dl = DataLoader(test_ds, batch_size=1)
 
    all_psnr, all_ssim = [], []
 
    for i, batch in enumerate(test_dl):
        lq = batch["LQ"].to(device)
        gt = batch["GT"].to(device)
 
        # Forward pass (same as validate())
        # ... run full pipeline ...
        # dehazed = decoder(ddu(z_clean, cee))
 
        psnr = compute_psnr(dehazed, gt)
        ssim = compute_ssim(dehazed, gt)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
 
        # Save output image
        img = (dehazed[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"{results_dir}/dehazed_{i:03d}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
 
    print(f"\nResults on {opt['name']}:")
    print(f"  PSNR : {np.mean(all_psnr):.2f} dB  (DehazeDiff: 21.06)")
    print(f"  SSIM : {np.mean(all_ssim):.4f}     (DehazeDiff: 0.7118)")
    print(f"  Saved to: {results_dir}/")
