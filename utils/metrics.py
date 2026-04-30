import numpy as np
import torch


def compute_psnr(pred, target, max_val=1.0):
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def compute_ssim(pred, target):
    from pytorch_msssim import ssim
    return ssim(pred, target, data_range=1.0).item()


def compute_lpips(pred, target, device="cuda"):
    import lpips
    fn = lpips.LPIPS(net="vgg").to(device)
    p = pred * 2 - 1
    t = target * 2 - 1
    return fn(p, t).item()
