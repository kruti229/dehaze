import numpy as np
import torch


def match_size(pred, target):
    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])
    return pred[..., :h, :w], target[..., :h, :w]


def compute_psnr(pred, target, max_val=1.0):
    pred, target = match_size(pred, target)

    pred = pred.float().clamp(0.0, max_val)
    target = target.float().clamp(0.0, max_val)

    mse = ((pred - target) ** 2).mean().item()

    if mse <= 1.0e-12:
        return float("inf")

    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def compute_ssim(pred, target):
    from pytorch_msssim import ssim

    pred, target = match_size(pred, target)

    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)

    return ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True,
        win_size=7,
    ).item()


class LPIPSMetric:
    def __init__(self, device="cuda"):
        import lpips

        self.device = device
        self.fn = lpips.LPIPS(net="vgg").to(device)
        self.fn.eval()

    @torch.no_grad()
    def __call__(self, pred, target):
        pred, target = match_size(pred, target)

        pred = pred.float().clamp(0.0, 1.0).to(self.device)
        target = target.float().clamp(0.0, 1.0).to(self.device)

        p = pred * 2.0 - 1.0
        t = target * 2.0 - 1.0

        return self.fn(p, t).item()