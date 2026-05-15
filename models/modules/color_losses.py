import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_hsv_torch(rgb):
    """
    rgb: [B, 3, H, W], range [0, 1]
    returns hsv: [B, 3, H, W], H/S/V in [0, 1]
    """
    rgb = rgb.float().clamp(0.0, 1.0)

    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]

    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)

    delta = maxc - minc

    v = maxc
    s = delta / (maxc + 1.0e-6)

    h = torch.zeros_like(maxc)
    mask = delta > 1.0e-6

    r_is_max = (maxc == r) & mask
    g_is_max = (maxc == g) & mask
    b_is_max = (maxc == b) & mask

    h_r = ((g - b) / (delta + 1.0e-6)) % 6.0
    h_g = ((b - r) / (delta + 1.0e-6)) + 2.0
    h_b = ((r - g) / (delta + 1.0e-6)) + 4.0

    h = torch.where(r_is_max, h_r, h)
    h = torch.where(g_is_max, h_g, h)
    h = torch.where(b_is_max, h_b, h)

    h = (h / 6.0) % 1.0

    return torch.cat([h, s, v], dim=1)


def rgb_to_ycbcr_torch(rgb):
    """
    rgb: [B, 3, H, W], range [0, 1]
    returns ycbcr: [B, 3, H, W], all channels approximately [0, 1]
    """
    rgb = rgb.float().clamp(0.0, 1.0)

    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    return torch.cat([y, cb, cr], dim=1)


class RGBHSVColorLoss(nn.Module):
    """
    Strong color preservation:
    1. RGB mean consistency
    2. RGB std consistency
    3. RGB channel ratio consistency
    4. HSV hue circular consistency
    5. HSV saturation/value consistency
    """

    def __init__(
        self,
        w_rgb_mean=1.0,
        w_rgb_std=0.5,
        w_ratio=0.3,
        w_hue=0.5,
        w_sv=0.5,
    ):
        super().__init__()
        self.w_rgb_mean = w_rgb_mean
        self.w_rgb_std = w_rgb_std
        self.w_ratio = w_ratio
        self.w_hue = w_hue
        self.w_sv = w_sv

    def forward(self, pred, target):
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)

        rgb_mean = F.l1_loss(
            pred.mean(dim=(2, 3)),
            target.mean(dim=(2, 3)),
        )

        rgb_std = F.l1_loss(
            pred.std(dim=(2, 3), unbiased=False),
            target.std(dim=(2, 3), unbiased=False),
        )

        pred_ratio = pred / (pred.mean(dim=1, keepdim=True) + 1.0e-6)
        target_ratio = target / (target.mean(dim=1, keepdim=True) + 1.0e-6)

        ratio = F.l1_loss(pred_ratio, target_ratio)

        pred_hsv = rgb_to_hsv_torch(pred)
        target_hsv = rgb_to_hsv_torch(target)

        pred_h = pred_hsv[:, 0:1]
        target_h = target_hsv[:, 0:1]

        hue_diff = torch.abs(pred_h - target_h)
        hue_diff = torch.minimum(hue_diff, 1.0 - hue_diff)
        hue = hue_diff.mean()

        sv = F.l1_loss(pred_hsv[:, 1:3], target_hsv[:, 1:3])

        return (
            self.w_rgb_mean * rgb_mean
            + self.w_rgb_std * rgb_std
            + self.w_ratio * ratio
            + self.w_hue * hue
            + self.w_sv * sv
        )


class ChromaConsistencyLoss(nn.Module):
    """
    Direct YCbCr chroma loss.
    This directly attacks desaturation / gray-looking output.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_ycc = rgb_to_ycbcr_torch(pred)
        target_ycc = rgb_to_ycbcr_torch(target)

        pred_cbcr = pred_ycc[:, 1:3]
        target_cbcr = target_ycc[:, 1:3]

        return F.l1_loss(pred_cbcr, target_cbcr)


class GrayPenaltyLoss(nn.Module):
    """
    Prevents the model from choosing an easy gray/desaturated solution.
    It matches the amount of channel separation in prediction and target.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)

        pr = pred[:, 0:1]
        pg = pred[:, 1:2]
        pb = pred[:, 2:3]

        tr = target[:, 0:1]
        tg = target[:, 1:2]
        tb = target[:, 2:3]

        pred_chroma = (pr - pg).abs() + (pg - pb).abs() + (pb - pr).abs()
        target_chroma = (tr - tg).abs() + (tg - tb).abs() + (tb - tr).abs()

        return F.l1_loss(pred_chroma, target_chroma)