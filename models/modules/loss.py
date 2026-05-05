import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_loss_fn


class FrequencyConsistencyLoss(nn.Module):
    def forward(self, pred, target):
        fp = torch.fft.fft2(pred, norm="ortho")
        ft = torch.fft.fft2(target, norm="ortho")
        return torch.abs(torch.log1p(torch.abs(fp)) - torch.log1p(torch.abs(ft))).mean()


class FrequencyPriorLoss(nn.Module):
    def forward(self, pred, target):
        _, _, h, w = pred.shape
        fp = torch.fft.fft2(pred, norm="ortho")
        ft = torch.fft.fft2(target, norm="ortho")
        y = torch.arange(h, device=pred.device).float() - h / 2
        x = torch.arange(w, device=pred.device).float() - w / 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        w_map = (xx**2 + yy**2) / ((w / 2) ** 2 + (h / 2) ** 2 + 1e-8)
        w_map = w_map.unsqueeze(0).unsqueeze(0)
        return (w_map * torch.abs(fp - ft)).mean()


class DifferentialPerceptionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, pred, target):
        diff = torch.abs(target - pred)
        pooled = self.gap(diff).squeeze(-1).squeeze(-1)
        w = torch.sigmoid(self.mlp(pooled)).unsqueeze(-1).unsqueeze(-1)
        return (w * diff).mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target):
        loss = 0.0
        for c in range(pred.shape[1]):
            p = pred[:, c:c + 1]
            t = target[:, c:c + 1]
            gx = F.conv2d(p, self.kx, padding=1) - F.conv2d(t, self.kx, padding=1)
            gy = F.conv2d(p, self.ky, padding=1) - F.conv2d(t, self.ky, padding=1)
            loss += (gx.abs() + gy.abs()).mean()
        return loss / pred.shape[1]


class AILDFreqLoss(nn.Module):
    def __init__(
        self,
        lambda_img=1.0,
        lambda_lat=0.1,
        lambda_freq=0.2,
        lambda_freq_prior=0.01,
        lambda_perc=0.0,
        lambda_ssim=0.15,
        lambda_edge=0.05,
    ):
        super().__init__()
        self.lambda_img = lambda_img
        self.lambda_lat = lambda_lat
        self.lambda_freq = lambda_freq
        self.lambda_freq_prior = lambda_freq_prior
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.freq = FrequencyConsistencyLoss()
        self.freq_prior = FrequencyPriorLoss()
        self.dp = DifferentialPerceptionLoss()
        self.edge_fn = EdgeLoss()

    def ssim_loss(self, pred, target):
        return 1.0 - ssim_loss_fn(
            pred.float(),
            target.float(),
            data_range=1.0,
            size_average=True,
        )

    def forward_stage0(self, pred, target):
        return self.l1(pred, target)

    def forward_stage1(self, pred, target):
        l1 = self.l1(pred, target)
        dp = 0.1 * self.dp(pred, target)
        ssim = self.ssim_loss(pred, target)
        return l1 + dp + self.lambda_ssim * ssim

    def forward_stage2(self, img_pred, img_gt, z_pred, z_gt):
        loss_img = self.l1(img_pred, img_gt)
        loss_dp = 0.1 * self.dp(img_pred, img_gt)
        loss_ssim = self.ssim_loss(img_pred, img_gt)
        loss_edge = self.edge_fn(img_pred, img_gt)
        loss_lat = self.l2(z_pred, z_gt)
        loss_freq = self.freq(img_pred, img_gt)
        loss_fp = self.freq_prior(img_pred, img_gt)

        return (
            self.lambda_img * (loss_img + loss_dp)
            + self.lambda_ssim * loss_ssim
            + self.lambda_edge * loss_edge
            + self.lambda_lat * loss_lat
            + self.lambda_freq * loss_freq
            + self.lambda_freq_prior * loss_fp
        )
