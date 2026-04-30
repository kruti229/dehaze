import torch
import torch.nn as nn


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


class AILDFreqLoss(nn.Module):
    def __init__(
        self,
        lambda_img=1.0,
        lambda_lat=0.1,
        lambda_freq=0.2,
        lambda_freq_prior=0.01,
        lambda_perc=0.0,
        lambda_ssim=0.1,
    ):
        super().__init__()
        self.lambda_img = lambda_img
        self.lambda_lat = lambda_lat
        self.lambda_freq = lambda_freq
        self.lambda_freq_prior = lambda_freq_prior
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.freq = FrequencyConsistencyLoss()
        self.freq_prior = FrequencyPriorLoss()
        self.dp = DifferentialPerceptionLoss()

    def forward_stage0(self, pred, target):
        return self.l1(pred, target)

    def forward_stage1(self, pred, target):
        return self.l1(pred, target) + 0.1 * self.dp(pred, target)

    def ssim_loss(self, pred, target):
        from pytorch_msssim import ssim

        return 1.0 - ssim(pred, target, data_range=1.0, size_average=True)

    def forward_stage2(self, img_pred, img_gt, z_pred, z_gt):
        loss_img = self.l1(img_pred, img_gt) + 0.1 * self.dp(img_pred, img_gt)
        loss_lat = self.l2(z_pred, z_gt)
        loss_freq = self.freq(img_pred, img_gt)
        loss_freq_prior = self.freq_prior(img_pred, img_gt)
        loss_ssim = self.ssim_loss(img_pred, img_gt)
        return (
            self.lambda_img * loss_img
            + self.lambda_lat * loss_lat
            + self.lambda_freq * loss_freq
            + self.lambda_freq_prior * loss_freq_prior
            + self.lambda_ssim * loss_ssim
        )
