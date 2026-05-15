import torch
import torch.nn as nn
import torch.nn.functional as F


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

        w_map = (xx ** 2 + yy ** 2) / ((w / 2) ** 2 + (h / 2) ** 2 + 1e-8)
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


class FocalL1Loss(nn.Module):
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        weight = diff.detach().pow(self.gamma)
        weight = weight / (weight.mean() + 1e-6)
        return (weight * diff).mean()


class ColorConsistencyLoss(nn.Module):
    def forward(self, pred, target):
        pred_mean = pred.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        pred_std = pred.std(dim=[2, 3], unbiased=False)
        target_std = target.std(dim=[2, 3], unbiased=False)

        mean_loss = F.l1_loss(pred_mean, target_mean)
        std_loss = F.l1_loss(pred_std, target_std)

        return mean_loss + 0.5 * std_loss


class TVLoss(nn.Module):
    def forward(self, x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        return dx + dy


class AtmosphericConsistencyLoss(nn.Module):
    def forward(self, dehazed, hazy):
        # Simple image-formation consistency:
        # I = J * t + A * (1 - t)
        # A estimated from brightest pixels.
        with torch.no_grad():
            A = hazy.amax(dim=[2, 3], keepdim=True)
            diff = torch.abs(hazy - dehazed).mean(dim=1, keepdim=True)
            t = torch.exp(-diff).clamp(0.05, 1.0)

        recon_hazy = dehazed * t + A * (1.0 - t)
        return F.l1_loss(recon_hazy, hazy)


class AILDFreqLoss(nn.Module):
    def __init__(
        self,
        lambda_img=1.0,
        lambda_lat=0.1,
        lambda_freq=0.2,
        lambda_freq_prior=0.01,
        lambda_perc=0.0,
        lambda_ssim=0.1,
        lambda_color=0.05,
        lambda_focal=0.2,
        lambda_tv=0.005,
        lambda_atm=0.05,
    ):
        super().__init__()

        self.lambda_img = lambda_img
        self.lambda_lat = lambda_lat
        self.lambda_freq = lambda_freq
        self.lambda_freq_prior = lambda_freq_prior
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_focal = lambda_focal
        self.lambda_tv = lambda_tv
        self.lambda_atm = lambda_atm

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        self.freq = FrequencyConsistencyLoss()
        self.freq_prior = FrequencyPriorLoss()
        self.dp = DifferentialPerceptionLoss()
        self.focal_l1 = FocalL1Loss(gamma=1.5)
        self.color = ColorConsistencyLoss()
        self.tv = TVLoss()
        self.atm = AtmosphericConsistencyLoss()

    def ssim_loss(self, pred, target):
        from pytorch_msssim import ssim
        return 1.0 - ssim(
            pred.float(),
            target.float(),
            data_range=1.0,
            size_average=True,
        )

    def forward_stage0(self, pred, target):
        return self.l1(pred, target)

    def forward_stage1(self, pred, target):
        l1 = self.l1(pred, target)
        dp = self.dp(pred, target)
        color = self.color(pred, target)
        tv = self.tv(pred)

        return (
            l1
            + 0.1 * dp
            + self.lambda_color * color
            + self.lambda_tv * tv
        )

    def forward_stage2(self, img_pred, img_gt, z_pred, z_gt, hazy=None):
        loss_img = self.l1(img_pred, img_gt)
        loss_focal = self.focal_l1(img_pred, img_gt)
        loss_dp = self.dp(img_pred, img_gt)
        loss_lat = self.l2(z_pred, z_gt)
        loss_freq = self.freq(img_pred, img_gt)
        loss_freq_prior = self.freq_prior(img_pred, img_gt)
        loss_ssim = self.ssim_loss(img_pred, img_gt)
        loss_color = self.color(img_pred, img_gt)
        loss_tv = self.tv(img_pred)

        total = (
            self.lambda_img * (loss_img + 0.1 * loss_dp)
            + self.lambda_focal * loss_focal
            + self.lambda_lat * loss_lat
            + self.lambda_freq * loss_freq
            + self.lambda_freq_prior * loss_freq_prior
            + self.lambda_ssim * loss_ssim
            + self.lambda_color * loss_color
            + self.lambda_tv * loss_tv
        )

        if hazy is not None and self.lambda_atm > 0:
            total = total + self.lambda_atm * self.atm(img_pred, hazy)

        return total