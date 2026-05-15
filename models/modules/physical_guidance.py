import torch
import torch.nn as nn
import torch.nn.functional as F


class TransmissionEstimator(nn.Module):
    """
    Predicts:
    - transmission map t(x), range [t_min, 1]
    - atmospheric light A, range [0, 1]

    Input:
        hazy image [B, 3, H, W]

    Output:
        t: [B, 1, H, W]
        A: [B, 3, 1, 1]
    """

    def __init__(self, base_ch=32, t_min=0.05):
        super().__init__()
        self.t_min = t_min

        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1),
            nn.GroupNorm(4, base_ch),
            nn.GELU(),

            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GroupNorm(4, base_ch),
            nn.GELU(),

            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),

            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.GELU(),
        )

        self.t_head = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.a_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch * 2, base_ch, 1),
            nn.GELU(),
            nn.Conv2d(base_ch, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, hazy):
        feat = self.encoder(hazy)

        t_low = self.t_head(feat)
        t = F.interpolate(
            t_low,
            size=hazy.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # avoid zero transmission instability
        t = self.t_min + (1.0 - self.t_min) * t

        A = self.a_head(feat)

        return t, A

class PhysicalContextFusion(nn.Module):
    def __init__(self, cee_ch=64):
        super().__init__()

        self.res_scale = nn.Parameter(torch.tensor(0.05))

        self.gate = nn.Sequential(
            nn.Conv2d(1, cee_ch, 3, padding=1),
            nn.GroupNorm(8, cee_ch),
            nn.GELU(),
            nn.Conv2d(cee_ch, cee_ch, 3, padding=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(cee_ch, cee_ch, 3, padding=1),
            nn.GroupNorm(8, cee_ch),
            nn.GELU(),
            nn.Conv2d(cee_ch, cee_ch, 1),
        )

    def forward(self, c, t):
        if t.shape[-2:] != c.shape[-2:]:
            t = F.interpolate(
                t,
                size=c.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        g = self.gate(t)
        residual = self.refine(c * g)

        scale = torch.clamp(self.res_scale, 0.0, 0.2)
        return c + scale * residual

# class PhysicalContextFusion(nn.Module):
#     """
#     Injects transmission guidance into context c_h.

#     Input:
#         c: context [B, C, h, w]
#         t: transmission [B, 1, H, W]

#     Output:
#         c_guided: [B, C, h, w]
#     """

#     def __init__(self, cee_ch=64):
#         super().__init__()

#         self.gate = nn.Sequential(
#             nn.Conv2d(1, cee_ch, 3, padding=1),
#             nn.GroupNorm(8, cee_ch),
#             nn.GELU(),
#             nn.Conv2d(cee_ch, cee_ch, 3, padding=1),
#             nn.Sigmoid(),
#         )

#         self.refine = nn.Sequential(
#             nn.Conv2d(cee_ch, cee_ch, 3, padding=1),
#             nn.GroupNorm(8, cee_ch),
#             nn.GELU(),
#             nn.Conv2d(cee_ch, cee_ch, 1),
#         )

#     def forward(self, c, t):
#         if t.shape[-2:] != c.shape[-2:]:
#             t = F.interpolate(
#                 t,
#                 size=c.shape[-2:],
#                 mode="bilinear",
#                 align_corners=False,
#             )

#         g = self.gate(t)
#         residual = self.refine(c * g)

#         return c + 0.1 * residual


def atmospheric_reconstruction(J, t, A):
    """
    Atmospheric scattering model:
        I_r = J * t + A * (1 - t)

    J: dehazed/clean estimate [B,3,H,W]
    t: transmission [B,1,H,W]
    A: atmospheric light [B,3,1,1]
    """

    if t.shape[-2:] != J.shape[-2:]:
        t = F.interpolate(
            t,
            size=J.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    return J * t + A * (1.0 - t)


def transmission_smoothness_loss(t):
    dh = torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :]).mean()
    dw = torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1]).mean()
    return dh + dw


def transmission_confidence_weight(t):
    """
    Higher t = clearer region.
    Lower t = heavier haze.
    This can be used as confidence/haze guidance.
    """
    return t.clamp(0.05, 1.0)