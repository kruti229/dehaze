# models/aild_freq_model.py
"""
AILD-Freq extends DehazeDiff by:
  - Running FDAA on latent z0 before DCU
  - Using HAG to control effective diffusion steps
  - Adding L_freq to the loss
"""
from models.dehazing_base_model import DehazingBaseModel
from models.modules.fdaa import FDAA
from models.modules.hag import HAG

class AILDFreqModel(DehazingBaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        latent_channels = opt['network_L']['latent_channels']  # e.g. 16

        self.fdaa = FDAA(channels=latent_channels).to(self.device)
        self.hag  = HAG(t_max=100, min_steps=20).to(self.device)

        # Add FDAA + HAG params to optimizer
        self._add_params_to_optimizer(
            list(self.fdaa.parameters()) + list(self.hag.parameters()))

    def encode_with_fdaa(self, img):
        """Encode image, then apply FDAA to latent"""
        latent, cee = self.latent_model.encode(img)   # z0, CEE
        z_fdaa, f_low, f_high = self.fdaa(latent)     # enriched latent
        return z_fdaa, cee, f_low, f_high

    def optimize_parameters(self, step, timesteps, sde):
        # Encode LQ with FDAA
        z_lq, cee_lq, f_low, f_high = self.encode_with_fdaa(self.lq)

        # HAG: get adaptive step count
        t_eff, g = self.hag(f_low, f_high)            # per-image step count

        # Encode GT (for loss target)
        z_gt, _ = self.latent_model.encode(self.gt)

        # Diffusion training step (pass t_eff to SDE)
        noisy_state = sde.noise_state(z_lq)
        self.generator.train()
        pred_noise = self.generator(noisy_state, z_lq, timesteps)

        # Loss computation
        loss_latent = self.matching_loss(pred_noise,
                                          sde.get_score_from_noise(
                                              pred_noise, timesteps))
        loss_freq   = self.freq_loss(self.dehazed_output, self.gt)
        loss_dp     = self.dp_loss(self.dehazed_output, self.gt)

        total_loss = (self.lambda1 * loss_latent +
                      self.lambda2 * loss_freq   +
                      self.lambda3 * loss_dp)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()