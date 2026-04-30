import torch.nn as nn

from .modules.convnext_encoder import ConvNextEncoder
from .modules.predehazer import SimplePreDehazer
from .modules.dcu_cee import DehazeDiffCodec
from .modules.ddu import DimensionalDecompressionUnit
from .modules.fdaa import FDAA
from .modules.hag import HAG
from .modules.nafnet import ConditionalNAFNet
from .modules.pixel_shuffle_decoder import PixelShuffleDecoder


class AILDFreqV2Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        lat_ch = opt["network_L"]["latent_channels"]
        cee_ch = opt["network_L"]["cee_channels"]
        impl_ch = opt["network_L"]["implicit_channels"]

        self.predehazer = SimplePreDehazer(ch=opt["network_pre"]["channels"])
        self.encoder = ConvNextEncoder(pretrained=opt["network_encoder"]["pretrained"])
        self.codec = DehazeDiffCodec(impl_ch, lat_ch, cee_ch)
        self.ddu = DimensionalDecompressionUnit(lat_ch, impl_ch, cee_ch)
        self.decoder = PixelShuffleDecoder(in_channels=impl_ch, upscale_factor=8)

        self.fdaa = FDAA(
            channels=opt["fdaa"]["channels"],
            num_heads=opt["fdaa"]["num_heads"],
            low_freq_ratio=opt["fdaa"]["low_freq_ratio"],
        )
        self.hag = HAG(
            t_max=opt["hag"]["t_max"],
            min_steps=opt["hag"]["min_steps"],
        )
        self.net = ConditionalNAFNet(**opt["network_G"]["setting"])
