import argparse
import time
import yaml
import torch
from thop import profile

from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.fdaa import FDAA
from models.modules.hag import HAG
from models.modules.nafnet import ConditionalNAFNet
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder
from utils.img_utils import normalize_for_encoder


class FullAILDFreq(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.encoder = ConvNextEncoder(
            pretrained=False,
            backbone=opt["network_encoder"].get("backbone", "convnext_small"),
        )

        impl_ch = opt["network_L"]["implicit_channels"]
        lat_ch = opt["network_L"]["latent_channels"]
        cee_ch = opt["network_L"]["cee_channels"]

        self.codec = DehazeDiffCodec(impl_ch, lat_ch, cee_ch)
        self.ddu = DimensionalDecompressionUnit(lat_ch, impl_ch, cee_ch)
        self.decoder = PixelShuffleDecoder(in_channels=impl_ch, upscale_factor=8)
        self.fdaa = FDAA(**opt["fdaa"])
        self.hag = HAG(
            t_max=opt["hag"]["t_max"],
            min_steps=opt["hag"]["min_steps"],
        )
        self.net = ConditionalNAFNet(**opt["network_G"]["setting"])

        self.latent_clip = float(opt["train"].get("latent_clip", 3.0))

    def forward(self, x):
        impl = self.encoder(normalize_for_encoder(x))
        z_h, c_h = self.codec.encode(impl)

        z_f, f_low, f_high, _ = self.fdaa(z_h)
        _, g = self.hag(f_low, f_high)

        t = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        z0_pred = self.net(z_f, z_f, t).clamp(-self.latent_clip, self.latent_clip)

        out = self.decoder(self.ddu(z0_pred, c_h)).clamp(0, 1)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def benchmark_runtime(model, x, warmup=10, repeat=50):
    device = x.device

    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(repeat):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    return (end - start) / repeat


def main(config):
    with open(config, "r") as f:
        opt = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullAILDFreq(opt).to(device).eval()

    x = torch.randn(1, 3, 256, 256).to(device)

    flops, params_thop = profile(model, inputs=(x,), verbose=False)

    runtime = benchmark_runtime(model, x)

    print(f"Dataset/config: {opt['name']}")
    print(f"Params: {count_params(model):.3f} M")
    print(f"THOP Params: {params_thop / 1e6:.3f} M")
    print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Runtime: {runtime * 1000:.3f} ms/image")
    print(f"FPS: {1.0 / runtime:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    args = parser.parse_args()
    main(args.config)