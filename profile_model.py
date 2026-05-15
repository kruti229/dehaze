import os
import time
import argparse

import torch
import yaml

from models.modules.convnext_encoder import ConvNextEncoder
from models.modules.dcu_cee import DehazeDiffCodec
from models.modules.ddu import DimensionalDecompressionUnit
from models.modules.pixel_shuffle_decoder import PixelShuffleDecoder
from models.modules.physical_guidance import TransmissionEstimator, PhysicalContextFusion
from utils.img_utils import normalize_for_encoder
from utils.checkpoint_utils import load_checkpoint


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def build_model(opt, device):
    encoder = ConvNextEncoder(
        pretrained=bool(opt["network_encoder"].get("pretrained", True)),
        backbone=opt["network_encoder"].get("backbone", "convnext_small"),
    ).to(device)

    codec = DehazeDiffCodec(
        impl_ch=int(opt["network_L"]["implicit_channels"]),
        lat_ch=int(opt["network_L"]["latent_channels"]),
        cee_ch=int(opt["network_L"]["cee_channels"]),
    ).to(device)

    ddu = DimensionalDecompressionUnit(
        in_ch=int(opt["network_L"]["latent_channels"]),
        out_ch=int(opt["network_L"]["implicit_channels"]),
        cee_ch=int(opt["network_L"]["cee_channels"]),
    ).to(device)

    decoder = PixelShuffleDecoder(
        in_channels=int(opt["network_L"]["implicit_channels"]),
        upscale_factor=8,
    ).to(device)

    phys_t = TransmissionEstimator(
        base_ch=int(opt.get("physical", {}).get("base_ch", 32)),
        t_min=float(opt.get("physical", {}).get("t_min", 0.05)),
    ).to(device)

    phys_fusion = PhysicalContextFusion(
        cee_ch=int(opt["network_L"]["cee_channels"]),
    ).to(device)

    return encoder, codec, ddu, decoder, phys_t, phys_fusion


@torch.no_grad()
def forward_model(x, encoder, codec, ddu, decoder, phys_t, phys_fusion):
    f = encoder(normalize_for_encoder(x))
    z, c = codec.encode(f)
    t, A = phys_t(x)
    c = phys_fusion(c, t)
    out = decoder(ddu(z, c)).clamp(0, 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nh_haze.yml")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        opt = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder, codec, ddu, decoder, phys_t, phys_fusion = build_model(opt, device)

    ckpt_path = os.path.join("checkpoints", opt["name"], "stage1_best.pth")
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, device)
        codec.load_state_dict(ckpt["codec"])
        ddu.load_state_dict(ckpt["ddu"])
        decoder.load_state_dict(ckpt["decoder"])

        if "phys_t" in ckpt:
            phys_t.load_state_dict(ckpt["phys_t"])
        if "phys_fusion" in ckpt:
            phys_fusion.load_state_dict(ckpt["phys_fusion"])

    for m in [encoder, codec, ddu, decoder, phys_t, phys_fusion]:
        m.eval()

    total_params = (
        count_params(encoder)
        + count_params(codec)
        + count_params(ddu)
        + count_params(decoder)
        + count_params(phys_t)
        + count_params(phys_fusion)
    )

    trainable_params = (
        sum(p.numel() for m in [codec, ddu, decoder, phys_t, phys_fusion] for p in m.parameters())
    )

    x = torch.randn(1, 3, args.size, args.size).to(device)

    # warmup
    for _ in range(10):
        _ = forward_model(x, encoder, codec, ddu, decoder, phys_t, phys_fusion)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    for _ in range(args.runs):
        _ = forward_model(x, encoder, codec, ddu, decoder, phys_t, phys_fusion)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    runtime_ms = elapsed / args.runs * 1000.0
    fps = 1000.0 / runtime_ms

    flops_g = None

    try:
        from thop import profile

        class WrappedModel(torch.nn.Module):
            def __init__(self, encoder, codec, ddu, decoder, phys_t, phys_fusion):
                super().__init__()
                self.encoder = encoder
                self.codec = codec
                self.ddu = ddu
                self.decoder = decoder
                self.phys_t = phys_t
                self.phys_fusion = phys_fusion

            def forward(self, x):
                return forward_model(
                    x,
                    self.encoder,
                    self.codec,
                    self.ddu,
                    self.decoder,
                    self.phys_t,
                    self.phys_fusion,
                )

        wrapped = WrappedModel(encoder, codec, ddu, decoder, phys_t, phys_fusion).to(device)
        flops, params = profile(wrapped, inputs=(x,), verbose=False)
        flops_g = flops / 1e9

    except Exception as e:
        print(f"[FLOPs] thop failed or not installed: {e}")

    print("")
    print("Efficiency Profile")
    print("------------------")
    print(f"Dataset/config      : {opt['name']}")
    print(f"Input size          : 1x3x{args.size}x{args.size}")
    print(f"Total params        : {total_params / 1e6:.3f} M")
    print(f"Trainable params    : {trainable_params / 1e6:.3f} M")
    if flops_g is not None:
        print(f"FLOPs               : {flops_g:.3f} G")
    else:
        print("FLOPs               : unavailable")
    print(f"Runtime             : {runtime_ms:.3f} ms/image")
    print(f"FPS                 : {fps:.3f}")


if __name__ == "__main__":
    main()