# pip install ptflops
from ptflops import get_model_complexity_info
import torch
 
# Measure FDAA FLOPs
from models.modules.fdaa import FDAA
model = FDAA(channels=16, num_heads=8)
macs, params = get_model_complexity_info(
    model, (16, 32, 32),  # z0 at 1024x1024 → 128x128 latent
    as_strings=True, print_per_layer_stat=False)
print(f"FDAA  — Params: {params}  MACs: {macs}")
 
# Measure full model FLOPs
# Compare against DehazeDiff: 106.56M params, 36.86G FLOPs, 4.273s runtime
# Target: params <= 115M, FLOPs <= 40G, runtime <= 5s
