import os
import torch


def save_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device, weights_only=False)
