import cv2
import torch
import numpy as np


def normalize_for_encoder(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).round().astype(np.uint8)


def save_image(save_path: str, img_tensor: torch.Tensor):
    img = tensor_to_uint8(img_tensor)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
