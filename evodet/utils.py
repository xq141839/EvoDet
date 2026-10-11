"""Shared helpers for EvoDet inference (image preprocessing, class names)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = [
    "ENDO_CLASS_NAMES",
    "ENDO_COLORS",
    "load_image_tensor",
    "box_xyxy_to_cxcywh",
    "box_cxcywh_to_xyxy",
]

# Endoscapes category order (category_id 0..5), matching the Endoscapes
# benchmark and the multi-class DEIM/DEIMv2 configuration.
ENDO_CLASS_NAMES = [
    "Cystic Plate",
    "Calot Triangle",
    "Cystic Artery",
    "Cystic Duct",
    "Gallbladder",
    "Tool",
]

# BGR colours for visualization (OpenCV).
ENDO_COLORS = {
    "Cystic Plate": (0, 0, 255),
    "Calot Triangle": (0, 255, 0),
    "Cystic Artery": (255, 0, 0),
    "Cystic Duct": (0, 255, 255),
    "Gallbladder": (255, 0, 255),
    "Tool": (255, 255, 0),
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_tensor(
    path: str, size: Tuple[int, int] = (640, 640), device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Load and preprocess an image for the ViT/DINOv3 backbone.

    Args:
        path: image path.
        size: ``(h, w)`` resize target.
        device: target device.

    Returns:
        tensor: ``[1, 3, h, w]`` normalised tensor.
        orig_size: ``[1, 2]`` original ``(w, h)``.
        orig_wh: ``(w, h)`` as a plain tuple.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    orig_size = torch.tensor([[w, h]], dtype=torch.float32, device=device)
    return tensor, orig_size, (w, h)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
