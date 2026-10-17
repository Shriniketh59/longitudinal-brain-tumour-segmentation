"""Grad-CAM for the VGG16 classifier: where in the slice the predicted class was found."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Output of the last conv block's ReLU (features[29]), 512 x 14 x 14 for a 224 input.
TARGET_LAYER = 29

# Anchor colours for the heatmap (dark blue, cyan, yellow, red), interpolated over 0..1.
_ANCHORS = np.array([[0.0, 20, 30, 120], [0.35, 0, 200, 255], [0.65, 255, 220, 40], [1.0, 230, 30, 30]], dtype=np.float32)


def _colormap(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(values, _ANCHORS[:, 0], _ANCHORS[:, c + 1])
    return out


def compute_cam(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Normalised 0..1 heatmap at the conv feature resolution."""
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.features[TARGET_LAYER].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach()[0], store["grad"].detach()[0]
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Heatmap blended over the original slice, at the original resolution."""
    h, w = gray.shape
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    heat = np.asarray(cam_img, dtype=np.float32) / 255.0
    colors = _colormap(heat)
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    # Below 20% of the peak is fully transparent; opacity ramps up to alpha by 80%.
    a = (alpha * np.clip((heat - 0.2) / 0.6, 0, 1))[..., None]
    return Image.fromarray((base * (1 - a) + colors * a).clip(0, 255).astype(np.uint8), mode="RGB")


def peak_location(cam: np.ndarray) -> tuple[float, float]:
    """Peak position as fractions of width and height (0..1)."""
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return (x + 0.5) / cam.shape[1], (y + 0.5) / cam.shape[0]


def focus_in_brain(gray: np.ndarray, cam: np.ndarray, threshold: float = 0.5) -> float:
    """Share of strong Grad-CAM activation (>= threshold of peak) that lies on head tissue.

    Head tissue is approximated as pixels brighter than the dark background. A low value means
    the prediction leaned on image borders or background rather than anatomy.
    """
    h, w = gray.shape
    heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    tissue = gray > max(15, np.percentile(gray, 30) * 0.5)
    strong = heat >= threshold
    total = float(heat[strong].sum())
    return round(float(heat[strong & tissue].sum()) / total, 3) if total > 0 else 0.0
