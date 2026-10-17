"""Preprocessing, prediction and overlay rendering for 2D MRI slices."""

import base64
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app import gradcam
from app.malignancy import assess
from app.preprocess import crop_to_head
from app.models.unet import SEG_CLASSES
from app.models.vgg16_classifier import CLASS_NAMES

CLS_SIZE = 224
SEG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# RGB per non-background subregion, matched to the frontend legend.
OVERLAY_COLORS = {
    1: (76, 155, 232),   # necrotic core: blue
    2: (240, 160, 48),   # edema: amber
    3: (229, 107, 176),  # enhancing: pink
}
OVERLAY_ALPHA = 0.45


class PreparedSlice:
    """Holds a decoded slice as grayscale uint8 plus its original size."""

    def __init__(self, gray: np.ndarray):
        self.gray = gray
        self.size = (gray.shape[1], gray.shape[0])  # (W, H) for PIL


def decode_slice(data: bytes) -> PreparedSlice:
    img = Image.open(io.BytesIO(data)).convert("L")
    return PreparedSlice(np.asarray(img, dtype=np.uint8))


def _to_tensor(gray: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    resized = Image.fromarray(gray).resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    rgb = np.stack([arr, arr, arr], axis=-1)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def classify(model: nn.Module, scan: PreparedSlice, device: torch.device, explain: bool = True, crop: bool = False) -> dict:
    # Models trained on head-cropped slices must see the same crop here.
    gray = crop_to_head(scan.gray) if crop else scan.gray
    x = _to_tensor(gray, CLS_SIZE, device)
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().tolist()
    best = int(np.argmax(probs))
    probabilities = {name: round(p, 6) for name, p in zip(CLASS_NAMES, probs)}
    result = {
        "label": CLASS_NAMES[best],
        "probabilities": probabilities,
        "malignancy": assess(probabilities),
        "gradcam": None,
        "preprocess": "crop_head" if crop else "none",
    }
    if explain:
        cam = gradcam.compute_cam(model, x, best)
        px, py = gradcam.peak_location(cam)
        result["gradcam"] = {
            "overlay_png_base64": _encode_png(gradcam.overlay(gray, cam)),
            "peak": {"x": round(px, 3), "y": round(py, 3)},
            "focus_in_brain": gradcam.focus_in_brain(gray, cam),
            "target_class": CLASS_NAMES[best],
            "layer": "features.29 (last conv block)",
        }
    return result


@torch.inference_mode()
def segment(model: nn.Module, scan: PreparedSlice, device: torch.device) -> dict:
    logits = model(_to_tensor(scan.gray, SEG_SIZE, device).contiguous(memory_format=torch.channels_last))
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Back to original resolution with nearest neighbour so labels stay discrete.
    mask = np.asarray(Image.fromarray(pred).resize(scan.size, Image.NEAREST), dtype=np.uint8)

    counts = {name: int((mask == idx).sum()) for idx, name in enumerate(SEG_CLASSES)}
    return {
        "overlay_png_base64": _encode_png(_overlay(scan.gray, mask)),
        "mask_png_base64": _encode_png(Image.fromarray(mask * 85)),  # scale labels 0-3 to visible grey
        "pixel_counts": counts,
        "image_size": [scan.size[0], scan.size[1]],
    }


def _overlay(gray: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    for label, color in OVERLAY_COLORS.items():
        sel = mask == label
        base[sel] = base[sel] * (1 - OVERLAY_ALPHA) + np.array(color, dtype=np.float32) * OVERLAY_ALPHA
    return Image.fromarray(base.clip(0, 255).astype(np.uint8), mode="RGB")


def _encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
