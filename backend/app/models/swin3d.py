"""3D SwinUNETR (Swin Transformer U-Net) for whole-volume BraTS segmentation.

Configuration and preprocessing match notebooks/train_swin_unetr_3d.ipynb:
two input sequences (T1 post-contrast, FLAIR), z-score per sequence over nonzero voxels,
three sigmoid outputs (tumor core, whole tumor, enhancing tumor).
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODALITIES = ("t1c", "t2f")
DEFAULT_ROI = (96, 96, 96)
FAST_SCALE = 0.75


def build_swin_unetr() -> nn.Module:
    from monai.networks.nets import SwinUNETR  # imported lazily: MONAI is only needed for the 3D model

    return SwinUNETR(in_channels=len(MODALITIES), out_channels=3, feature_size=48, use_checkpoint=False)


def load_swin_unetr(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    model = build_swin_unetr()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    meta_path = path.with_suffix(".metrics.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model.to(device).eval(), meta


def normalise(channels: np.ndarray) -> np.ndarray:
    """(C, X, Y, Z) float32, z-scored per channel over nonzero voxels; zeros stay zero."""
    out = np.zeros_like(channels, dtype=np.float32)
    for c in range(channels.shape[0]):
        ch = channels[c]
        nz = ch != 0
        if nz.any():
            mean, std = ch[nz].mean(), ch[nz].std()
            out[c][nz] = (ch[nz] - mean) / (std if std > 0 else 1.0)
    return out


@torch.inference_mode()
def predict_labels(
    model: nn.Module,
    channels: np.ndarray,
    device: torch.device,
    roi: tuple[int, int, int] = DEFAULT_ROI,
    overlap: float = 0.25,
    scale: float = 1.0,
) -> np.ndarray:
    """Label volume (X, Y, Z) uint8 in BraTS convention: 1 necrotic core, 2 edema, 3 enhancing."""
    from monai.inferers import sliding_window_inference

    x = normalise(channels.astype(np.float32))
    # Run only on the head's bounding box; the rest is background.
    nz = np.argwhere(np.any(x != 0, axis=0))
    if nz.size == 0:
        return np.zeros(channels.shape[1:], dtype=np.uint8)
    lo, hi = nz.min(0), nz.max(0) + 1
    crop = x[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    x_in = torch.from_numpy(crop)[None].to(device)
    if scale != 1.0:
        # Fast mode: infer on a downscaled volume and resize the logits back.
        # At 0.75 this is ~3x faster on CPU (72 s -> 24 s); WT Dice moved -0.04 / +0.07 on two cases.
        x_in = torch.nn.functional.interpolate(x_in, scale_factor=scale, mode="trilinear", align_corners=False)
    logits = sliding_window_inference(x_in, roi, sw_batch_size=1, predictor=model, overlap=overlap)
    if scale != 1.0:
        logits = torch.nn.functional.interpolate(logits, size=crop.shape[1:], mode="trilinear", align_corners=False)
    tc, wt, et = (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()

    region = np.zeros(crop.shape[1:], dtype=np.uint8)
    region[wt] = 2           # whole tumor not in core = edema
    region[tc] = 1           # core not enhancing = necrotic core
    region[et] = 3           # enhancing tumor
    labels = np.zeros(channels.shape[1:], dtype=np.uint8)
    labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = region
    return labels
