"""Batch analysis of video frames: tumor-type classification, segmentation and detection boxes.

Frames are independent 2D images (sampled in the browser). Each batch runs the VGG16
classifier and the 2D U-Net once over all frames, which is much faster than one request
per frame. Detection boxes come from the largest connected tumor region of the mask, and
count only when the classifier also finds tumor (tumor probability >= 0.5).
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage

from app.inference import CLS_SIZE, SEG_SIZE, PreparedSlice, _to_tensor
from app.malignancy import assess
from app.models.vgg16_classifier import CLASS_NAMES
from app.volume import mask_png

MAX_FRAMES = 16
# Ignore specks: a region must cover at least this share of the frame to count as a detection.
MIN_AREA_FRACTION = 0.002


def _detect(mask: np.ndarray) -> tuple[dict | None, float]:
    """Bounding box (fractions of width/height) of the largest tumor component, and tumor area %."""
    tumor = mask > 0
    area_pct = float(tumor.mean() * 100)
    if tumor.mean() < MIN_AREA_FRACTION:
        return None, round(area_pct, 3)
    labels, n = ndimage.label(tumor)
    sizes = ndimage.sum(tumor, labels, index=range(1, n + 1))
    biggest = labels == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(biggest)
    h, w = mask.shape
    box = {
        "x": round(xs.min() / w, 4),
        "y": round(ys.min() / h, 4),
        "w": round((xs.max() + 1 - xs.min()) / w, 4),
        "h": round((ys.max() + 1 - ys.min()) / h, 4),
    }
    return box, round(area_pct, 3)


@torch.inference_mode()
def analyse_frames(
    frames: list[PreparedSlice],
    classifier: nn.Module | None,
    segmenter: nn.Module | None,
    device: torch.device,
) -> list[dict]:
    results: list[dict] = [{} for _ in frames]

    if classifier is not None:
        x = torch.cat([_to_tensor(f.gray, CLS_SIZE, device) for f in frames])
        probs = torch.softmax(classifier(x), dim=1).cpu().numpy()
        for r, p in zip(results, probs):
            probabilities = {name: round(float(v), 6) for name, v in zip(CLASS_NAMES, p)}
            label = CLASS_NAMES[int(np.argmax(p))]
            r.update({
                "label": label,
                "probabilities": probabilities,
                "tumor_probability": round(1.0 - probabilities["no_tumor"], 6),
                "malignancy": assess(probabilities),
            })

    if segmenter is not None:
        x = torch.cat([_to_tensor(f.gray, SEG_SIZE, device) for f in frames]).contiguous(memory_format=torch.channels_last)
        preds = segmenter(x).argmax(1).cpu().numpy().astype(np.uint8)
        for r, f, pred in zip(results, frames, preds):
            mask = np.asarray(Image.fromarray(pred).resize(f.size, Image.NEAREST), dtype=np.uint8)
            box, area_pct = _detect(mask)
            r.update({
                "mask_png_base64": mask_png(mask),
                "box": box,
                "area_pct": area_pct,
                "counts": {
                    "necrotic_core": int((mask == 1).sum()),
                    "edema": int((mask == 2).sum()),
                    "enhancing": int((mask == 3).sum()),
                },
            })

    for r in results:
        seg_found = r.get("box") is not None
        cls_tumor = r.get("tumor_probability", 1.0) >= 0.5
        # The U-Net was trained on skull-stripped BraTS gliomas and can mark skull or orbits on
        # ordinary scans, so a detection needs both models to agree when both are loaded.
        if "box" in r and "tumor_probability" in r:
            r["detected"] = seg_found and cls_tumor
            if seg_found and not cls_tumor:
                r.update({"box": None, "mask_png_base64": None, "area_pct": 0.0, "suppressed": "classifier: no tumor"})
        elif "box" in r:
            r["detected"] = seg_found
        else:
            r["detected"] = cls_tumor
    return results
