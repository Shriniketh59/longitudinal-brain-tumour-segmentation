"""Head cropping for 2D MRI slices.

Removes background, image borders and burned-in artifacts so the classifier can only
use anatomy. The same function is copied into the Colab training notebook; keep both
in sync, or a model trained with one will see different inputs at inference.
"""

import numpy as np
from scipy import ndimage


def crop_to_head(gray: np.ndarray, margin: float = 0.04) -> np.ndarray:
    """Square crop around the largest bright connected region (the head), zero padded."""
    thr = max(15.0, float(np.percentile(gray, 30)) * 0.5)
    mask = ndimage.binary_opening(gray > thr, iterations=2)
    labels, n = ndimage.label(mask)
    if n == 0:
        return gray
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    head = labels == (int(np.argmax(sizes)) + 1)
    if head.sum() < 0.02 * gray.size:  # nothing head-like found; leave the image alone
        return gray

    ys, xs = np.where(head)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = int(margin * max(y1 - y0, x1 - x0))
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(gray.shape[0], y1 + pad), min(gray.shape[1], x1 + pad)

    # Keep only the head component so text, rulers or border lines inside the box vanish.
    region = np.where(ndimage.binary_dilation(head, iterations=3), gray, 0)[y0:y1, x0:x1]
    h, w = region.shape
    side = max(h, w)
    out = np.zeros((side, side), dtype=gray.dtype)
    out[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = region
    return out
