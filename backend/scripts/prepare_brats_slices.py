"""Cut BraTS 3D volumes into 2D axial slice/mask PNG pairs for train_unet.py.

Reads   data/brats2023/**/<case>/<case>-<modality>.nii.gz and <case>-seg.nii.gz
Writes  data/segmentation/{train,val}/{images,masks}/<case>_z<idx>.png

Masks keep BraTS labels as pixel values (0 background, 1 necrotic core,
2 edema, 3 enhancing), matching app.models.unet.SEG_CLASSES.
The split is by case, so no patient appears in both train and val.

Example:
    python -m scripts.prepare_brats_slices --modality t1c --val-fraction 0.2
"""

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]


def normalise(volume: np.ndarray) -> np.ndarray:
    """Clip to the 1st-99th percentile of brain voxels and scale to uint8."""
    brain = volume[volume > 0]
    if brain.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    lo, hi = np.percentile(brain, (1, 99))
    scaled = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def axial(volume: np.ndarray, z: int) -> np.ndarray:
    # NIfTI (x, y, z) to an upright radiological view.
    return np.rot90(volume[:, :, z])


def find_cases(root: Path, modality: str) -> list[tuple[str, Path, Path]]:
    cases = []
    for seg in sorted(root.rglob("*-seg.nii.gz")):
        case = seg.name.removesuffix("-seg.nii.gz")
        image = seg.with_name(f"{case}-{modality}.nii.gz")
        if image.exists():
            cases.append((case, image, seg))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=BACKEND_DIR / "data" / "brats2023")
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "data" / "segmentation")
    parser.add_argument("--modality", default="t1c", choices=["t1n", "t1c", "t2w", "t2f"])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--empty-ratio", type=float, default=0.15,
                        help="share of tumor-free brain slices to keep, relative to tumor slices")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = find_cases(args.src, args.modality)
    if not cases:
        raise SystemExit(f"No cases with {args.modality} + seg found under {args.src}")

    rng = random.Random(args.seed)
    rng.shuffle(cases)
    n_val = max(1, round(len(cases) * args.val_fraction))
    splits = {"val": cases[:n_val], "train": cases[n_val:]}

    for split, items in splits.items():
        img_dir = args.out / split / "images"
        mask_dir = args.out / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        written = 0

        for case, image_path, seg_path in items:
            image = normalise(np.asarray(nib.load(image_path).dataobj, dtype=np.float32))
            seg = np.asarray(nib.load(seg_path).dataobj).astype(np.uint8)

            tumor_z = [z for z in range(seg.shape[2]) if seg[:, :, z].any()]
            brain_z = [z for z in range(image.shape[2]) if image[:, :, z].any() and z not in set(tumor_z)]
            n_empty = min(len(brain_z), round(len(tumor_z) * args.empty_ratio))
            keep = tumor_z + rng.sample(brain_z, n_empty)

            for z in keep:
                name = f"{case}_z{z:03d}.png"
                Image.fromarray(axial(image, z)).save(img_dir / name)
                Image.fromarray(axial(seg, z)).save(mask_dir / name)
                written += 1

        print(f"{split}: {len(items)} cases, {written} slices")


if __name__ == "__main__":
    main()
