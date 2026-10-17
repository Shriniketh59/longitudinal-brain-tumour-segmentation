"""Download the 4-class brain tumor MRI dataset and lay it out for train_classifier.py.

Source: BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset on Hugging Face, a mirror of
Masoud Nickparvar's Kaggle "Brain Tumor MRI Dataset" (CC BY 4.0): 5,600 training and
1,600 testing JPEGs in glioma / meningioma / notumor / pituitary.

Result:
    data/classifier/train/{glioma,meningioma,no_tumor,pituitary}/*.jpg   (from Training/)
    data/classifier/val/{...}/*.jpg                                        (from Testing/)
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset"
BACKEND_DIR = Path(__file__).resolve().parents[1]
FOLDER_TO_CLASS = {"glioma": "glioma", "meningioma": "meningioma", "notumor": "no_tumor", "pituitary": "pituitary"}
SPLITS = {"Training": "train", "Testing": "val"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=BACKEND_DIR / "data" / "raw" / "nickparvar")
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "data" / "classifier")
    args = parser.parse_args()

    snapshot_download(REPO_ID, repo_type="dataset", local_dir=args.raw, allow_patterns=["Training/*", "Testing/*"], max_workers=8)

    for src_split, dst_split in SPLITS.items():
        for folder, cls in FOLDER_TO_CLASS.items():
            src = args.raw / src_split / folder
            dst = args.out / dst_split / cls
            dst.mkdir(parents=True, exist_ok=True)
            files = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            for f in files:
                target = dst / f.name
                if not target.exists():
                    shutil.copy2(f, target)
            print(f"{dst_split}/{cls}: {len(files)}")


if __name__ == "__main__":
    main()
