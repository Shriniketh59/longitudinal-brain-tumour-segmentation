"""Download BraTS 2023 Adult Glioma 3D MRI volumes from Hugging Face.

Source: MedOtter/brats2023-gli-dataset (ASNR-MICCAI BraTS 2023 GLI training data,
about 1,250 cases, 13.4 GB for all modalities). Each case has NIfTI volumes:
t1n, t1c, t2w, t2f (FLAIR) and seg (labels 0 background, 1 necrotic core,
2 edema, 3 enhancing tumor).

Examples:
    python -m scripts.download_brats --cases 20
    python -m scripts.download_brats --all --modalities t1c t2f seg
"""

import argparse
import re
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

REPO_ID = "MedOtter/brats2023-gli-dataset"
MODALITIES = ("t1n", "t1c", "t2w", "t2f", "seg")
BACKEND_DIR = Path(__file__).resolve().parents[1]
CASE_RE = re.compile(r"/(BraTS-GLI-\d{5}-\d{3})/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "data" / "brats2023")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cases", type=int, default=20, help="number of cases to fetch (default 20)")
    group.add_argument("--all", action="store_true", help="fetch every case")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=["t1c", "t2f", "seg"])
    args = parser.parse_args()

    files = list_repo_files(REPO_ID, repo_type="dataset")
    cases = sorted({m.group(1) for f in files if (m := CASE_RE.search(f))})
    chosen = cases if args.all else cases[: args.cases]
    print(f"{len(cases)} cases available, downloading {len(chosen)} ({', '.join(args.modalities)})")

    patterns = [f"*/{case}/{case}-{mod}.nii.gz" for case in chosen for mod in args.modalities]
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=args.out,
        allow_patterns=patterns,
        max_workers=8,
    )
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
