"""Pre-compute every BraTS sample case through the running API so the site answers instantly.

Usage (API must be running):  python -m scripts.warm_cache [--models unet2d swin3d] [--fast]
Runs one case at a time; expect ~45 s (2D) and ~80 s (3D) per case on a laptop CPU, ~25 s each in fast mode.
"""

import argparse
import json
import time
import urllib.request

API = "http://127.0.0.1:8000"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["unet2d", "swin3d"], choices=["unet2d", "swin3d"])
    parser.add_argument("--fast", action="store_true", help="warm the fast-mode results instead")
    parser.add_argument("--api", default=API)
    args = parser.parse_args()

    cases = json.load(urllib.request.urlopen(f"{args.api}/api/volume/samples"))["cases"]
    for model in args.models:
        for case in cases:
            t = time.time()
            url = f"{args.api}/api/volume/samples/{case['id']}?model={model}&fast={str(args.fast).lower()}"
            try:
                res = json.load(urllib.request.urlopen(url, timeout=1800))
                state = "cached" if res.get("cached") else f"{time.time() - t:.0f}s"
            except Exception as exc:  # keep going if one case fails
                state = f"failed: {exc}"
            print(f"{model:7s} {case['id']}  {state}", flush=True)


if __name__ == "__main__":
    main()
