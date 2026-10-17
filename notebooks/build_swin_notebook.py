"""Builds notebooks/train_swin_unetr_3d.ipynb. Run: python notebooks/build_swin_notebook.py"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells: list[tuple[str, str]] = []
md = lambda s: cells.append(("markdown", s.strip("\n")))
code = lambda s: cells.append(("code", s.strip("\n")))

md(r"""
# NeuroScan AI · 3D Vision Transformer (SwinUNETR) for brain tumor segmentation

Trains a **3D Swin Transformer U-Net (SwinUNETR, MONAI)** on whole BraTS 2023 MRI volumes, not slices.
The encoder starts from MONAI's **self-supervised pretrained Swin ViT** (trained on 5,050 CT scans).

- Input: two co-registered 3D sequences per patient, **T1 post-contrast + FLAIR** (what the NeuroScan backend sends).
- Output: three 3D masks, **tumor core, whole tumor, enhancing tumor** (BraTS regions).
- Produces `swin_unetr_brats.pt` and `swin_unetr_brats.metrics.json` (Dice per region on held-out patients).

| | Colab (free T4) | Kaggle (free T4) |
|---|---|---|
| Setup | Runtime → Change runtime type → **T4 GPU** | Accelerator **GPU T4 x2**, Internet **On** |
| Run | Runtime → **Run all**, allow Google Drive | **Run All** |
| Output | downloads `neuroscan_swin3d.zip`, copy in My Drive → NeuroScanAI | Output panel → `neuroscan_swin3d.zip` |

**Memory:** prepared volumes are cached on disk (`monai_cache`), so RAM stays well under Colab's 12.7 GB.
The first epoch is slower while that cache is built.

**Time:** about 2–3 minutes per epoch with 96 training patients, so 30 epochs ≈ 70–90 minutes plus data loading.
**Resumable:** a resume file is saved after every epoch. If the session stops, Run all again and it continues.

Research and education only. Not for clinical use.
""")

code(r"""
# Settings
import os, sys, json, time, random, subprocess, shutil, urllib.request
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle/working")
SMOKE = os.environ.get("NEUROSCAN_SMOKE") == "1"   # tiny CPU run used to test this notebook

WORK = Path(os.environ.get("NEUROSCAN_WORK", "/kaggle/working" if IN_KAGGLE else "/content"))
BRATS_RAW = WORK / "brats2023"
OUT = WORK / "neuroscan_swin"
SAVE_TO_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/NeuroScanAI"

MODALITIES = ("t1c", "t2f")      # the backend sends exactly these two sequences, in this order
CASES = 2 if SMOKE else 120      # of 1,251; 20% are held out for validation
ROI = (64, 64, 64) if SMOKE else (96, 96, 96)
EPOCHS = 1 if SMOKE else 30
VAL_EVERY = 1 if SMOKE else 3
USE_PRETRAINED = True            # self-supervised Swin ViT encoder weights from MONAI
SSL_URL = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
SEED = 42

if IN_COLAB or IN_KAGGLE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "monai>=1.4", "einops", "nibabel", "huggingface_hub"], check=True)
    if IN_COLAB and SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")   # allow access in the popup
        OUT = Path(DRIVE_DIR) / "weights_swin"

import numpy as np
import torch
from monai import transforms as T
from monai.data import DataLoader, PersistentDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.utils import set_determinism

set_determinism(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
OUT.mkdir(parents=True, exist_ok=True)
print("saving to:", OUT)
print("device:", DEVICE, torch.cuda.get_device_name(0) if AMP else "", "| smoke test" if SMOKE else "")
if not AMP and not SMOKE:
    print("WARNING: no GPU. Switch the runtime to a T4 GPU and run again.")
""")

md("## 1 · Data: BraTS 2023 3D volumes (T1c + FLAIR + expert segmentation)")

code(r"""
from huggingface_hub import list_repo_files, snapshot_download

REPO = "MedOtter/brats2023-gli-dataset"

def complete_cases():
    out = []
    for seg in BRATS_RAW.rglob("*-seg.nii.gz"):
        d = seg.parent
        if all((d / f"{d.name}-{m}.nii.gz").exists() for m in MODALITIES):
            out.append(d)
    return sorted(out, key=lambda p: p.name)

if len(complete_cases()) < CASES:
    files = list_repo_files(REPO, repo_type="dataset")
    ids = sorted({f.split("/")[1] for f in files if f.endswith("-seg.nii.gz")})[:CASES]
    t = time.time()
    snapshot_download(REPO, repo_type="dataset", local_dir=BRATS_RAW, max_workers=16,
                      allow_patterns=[f"*/{c}/{c}-{m}.nii.gz" for c in ids for m in (*MODALITIES, "seg")])
    print(f"downloaded {len(ids)} cases in {time.time() - t:.0f}s")

dirs = complete_cases()[:CASES]
items = [{"image": [str(d / f"{d.name}-{m}.nii.gz") for m in MODALITIES],
          "label": str(d / f"{d.name}-seg.nii.gz"), "id": d.name} for d in dirs]
random.Random(SEED).shuffle(items)
n_val = max(1, round(len(items) * 0.2))
val_items, train_items = items[:n_val], items[n_val:] or items[:1]
print(f"train {len(train_items)} patients · val {len(val_items)} patients")
""")

md(r"""
## 2 · Transforms

BraTS 2023 labels (1 necrotic core, 2 edema, 3 enhancing) become three overlapping region masks.
Intensities are z-scored per sequence over brain voxels only; the backend repeats exactly this.
""")

code(r"""
class BratsRegionsd(T.MapTransform):
    # Label map -> 3 channels: tumor core (1|3), whole tumor (1|2|3), enhancing tumor (3).
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            lab = torch.as_tensor(d[k])
            # uint8 keeps the cached masks 4x smaller than float; cast to float in the training loop.
            d[k] = torch.cat([(lab == 1) | (lab == 3), (lab == 1) | (lab == 2) | (lab == 3), lab == 3], dim=0).to(torch.uint8)
        return d

base = [
    T.LoadImaged(keys=["image", "label"]),
    T.EnsureChannelFirstd(keys=["image", "label"]),
    BratsRegionsd(keys="label"),
    T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    T.CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=list(ROI)),
]
train_tf = T.Compose(base + [
    T.RandSpatialCropd(keys=["image", "label"], roi_size=ROI, random_size=False),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
])
val_tf = T.Compose(base)

workers = 2 if (IN_COLAB or IN_KAGGLE) else 0
# Deterministic steps (load, normalise, crop) are cached on DISK once, not in RAM:
# 120 patients do not fit in Colab's 12.7 GB of memory. Random crops/flips run every epoch.
CACHE = WORK / "monai_cache"
train_ds = PersistentDataset(train_items, train_tf, cache_dir=CACHE / "train")
val_ds = PersistentDataset(val_items, val_tf, cache_dir=CACHE / "val")
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=workers, pin_memory=AMP)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=AMP)
x0 = train_ds[0]["image"]
print("training patch:", tuple(x0.shape), "(channels, x, y, z)")
""")

md("## 3 · SwinUNETR with the pretrained Swin ViT encoder")

code(r"""
def build_swin_unetr(checkpointing=True):
    # Same configuration as backend/app/models/swin3d.py. Do not change without updating both.
    return SwinUNETR(in_channels=len(MODALITIES), out_channels=3, feature_size=48, use_checkpoint=checkpointing)

def load_ssl_encoder(model, path):
    # Self-supervised weights were saved by an older Swin implementation with 1 input channel:
    # rename the MLP layers and spread the first conv over our input channels.
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k.removeprefix("module.").replace("mlp.fc1", "mlp.linear1").replace("mlp.fc2", "mlp.linear2"): v
          for k, v in sd.items()}
    own = model.swinViT.state_dict()
    w = sd.get("patch_embed.proj.weight")
    if w is not None and w.shape[1] != own["patch_embed.proj.weight"].shape[1]:
        c = own["patch_embed.proj.weight"].shape[1]
        sd["patch_embed.proj.weight"] = w.repeat(1, c, 1, 1, 1) / c
    loaded = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    own.update(loaded)
    model.swinViT.load_state_dict(own)
    return len(loaded), len(own)

model = build_swin_unetr().to(DEVICE)
pretrained = False
if USE_PRETRAINED:
    ssl_path = WORK / "model_swinvit.pt"
    try:
        if not ssl_path.exists():
            urllib.request.urlretrieve(SSL_URL, ssl_path)
        n, total = load_ssl_encoder(model, ssl_path)
        pretrained = n == total
        print(f"pretrained Swin ViT encoder: {n}/{total} tensors loaded")
    except Exception as exc:  # network hiccup etc.: train from scratch rather than stop
        print("could not load pretrained encoder, training from scratch:", exc)
print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
""")

md("## 4 · Train (resumable)")

code(r"""
REGIONS = ("tumor_core", "whole_tumor", "enhancing_tumor")
loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, sigmoid=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler(enabled=AMP)
dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
post = T.Compose([T.Activations(sigmoid=True), T.AsDiscrete(threshold=0.5)])

weights_path, resume_path = OUT / "swin_unetr_brats.pt", OUT / "swin_unetr_resume.pt"
best = {"mean": -1.0, "epoch": 0, "dice": None}
start = 1
if resume_path.exists():
    st = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
    best, start = st["best"], st["epoch"] + 1
    print(f"resuming after epoch {st['epoch']} (best mean Dice {best['mean']:.4f})")

def validate():
    model.eval()
    with torch.inference_mode(), torch.autocast(DEVICE.type, enabled=AMP):
        for batch in val_loader:
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE).float()
            logits = sliding_window_inference(x, ROI, sw_batch_size=2, predictor=model, overlap=0.5)
            dice_metric(y_pred=[post(i) for i in decollate_batch(logits)], y=decollate_batch(y))
    scores = dice_metric.aggregate().tolist()
    dice_metric.reset()
    return dict(zip(REGIONS, [round(float(s), 4) for s in scores]))

for epoch in range(start, EPOCHS + 1):
    model.train()
    t, total = time.time(), 0.0
    for step, batch in enumerate(train_loader, 1):
        x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE).float()
        with torch.autocast(DEVICE.type, enabled=AMP):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
    sched.step()
    msg = f"epoch {epoch:02d}  loss {total / step:.4f}  {time.time() - t:.0f}s"
    if epoch % VAL_EVERY == 0 or epoch == EPOCHS:
        dice = validate()
        mean = sum(dice.values()) / 3
        msg += f"  val Dice TC {dice['tumor_core']:.3f} WT {dice['whole_tumor']:.3f} ET {dice['enhancing_tumor']:.3f}"
        if mean > best["mean"]:
            best = {"mean": round(mean, 4), "epoch": epoch, "dice": dice}
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights_path)
            msg += "  ← saved"
    torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "best": best}, resume_path)
    print(msg)
""")

md("## 5 · Metrics, check and download")

code(r"""
metrics = {
    "model": "SwinUNETR (3D Swin Transformer U-Net)",
    "dice": best["dice"],
    "mean_dice": best["mean"],
    "best_epoch": best["epoch"],
    "epochs": EPOCHS,
    "modalities": list(MODALITIES),
    "roi": list(ROI),
    "feature_size": 48,
    "pretrained_encoder": pretrained,
    "train_cases": len(train_items),
    "val_cases": len(val_items),
    "normalization": "z-score per sequence over nonzero voxels",
}
(OUT / "swin_unetr_brats.metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# Reload the way the backend does: fresh model, strict keys, weights_only.
m = build_swin_unetr(checkpointing=False)
m.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
with torch.inference_mode():
    assert m.eval()(torch.zeros(1, len(MODALITIES), *ROI)).shape == (1, 3, *ROI)

pack = WORK / "neuroscan_swin3d"
pack.mkdir(exist_ok=True)
for name in ("swin_unetr_brats.pt", "swin_unetr_brats.metrics.json"):
    shutil.copy2(OUT / name, pack / name)
archive = shutil.make_archive(str(OUT.parent / "neuroscan_swin3d"), "zip", pack)
for f in sorted(pack.iterdir()):
    print(f"{f.name:<34} {f.stat().st_size / 1e6:7.1f} MB")
print("archive:", archive)
if IN_COLAB:
    from google.colab import files
    files.download(archive)
elif IN_KAGGLE:
    print("Download neuroscan_swin3d.zip from the Output panel (/kaggle/working).")
""")

md(r"""
## Install on your computer

```bash
cd "Longitutdinal Brain Tumour Segmentation/backend"
unzip -o ~/Downloads/neuroscan_swin3d.zip -d weights/
pip install "monai>=1.4" einops
```

Restart the backend. The 3D volume lab then offers **3D SwinUNETR** next to the 2D U-Net.
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
    "cells": [{"cell_type": k, "metadata": {}, "id": f"cell-{i}", "source": s.splitlines(keepends=True),
               **({"execution_count": None, "outputs": []} if k == "code" else {})} for i, (k, s) in enumerate(cells)],
}
(HERE / "train_swin_unetr_3d.ipynb").write_text(json.dumps(nb, indent=1))
(HERE.parent / "backend" / "data" / "swin-smoke.py").write_text("\n\n".join(s for k, s in cells if k == "code"))
print("wrote train_swin_unetr_3d.ipynb")
