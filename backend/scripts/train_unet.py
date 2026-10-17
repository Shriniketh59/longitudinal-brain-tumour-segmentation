"""Train the 2D U-Net (VGG16 encoder) on slice/mask pairs.

Expected layout:
    data/segmentation/train/images/*.png   grayscale slices
    data/segmentation/train/masks/*.png    label maps, pixel values 0..3 (see SEG_CLASSES)
    data/segmentation/val/images/*.png
    data/segmentation/val/masks/*.png
Image and mask filenames must match.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from app.models.unet import SEG_CLASSES, UNetVGG16

BACKEND_DIR = Path(__file__).resolve().parents[1]
SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SliceDataset(Dataset):
    def __init__(self, root: Path, augment: bool):
        self.images = sorted((root / "images").glob("*.png"))
        self.masks_dir = root / "masks"
        self.augment = augment
        if not self.images:
            raise SystemExit(f"No images found in {root / 'images'}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        mask = Image.open(self.masks_dir / img_path.name).convert("L").resize((SIZE, SIZE), Image.NEAREST)

        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.int64)

        if self.augment and np.random.rand() < 0.5:
            img_arr = np.fliplr(img_arr).copy()
            mask_arr = np.fliplr(mask_arr).copy()

        rgb = np.stack([img_arr] * 3, axis=-1)
        rgb = (rgb - MEAN) / STD
        return torch.from_numpy(rgb.transpose(2, 0, 1)), torch.from_numpy(mask_arr)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=len(SEG_CLASSES)).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    # Skip background in the Dice term; it dominates the pixel count.
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice[1:].mean()


ENCODER = ("enc1", "enc2", "enc3", "enc4", "bottleneck")
REGIONS = {"whole_tumor": (1, 2, 3), "tumor_core": (1, 3), "enhancing_tumor": (3,)}


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total, seen = 0.0, 0
    start = time.time()
    with torch.set_grad_enabled(train):
        for step, (images, masks) in enumerate(loader, 1):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks) + dice_loss(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * images.size(0)
            seen += images.size(0)
            if train and step % 20 == 0:
                rate = seen / (time.time() - start)
                left = (len(loader.dataset) - seen) / rate / 60
                print(f"    step {step}/{len(loader)} loss {total / seen:.4f} {rate:.1f} img/s eta {left:.1f} min", flush=True)
    return total / seen


@torch.inference_mode()
def evaluate_dice(model, loader, device) -> dict:
    """Dataset-level Dice per BraTS region (sums over all val pixels)."""
    model.eval()
    inter = {k: 0 for k in REGIONS}
    denom = {k: 0 for k in REGIONS}
    for images, masks in loader:
        pred = model(images.to(device)).argmax(1).cpu()
        for k, labels in REGIONS.items():
            lab = torch.tensor(labels)
            p, g = torch.isin(pred, lab), torch.isin(masks, lab)
            inter[k] += (p & g).sum().item()
            denom[k] += p.sum().item() + g.sum().item()
    return {k: round(2 * inter[k] / denom[k], 4) if denom[k] else None for k in REGIONS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BACKEND_DIR / "data" / "segmentation")
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "weights" / "unet_vgg16.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--freeze-encoder", action="store_true", help="keep the ImageNet VGG16 encoder fixed (CPU friendly)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    train_loader = DataLoader(SliceDataset(args.data / "train", augment=True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(SliceDataset(args.data / "val", augment=False), batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    model = UNetVGG16(num_classes=len(SEG_CLASSES), pretrained_encoder=True).to(device)
    if args.freeze_encoder:
        for name in ENCODER:
            for p in getattr(model, name).parameters():
                p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"device {device} · threads {torch.get_num_threads()} · trainable params {sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va = run_epoch(model, val_loader, criterion, None, device, train=False)
        scheduler.step()
        print(f"epoch {epoch:02d} train loss {tr:.4f} | val loss {va:.4f}", flush=True)
        if va < best:
            best = va
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out} (val loss {best:.4f})", flush=True)

    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True))
    dice = evaluate_dice(model, val_loader, device)
    metrics = {
        "dice": dice,
        "best_val_loss": round(best, 4),
        "epochs": args.epochs,
        "freeze_encoder": args.freeze_encoder,
        "train_slices": len(train_loader.dataset),
        "val_slices": len(val_loader.dataset),
    }
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
