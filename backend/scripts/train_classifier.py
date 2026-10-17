"""Fine-tune the VGG16 slice classifier.

Expected layout (ImageFolder):
    data/classifier/train/<class>/*.jpg
    data/classifier/val/<class>/*.jpg
Class folder names must match app.models.vgg16_classifier.CLASS_NAMES.

Modes:
    full   train every layer with augmentation (needs a GPU to be practical)
    head   CPU friendly transfer learning: keep the ImageNet conv layers and fc6 frozen,
           compute their activations once, then train fc7 + the 4-way output layer on the cache.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from app.models.vgg16_classifier import CLASS_NAMES, build_vgg16_classifier

BACKEND_DIR = Path(__file__).resolve().parents[1]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_transforms(train: bool, flip: bool = False) -> transforms.Compose:
    steps = [transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    elif flip:
        steps += [transforms.RandomHorizontalFlip(p=1.0)]
    steps += [transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
    return transforms.Compose(steps)


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.inference_mode()
def cache_fc6(model, folder: Path, device, batch_size: int, workers: int, flip: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """fc6 activations (after ReLU) for every image in an ImageFolder, stored as float16."""
    ds = datasets.ImageFolder(folder, transform=make_transforms(train=False, flip=flip))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    model.eval()
    feats, labels = [], []
    start, done = time.time(), 0
    for images, y in loader:
        x = model.avgpool(model.features(images.to(device)))
        x = model.classifier[1](model.classifier[0](torch.flatten(x, 1)))
        feats.append(x.half().cpu())
        labels.append(y)
        done += len(y)
        rate = done / (time.time() - start)
        print(f"  cache {folder.name}{' (flipped)' if flip else ''}: {done}/{len(ds)}  {rate:.1f} img/s  eta {(len(ds) - done) / rate / 60:.1f} min", flush=True)
    return torch.cat(feats), torch.cat(labels)


def train_head(model, args, device) -> float:
    cache_path = args.out.parent / "fc6_cache.pt"
    if cache_path.exists() and not args.rebuild_cache:
        cache = torch.load(cache_path)
        print(f"loaded feature cache {cache_path}")
    else:
        tr_x, tr_y = cache_fc6(model, args.data / "train", device, args.batch_size, args.workers, flip=False)
        if args.cache_flip:
            fx, fy = cache_fc6(model, args.data / "train", device, args.batch_size, args.workers, flip=True)
            tr_x, tr_y = torch.cat([tr_x, fx]), torch.cat([tr_y, fy])
        va_x, va_y = cache_fc6(model, args.data / "val", device, args.batch_size, args.workers, flip=False)
        cache = {"train": (tr_x, tr_y), "val": (va_x, va_y)}
        torch.save(cache, cache_path)

    # Head = dropout, fc7, ReLU, dropout, fc8. Conv layers and fc6 stay as ImageNet weights.
    head = model.classifier[2:]
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True

    train_loader = DataLoader(TensorDataset(*cache["train"]), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(*cache["val"]), batch_size=256)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(head, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(head, val_loader, criterion, None, device, train=False)
        scheduler.step()
        print(f"epoch {epoch:02d} train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}", flush=True)
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out} (val acc {best:.4f})", flush=True)
    return best


def train_full(model, args, device) -> float:
    train_ds = datasets.ImageFolder(args.data / "train", transform=make_transforms(train=True))
    val_ds = datasets.ImageFolder(args.data / "val", transform=make_transforms(train=False))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        scheduler.step()
        print(f"epoch {epoch:02d} train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}", flush=True)
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out} (val acc {best:.4f})", flush=True)
    return best


@torch.inference_mode()
def evaluate(model, args, device) -> dict:
    """Per-class metrics and confusion matrix on the val split with the saved best weights."""
    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True))
    model.eval()
    ds = datasets.ImageFolder(args.data / "val", transform=make_transforms(train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    n = len(CLASS_NAMES)
    cm = [[0] * n for _ in range(n)]
    for images, y in loader:
        pred = model(images.to(device)).argmax(1).cpu()
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t][p] += 1
    per_class = {}
    for i, name in enumerate(ds.classes):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[r][i] for r in range(n)) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": sum(cm[i])}
    total = sum(map(sum, cm))
    return {
        "accuracy": round(sum(cm[i][i] for i in range(n)) / total, 4),
        "classes": ds.classes,
        "confusion_matrix": cm,
        "per_class": per_class,
        "val_images": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BACKEND_DIR / "data" / "classifier")
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "weights" / "vgg16_classifier.pt")
    parser.add_argument("--mode", choices=["head", "full"], default="head")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="default 3e-4 (head) or 1e-4 (full)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--cache-flip", action="store_true", help="head mode: also cache horizontally flipped train images")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    args.lr = args.lr or (3e-4 if args.mode == "head" else 1e-4)

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = datasets.ImageFolder(args.data / "train").classes
    if classes != sorted(CLASS_NAMES):
        raise SystemExit(f"Class folders {classes} do not match {sorted(CLASS_NAMES)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = build_vgg16_classifier(num_classes=len(CLASS_NAMES), pretrained=True).to(device)
    print(f"mode {args.mode} · device {device} · threads {torch.get_num_threads()}", flush=True)
    best = train_head(model, args, device) if args.mode == "head" else train_full(model, args, device)

    metrics = evaluate(model, args, device)
    metrics.update({"mode": args.mode, "epochs": args.epochs, "best_val_acc_during_training": round(best, 4)})
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
