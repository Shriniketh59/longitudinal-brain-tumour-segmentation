# NeuroScan AI backend

FastAPI service. Two pretrained models:

- `VGG16` slice classifier (ImageNet weights, 4-way head): `POST /api/classify`
- `U-Net` with VGG16 encoder for segmentation: `POST /api/segment`

## Run

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Weights go in `backend/weights/`:

- `vgg16_classifier.pt`
- `unet_vgg16.pt`

Without weights the API starts, and the matching endpoint returns 503.

The browser never calls port 8000 directly: the Next.js app on http://localhost:3000
proxies `/api/*` here (set `BACKEND_URL` in the frontend to change the target).
Override paths with `CLASSIFIER_WEIGHTS`, `UNET_WEIGHTS`, `DEVICE`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`.

## Data

```bash
python -m scripts.download_brats --cases 20        # 3D BraTS 2023 volumes (use --all for ~1,250 cases, 13.4 GB)
python -m scripts.prepare_brats_slices             # 2D slice/mask PNGs, split by case
```

## Train

```bash
python -m scripts.train_classifier --data data/classifier --epochs 10
python -m scripts.train_unet --data data/segmentation --epochs 30
```

Dataset layouts are documented at the top of each script.

## Test

```bash
pytest -q
```

Research use only. Not cleared for clinical use.
