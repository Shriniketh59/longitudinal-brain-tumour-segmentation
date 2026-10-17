"""Serverless deployment of the NeuroScan AI backend on Modal (https://modal.com).

One-time setup:
    pip install modal && python -m modal setup
    python -m modal volume create neuroscan-weights
    python -m modal volume put neuroscan-weights weights/ /             # the three .pt files + metrics
    python -m modal run modal_app.py::download_samples                 # BraTS sample cases for the 3D lab

Deploy (from backend/):
    python -m modal deploy modal_app.py

Then set NEXT_PUBLIC_API_URL in Vercel to the printed https://...modal.run URL and add that
Vercel URL to FRONTEND_ORIGINS below (or the CORS_ORIGINS secret).
"""

import modal

FRONTEND_ORIGINS = "https://longitudinal-brain-tumour-segmentat.vercel.app,http://localhost:3000"

weights = modal.Volume.from_name("neuroscan-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # CPU-only PyTorch keeps the image small and starts faster.
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "fastapi>=0.115", "python-multipart>=0.0.9", "pillow>=10.0", "numpy>=1.26", "nibabel>=5.2",
        "huggingface_hub>=0.24", "reportlab>=4.0", "scipy>=1.11", "monai>=1.4", "einops>=0.7",
    )
    .env({
        "CLASSIFIER_WEIGHTS": "/data/vgg16_classifier.pt",
        "UNET_WEIGHTS": "/data/unet_vgg16.pt",
        "SWIN_WEIGHTS": "/data/swin_unetr_brats.pt",
        "BRATS_DIR": "/data/brats2023",
        "CACHE_DIR": "/tmp/neuroscan-cache",
        "CORS_ORIGINS": FRONTEND_ORIGINS,
        "DEVICE": "cpu",
    })
    .add_local_dir("app", remote_path="/root/app")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

app = modal.App("neuroscan-api", image=image)


@app.function(volumes={"/data": weights}, cpu=4.0, memory=6144, timeout=900, scaledown_window=600)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def api():
    from app.main import app as fastapi_app

    return fastapi_app


@app.function(volumes={"/data": weights}, timeout=1800)
def download_samples(cases: int = 10):
    """Fetch BraTS 2023 sample cases (CC BY 4.0) into the volume for the 3D lab."""
    import subprocess

    subprocess.run(
        ["python", "-m", "scripts.download_brats", "--cases", str(cases), "--out", "/data/brats2023"],
        check=True, cwd="/root",
    )
    weights.commit()
