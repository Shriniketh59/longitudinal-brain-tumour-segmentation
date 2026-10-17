import json
import re
import threading
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

import nibabel as nib
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import UnidentifiedImageError

import hashlib

from app import inference, video, volume
from app.cache import ResultCache
from app.config import settings
from app.longitudinal import LongitudinalRequest, analyse as analyse_longitudinal
from app.report import ReportRequest, render_report
from app.models.unet import load_unet
from app.models.vgg16_classifier import load_vgg16_classifier


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = resolve_device(settings.device)
    app.state.device = device
    # Missing weights leave the model unset; the matching endpoint returns 503.
    app.state.classifier = (
        load_vgg16_classifier(str(settings.classifier_weights), device)
        if settings.classifier_weights.exists()
        else None
    )
    app.state.segmenter = (
        load_unet(str(settings.unet_weights), device) if settings.unet_weights.exists() else None
    )
    # The 3D transformer is optional: it needs MONAI and its own weights file.
    app.state.swin3d, app.state.swin3d_meta = None, {}
    if settings.swin_weights.exists():
        try:
            from app.models.swin3d import load_swin_unetr

            app.state.swin3d, app.state.swin3d_meta = load_swin_unetr(settings.swin_weights, device)
        except ImportError as exc:
            print(f"3D SwinUNETR weights found but MONAI is missing ({exc}); pip install monai einops")
    # Cached results are keyed on the weights' modification times, so retrained models never serve stale output.
    app.state.cache_version = "|".join(
        f"{p.name}:{int(p.stat().st_mtime)}" for p in (settings.unet_weights, settings.swin_weights) if p.exists()
    )
    # The training notebook records its input preprocessing next to the weights.
    cls_meta = settings.classifier_weights.with_suffix(".metrics.json")
    app.state.classifier_preprocess = (
        json.loads(cls_meta.read_text()).get("preprocess", "none") if cls_meta.exists() else "none"
    )
    yield


app = FastAPI(title="NeuroScan AI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    # The frontend reads the PDF filename from this header on cross-origin deployments.
    expose_headers=["Content-Disposition"],
)


async def _read_scan(file: UploadFile) -> inference.PreparedSlice:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return inference.decode_slice(data)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Not a readable image") from exc


@app.get("/health")
@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(app.state.device),
        "classifier_loaded": app.state.classifier is not None,
        "segmenter_loaded": app.state.segmenter is not None,
        "classifier_preprocess": app.state.classifier_preprocess,
        "swin3d_loaded": app.state.swin3d is not None,
    }


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)) -> dict[str, Any]:
    if app.state.classifier is None:
        raise HTTPException(status_code=503, detail="Classifier weights not loaded")
    scan = await _read_scan(file)
    crop = app.state.classifier_preprocess == "crop_head"
    return await run_in_threadpool(inference.classify, app.state.classifier, scan, app.state.device, crop=crop)


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)) -> dict[str, Any]:
    if app.state.segmenter is None:
        raise HTTPException(status_code=503, detail="Segmenter weights not loaded")
    scan = await _read_scan(file)
    return inference.segment(app.state.segmenter, scan, app.state.device)


@app.post("/api/video/analyze")
async def video_analyze(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Classify, segment and detect tumor in a batch of video frames (max 16 per request)."""
    if app.state.classifier is None and app.state.segmenter is None:
        raise HTTPException(status_code=503, detail="No models loaded")
    if not files or len(files) > video.MAX_FRAMES:
        raise HTTPException(status_code=400, detail=f"Send 1 to {video.MAX_FRAMES} frames per request")
    frames = [await _read_scan(f) for f in files]
    results = await run_in_threadpool(
        video.analyse_frames, frames, app.state.classifier, app.state.segmenter, app.state.device
    )
    return {
        "frames": results,
        "classifier": app.state.classifier is not None,
        "segmenter": app.state.segmenter is not None,
    }


@app.get("/api/models/metrics")
async def model_metrics() -> dict[str, Any]:
    """Evaluation metrics written by the training scripts next to each weights file."""
    out: dict[str, Any] = {}
    for key, weights in (
        ("classifier", settings.classifier_weights),
        ("segmenter", settings.unet_weights),
        ("swin3d", settings.swin_weights),
    ):
        path = weights.with_suffix(".metrics.json")
        out[key] = json.loads(path.read_text()) if path.exists() else None
    return out


# ---------------------------------------------------------------------------
# 3D volumes
# ---------------------------------------------------------------------------

CASE_ID_RE = re.compile(r"^BraTS-[A-Z]+-\d{5}-\d{3}$")
NIFTI_SUFFIXES = (".nii", ".nii.gz")


@app.get("/api/volume/samples")
async def volume_samples() -> dict[str, Any]:
    return {"cases": volume.list_sample_cases(settings.brats_dir)}


SWIN_NAME = "3D SwinUNETR (Swin Transformer)"


CACHE = ResultCache(settings.cache_dir)

# One 3D job at a time: each sliding-window pass needs 1.5-2.5 GB of RAM on CPU.
_SWIN_LOCK = threading.Lock()


def _swin_roi() -> tuple[int, int, int]:
    from app.models.swin3d import DEFAULT_ROI

    if settings.swin_roi:
        return settings.swin_roi
    # On CPU a 64^3 window cuts peak memory from ~2.4 GB to ~1.5 GB with the same Dice
    # (measured on BraTS-GLI-00000-000: WT 0.930 vs 0.917). On GPU use the training patch.
    if app.state.device.type == "cpu":
        return (64, 64, 64)
    return tuple(app.state.swin3d_meta.get("roi", DEFAULT_ROI))


def _swin_predict(channels: list[np.ndarray], fast: bool = False) -> np.ndarray:
    from app.models.swin3d import FAST_SCALE, predict_labels

    with _SWIN_LOCK:
        return predict_labels(
            app.state.swin3d, np.stack(channels), app.state.device, roi=_swin_roi(), scale=FAST_SCALE if fast else 1.0
        )


def _swin_name(fast: bool) -> str:
    return SWIN_NAME + (" · fast mode (75% scale)" if fast else "")


def _require_swin() -> None:
    if app.state.swin3d is None:
        raise HTTPException(status_code=503, detail="3D SwinUNETR weights not loaded")


def _sample_analysis(
    case_id: str, modality: str, step: int, predict: bool, model_kind: str = "unet2d", fast: bool = False
) -> dict[str, Any]:
    case_dir = volume.find_case_dir(settings.brats_dir, case_id)
    if case_dir is None:
        raise HTTPException(status_code=404, detail="Case not found")
    image_path = case_dir / f"{case_id}-{modality}.nii.gz"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Modality {modality} not available for this case")
    vol, spacing = volume.to_volume(nib.load(str(image_path)))
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    gt = np.asarray(nib.load(str(seg_path)).dataobj).astype(np.uint8) if seg_path.exists() else None
    if predict and model_kind == "swin3d":
        paths = [case_dir / f"{case_id}-{m}.nii.gz" for m in ("t1c", "t2f")]
        if not all(p.exists() for p in paths):
            raise HTTPException(status_code=404, detail="3D model needs T1c and FLAIR for this case")
        channels = [volume.to_volume(nib.load(str(p)))[0] for p in paths]
        result = volume.analyse(vol, spacing, step=step, ground_truth=gt,
                                prediction=_swin_predict(channels, fast), model_name=_swin_name(fast))
    else:
        model = app.state.segmenter if predict else None
        result = volume.analyse(vol, spacing, step=step, ground_truth=gt, model=model, device=app.state.device, fast=fast)
    result.update({"id": case_id, "modality": modality, "source": "BraTS 2023 GLI"})
    return result


async def _cached(key: str, compute, *args) -> dict[str, Any]:
    """Serve a stored result instantly, otherwise compute it in the threadpool and store it."""
    hit = await run_in_threadpool(CACHE.get, key)
    if hit is not None:
        return {**hit, "cached": True}
    result = await run_in_threadpool(compute, *args)
    await run_in_threadpool(CACHE.put, key, result)
    return {**result, "cached": False}


@app.get("/api/volume/samples/{case_id}")
async def volume_sample(
    case_id: str,
    modality: str = Query("t1c", pattern="^(t1c|t1n|t2w|t2f)$"),
    step: int = Query(1, ge=1, le=8),
    predict: bool = True,
    model: str = Query("unet2d", pattern="^(unet2d|swin3d)$"),
    fast: bool = False,
) -> dict[str, Any]:
    if not CASE_ID_RE.match(case_id):
        raise HTTPException(status_code=400, detail="Invalid case id")
    if predict and model == "swin3d":
        _require_swin()
    key = CACHE.key("sample", case_id, modality, step, predict, model, fast, app.state.cache_version, _swin_roi_key())
    return await _cached(key, _sample_analysis, case_id, modality, step, predict, model, fast)


def _swin_roi_key() -> str:
    return "x".join(map(str, _swin_roi())) if app.state.swin3d is not None else ""


async def _read_nifti(file: UploadFile, digests: list[str] | None = None) -> nib.Nifti1Image:
    name = (file.filename or "").lower()
    if not name.endswith(NIFTI_SUFFIXES):
        raise HTTPException(status_code=400, detail="Upload a NIfTI file (.nii or .nii.gz)")
    data = await file.read(settings.max_volume_bytes + 1)
    if len(data) > settings.max_volume_bytes:
        raise HTTPException(status_code=413, detail="Volume too large")
    if digests is not None:
        digests.append(hashlib.sha256(data).hexdigest())
    try:
        return await run_in_threadpool(volume.load_nifti_bytes, data, name)
    except Exception as exc:  # nibabel raises many types for bad files
        raise HTTPException(status_code=400, detail="Not a readable NIfTI volume") from exc


@app.post("/api/volume/upload")
async def volume_upload(
    file: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    flair: UploadFile | None = File(None),
    step: int = Form(1),
    predict: bool = Form(True),
    model: str = Form("unet2d", pattern="^(unet2d|swin3d)$"),
    fast: bool = Form(False),
) -> dict[str, Any]:
    digests: list[str] = []
    img = await _read_nifti(file, digests)
    try:
        vol, spacing = volume.to_volume(img)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    gt = None
    if mask is not None and mask.filename:
        gt = np.asarray((await _read_nifti(mask, digests)).dataobj).astype(np.uint8)
        if gt.shape[:3] != vol.shape:
            raise HTTPException(status_code=400, detail="Mask shape does not match the volume")
        gt = gt[..., 0] if gt.ndim == 4 else gt

    step = max(1, min(step, 8))
    flair_vol = None
    if predict and model == "swin3d":
        _require_swin()
        if flair is None or not flair.filename:
            raise HTTPException(status_code=400, detail="The 3D model needs a FLAIR volume as well as the T1c volume")
        flair_vol, _ = volume.to_volume(await _read_nifti(flair, digests))
        if flair_vol.shape != vol.shape:
            raise HTTPException(status_code=400, detail="FLAIR shape does not match the T1c volume")

    def compute() -> dict[str, Any]:
        if flair_vol is not None:
            pred = _swin_predict([vol, flair_vol], fast)
            res = volume.analyse(vol, spacing, step=step, ground_truth=gt, prediction=pred, model_name=_swin_name(fast))
        else:
            seg_model = app.state.segmenter if predict else None
            res = volume.analyse(vol, spacing, step=step, ground_truth=gt, model=seg_model,
                                 device=app.state.device, fast=fast)
        res.update({"id": file.filename, "modality": "upload", "source": "Uploaded NIfTI"})
        return res

    # Same files + same settings = same answer: key on the uploaded bytes, not the filename.
    key = CACHE.key("upload", digests, step, predict, model, fast, app.state.cache_version, _swin_roi_key())
    result = await _cached(key, compute)
    return result


# ---------------------------------------------------------------------------
# Longitudinal tracking
# ---------------------------------------------------------------------------

@app.post("/api/longitudinal/analyze")
async def longitudinal(req: LongitudinalRequest) -> dict[str, Any]:
    """Volume change, growth rate, response category and probabilistic forecast across timepoints."""
    return analyse_longitudinal(req)


# ---------------------------------------------------------------------------
# PDF reports
# ---------------------------------------------------------------------------

@app.post("/api/report")
async def report(req: ReportRequest) -> Response:
    try:
        pdf, report_id = await run_in_threadpool(render_report, req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="neuroscan-report-{report_id}.pdf"'},
    )
