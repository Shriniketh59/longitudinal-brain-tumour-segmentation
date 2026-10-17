import io

import nibabel as nib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app import volume
from app.main import app
from app.models.unet import UNetVGG16


def _nifti_bytes(shape=(32, 40, 12), label: bool = False) -> bytes:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, shape).astype(np.uint8) if label else (rng.random(shape) * 1000).astype(np.float32)
    img = nib.Nifti1Image(data, np.diag([1.0, 1.0, 2.0, 1.0]))
    return img.to_bytes()


def test_region_volumes_use_spacing():
    seg = np.zeros((10, 10, 10), np.uint8)
    seg[:5, :5, :4] = 3  # 100 voxels
    vols = volume.region_volumes_cm3(seg, (1.0, 1.0, 2.0))
    assert vols["enhancing_tumor"] == pytest.approx(0.2)
    assert vols["whole_tumor"] == vols["tumor_core"] == vols["enhancing_tumor"]


def test_region_dice_perfect_and_empty():
    seg = np.zeros((8, 8, 8), np.uint8)
    seg[2:4, 2:4, 2:4] = 2
    dice = volume.region_dice(seg, seg)
    assert dice["whole_tumor"] == 1.0
    assert dice["enhancing_tumor"] is None


def test_segment_volume_keeps_shape():
    model = UNetVGG16(pretrained_encoder=False).eval()
    vol8 = (np.random.rand(30, 40, 3) * 255).astype(np.uint8)
    out = volume.segment_volume(model, vol8, torch.device("cpu"))
    assert out.shape == vol8.shape


def test_upload_nifti_with_mask():
    with TestClient(app) as client:
        res = client.post(
            "/api/volume/upload",
            files={
                "file": ("scan.nii", _nifti_bytes(), "application/octet-stream"),
                "mask": ("seg.nii", _nifti_bytes(label=True), "application/octet-stream"),
            },
            data={"step": "1", "predict": "false"},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["shape"] == [32, 40, 12]
    assert body["spacing_mm"] == [1.0, 1.0, 2.0]
    assert len(body["frames"]) == len(body["ground_truth"]["masks"]) == len(body["slice_indices"])


def test_upload_rejects_wrong_extension():
    with TestClient(app) as client:
        res = client.post("/api/volume/upload", files={"file": ("scan.png", b"x", "image/png")})
    assert res.status_code == 400


def test_sample_case_id_validated():
    with TestClient(app) as client:
        assert client.get("/api/volume/samples/not-a-case").status_code == 400


def test_report_pdf():
    with TestClient(app) as client:
        res = client.post("/api/report", json={
            "title": "Unit test <report>",
            "study_id": "T-1",
            "notes": "line one\nline two",
            "sections": [{"heading": "Result", "rows": [["Prediction", "Glioma"]]}],
        })
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")


def test_report_rejects_non_image_data_url():
    with TestClient(app) as client:
        res = client.post("/api/report", json={
            "title": "x", "images": [{"caption": "bad", "data_url": "data:text/html;base64,PGI+"}],
        })
    assert res.status_code == 422


def test_swin3d_predict_labels_shape_and_convention():
    pytest.importorskip("monai")
    pytest.importorskip("einops")
    from app.models.swin3d import build_swin_unetr, predict_labels

    model = build_swin_unetr().eval()
    channels = np.zeros((2, 40, 44, 36), np.float32)
    channels[:, 8:32, 8:36, 6:30] = np.random.default_rng(0).random((2, 24, 28, 24)) + 1
    labels = predict_labels(model, channels, torch.device("cpu"), roi=(64, 64, 64))
    assert labels.shape == (40, 44, 36)
    assert set(np.unique(labels)) <= {0, 1, 2, 3}
    assert labels[:8].max() == 0          # background outside the head box stays 0


def test_swin3d_endpoint_503_without_weights():
    with TestClient(app) as client:
        app.state.swin3d = None
        res = client.get("/api/volume/samples/BraTS-GLI-00000-000?model=swin3d")
    assert res.status_code == 503


def test_upload_second_identical_request_is_cached():
    files = {"file": ("scan.nii", _nifti_bytes(), "application/octet-stream")}
    with TestClient(app) as client:
        first = client.post("/api/volume/upload", files=files, data={"predict": "false"}).json()
        second = client.post("/api/volume/upload", files=files, data={"predict": "false"}).json()
    assert first["cached"] is False and second["cached"] is True
    assert first["frames"] == second["frames"]


def test_unet_fast_mode_runs_and_labels_model():
    from app.models.unet import UNetVGG16

    model = UNetVGG16(pretrained_encoder=False).eval().to(memory_format=torch.channels_last)
    vol = (np.random.default_rng(1).random((40, 44, 6)) * 1000).astype(np.float32)
    out = volume.analyse(vol, (1.0, 1.0, 1.0), model=model, device=torch.device("cpu"), fast=True)
    assert "fast mode" in out["prediction"]["model"]
    assert len(out["prediction"]["masks"]) == len(out["frames"])


def test_report_with_patient_details_and_page_numbers():
    with TestClient(app) as client:
        res = client.post("/api/report", json={
            "title": "Brain MRI slice classification",
            "study_id": "ACC-1",
            "modality": "MRI brain",
            "impression": "Appearance most consistent with glioma.",
            "patient": {"name": "Test <Patient>", "patient_id": "PT-1", "age": 54, "sex": "female",
                        "date_of_birth": "1972-03-14", "scan_date": "2026-10-15",
                        "referring_physician": "Dr. A", "institution": "Test Centre", "clinical_indication": "Headache"},
            "sections": [{"heading": f"Section {i}", "rows": [[f"Row {j}", "value"] for j in range(20)]} for i in range(4)],
        })
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")
    assert b"Page 1 of" in res.content or res.content.count(b"/Type /Page") >= 2


def test_report_rejects_invalid_patient_age():
    with TestClient(app) as client:
        res = client.post("/api/report", json={"title": "x", "patient": {"age": 200}})
    assert res.status_code == 422


def test_video_detect_box_and_area():
    from app.video import _detect

    mask = np.zeros((100, 200), np.uint8)
    mask[20:40, 50:90] = 2       # main region
    mask[90:91, 0:1] = 1         # speck, ignored for the box
    box, area = _detect(mask)
    assert box == {"x": 0.25, "y": 0.2, "w": 0.2, "h": 0.2}
    assert area == pytest.approx(4.005, abs=1e-3)
    assert _detect(np.zeros((50, 50), np.uint8)) == (None, 0.0)


def test_video_analyse_frames_with_random_models(png_bytes_small=None):
    from app import inference
    from app.models.unet import UNetVGG16
    from app.models.vgg16_classifier import build_vgg16_classifier
    from app.video import analyse_frames

    rng = np.random.default_rng(3)
    frames = [inference.PreparedSlice((rng.random((90, 120)) * 255).astype(np.uint8)) for _ in range(3)]
    cls = build_vgg16_classifier(pretrained=False).eval()
    seg = UNetVGG16(pretrained_encoder=False).eval().to(memory_format=torch.channels_last)
    out = analyse_frames(frames, cls, seg, torch.device("cpu"))
    assert len(out) == 3
    for r in out:
        assert {"label", "probabilities", "malignancy", "mask_png_base64", "area_pct", "detected"} <= r.keys()


def test_video_endpoint_limits():
    with TestClient(app) as client:
        app.state.classifier = None
        app.state.segmenter = None
        res = client.post("/api/video/analyze", files=[("files", ("a.png", b"x", "image/png"))])
    assert res.status_code == 503


def test_video_detection_needs_classifier_agreement(monkeypatch):
    from app import video
    from app.inference import PreparedSlice

    class AlwaysNoTumor(torch.nn.Module):
        def forward(self, x):  # logits favouring "no_tumor" (index 2)
            out = torch.zeros(x.shape[0], 4)
            out[:, 2] = 10.0
            return out

    class EverythingTumor(torch.nn.Module):
        def forward(self, x):  # label 2 (edema) everywhere
            out = torch.zeros(x.shape[0], 4, x.shape[2], x.shape[3])
            out[:, 2] = 10.0
            return out

    frames = [PreparedSlice(np.full((64, 64), 120, np.uint8))]
    r = video.analyse_frames(frames, AlwaysNoTumor(), EverythingTumor(), torch.device("cpu"))[0]
    assert r["detected"] is False and r["box"] is None and r["mask_png_base64"] is None
    assert r["suppressed"]
