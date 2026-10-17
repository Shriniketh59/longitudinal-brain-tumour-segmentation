import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import inference
from app.models.unet import SEG_CLASSES, UNetVGG16
from app.models.vgg16_classifier import CLASS_NAMES, build_vgg16_classifier


@pytest.fixture(scope="module")
def png_bytes() -> bytes:
    arr = (np.random.rand(180, 200) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_vgg16_head_shape():
    model = build_vgg16_classifier(pretrained=False).eval()
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, len(CLASS_NAMES))


def test_unet_output_shape():
    model = UNetVGG16(pretrained_encoder=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out.shape == (1, len(SEG_CLASSES), 256, 256)


def test_classify_probabilities_sum_to_one(png_bytes):
    model = build_vgg16_classifier(pretrained=False).eval()
    scan = inference.decode_slice(png_bytes)
    result = inference.classify(model, scan, torch.device("cpu"))
    assert result["label"] in CLASS_NAMES
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4


def test_segment_counts_match_original_size(png_bytes):
    model = UNetVGG16(pretrained_encoder=False).eval()
    scan = inference.decode_slice(png_bytes)
    result = inference.segment(model, scan, torch.device("cpu"))
    assert result["image_size"] == [200, 180]
    assert sum(result["pixel_counts"].values()) == 200 * 180


def test_api_503_without_weights(monkeypatch, png_bytes):
    from app.main import app

    with TestClient(app) as client:
        app.state.classifier = None
        app.state.segmenter = None
        res = client.post("/api/classify", files={"file": ("slice.png", png_bytes, "image/png")})
        assert res.status_code == 503


def test_api_rejects_non_image():
    from app.main import app

    with TestClient(app) as client:
        res = client.post("/api/classify", files={"file": ("x.png", b"not an image", "image/png")})
        assert res.status_code in (400, 503)


def test_malignancy_groups_sum_and_assessment():
    from app.malignancy import assess

    out = assess({"glioma": 0.9, "meningioma": 0.05, "no_tumor": 0.02, "pituitary": 0.03})
    assert out["assessment"] == "likely_cancerous"
    assert out["confidence"] == "high"
    assert abs(out["malignant"] + out["benign"] + out["no_tumor"] - 1.0) < 1e-6

    out = assess({"glioma": 0.1, "meningioma": 0.45, "no_tumor": 0.05, "pituitary": 0.4})
    assert out["assessment"] == "likely_non_cancerous"
    assert out["benign"] == 0.85
    assert 0 < out["uncertainty"] < 1


def test_gradcam_overlay_matches_input(png_bytes):
    model = build_vgg16_classifier(pretrained=False).eval()
    scan = inference.decode_slice(png_bytes)
    result = inference.classify(model, scan, torch.device("cpu"))
    cam = result["gradcam"]
    assert cam is not None and cam["target_class"] == result["label"]
    img = Image.open(io.BytesIO(__import__("base64").b64decode(cam["overlay_png_base64"])))
    assert img.size == (200, 180)
    assert 0 <= cam["peak"]["x"] <= 1 and 0 <= cam["peak"]["y"] <= 1
    assert 0 <= cam["focus_in_brain"] <= 1


def test_focus_in_brain_detects_background_focus():
    import numpy as np
    from app.gradcam import focus_in_brain

    gray = np.zeros((100, 100), np.uint8)
    gray[30:70, 30:70] = 200          # "head" in the centre
    on_head = np.zeros((10, 10), np.float32); on_head[5, 5] = 1.0
    on_edge = np.zeros((10, 10), np.float32); on_edge[9, 0] = 1.0
    assert focus_in_brain(gray, on_head) > 0.9
    assert focus_in_brain(gray, on_edge) < 0.1


def test_crop_to_head_removes_border_artifacts():
    import numpy as np
    from app.preprocess import crop_to_head

    img = np.zeros((200, 200), np.uint8)
    yy, xx = np.ogrid[:200, :200]
    img[(yy - 90) ** 2 + (xx - 100) ** 2 < 60 ** 2] = 150   # head
    img[195:, :] = 255                                      # bright border line at the bottom
    out = crop_to_head(img)
    assert out.shape[0] == out.shape[1] < 200
    assert out[-2:, :].max() == 0                            # border line gone


def test_classify_with_crop_runs(png_bytes):
    model = build_vgg16_classifier(pretrained=False).eval()
    scan = inference.decode_slice(png_bytes)
    result = inference.classify(model, scan, torch.device("cpu"), crop=True)
    assert result["preprocess"] == "crop_head"
