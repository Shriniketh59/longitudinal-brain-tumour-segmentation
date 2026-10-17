"""Capture README screenshots, an animated GIF and a walkthrough video from the running app.

Requires the frontend on http://localhost:3000 and the backend with weights loaded.

    pip install playwright pymupdf imageio-ffmpeg pillow && python -m playwright install chromium
    python tools/capture_media.py
"""

import io
import shutil
import subprocess
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import imageio_ffmpeg
from PIL import Image
from playwright.sync_api import Page, sync_playwright

BASE = "http://localhost:3000"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
VIEWPORT = {"width": 1440, "height": 900}


def shot(page: Page, name: str, full: bool = False) -> None:
    page.evaluate("window.scrollTo(0, 0)")
    if full:
        # Cards below the fold fade in on scroll; a full-page capture never scrolls, so turn motion off.
        page.emulate_media(reduced_motion="reduce")
    page.wait_for_timeout(900)  # let entrance animations settle
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    if full:
        page.emulate_media(reduced_motion="no-preference")
    print("  saved", name)


def gif_from_frames(frames: list[Image.Image], path: Path, width: int, ms: int) -> None:
    scaled = [f.convert("RGB").resize((width, round(f.height * width / f.width)), Image.LANCZOS) for f in frames]
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in scaled]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    print("  saved", path.name, f"{path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    video_dir = OUT / "_raw_video"
    shutil.rmtree(video_dir, ignore_errors=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=VIEWPORT, record_video_dir=str(video_dir), record_video_size=VIEWPORT, accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(120_000)

        print("classification")
        page.goto(f"{BASE}/analyze")
        page.get_by_label("Load sample: Meningioma").click()
        page.get_by_role("button", name="Run classifier").click()
        page.get_by_text("Cancer risk estimate").wait_for()
        shot(page, "classification")
        shot(page, "classification-full", full=True)

        print("pdf report")
        page.get_by_label("Full name").fill("Sample Patient")
        page.get_by_label("Patient ID / MRN").fill("PT-000123")
        page.get_by_label("Age").fill("54")
        page.get_by_label("Sex").select_option("female")
        page.get_by_label("Scan date").fill("2026-10-15")
        page.get_by_label("Accession / study ID").fill("ACC-2026-0142")
        page.get_by_label("Referring physician").fill("Dr. A. Kumar")
        page.get_by_label("Institution").fill("City Neuro-oncology Centre")
        page.get_by_label("Clinical indication").fill("Progressive headaches and new-onset seizures. Rule out intracranial mass.")
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download PDF report").click()
        pdf_path = OUT / "_report.pdf"
        dl.value.save_as(str(pdf_path))
        doc = fitz.open(pdf_path)
        pix = doc[0].get_pixmap(dpi=110)
        Image.open(io.BytesIO(pix.tobytes("png"))).save(OUT / "pdf-report.png", optimize=True)
        doc.close()
        pdf_path.unlink()
        print("  saved pdf-report")

        print("segmentation")
        page.goto(f"{BASE}/segmentation")
        page.get_by_label("Load sample: Glioma · case A").click()
        page.get_by_role("button", name="Run segmentation").click()
        page.get_by_text("of slice area").wait_for()
        shot(page, "segmentation")

        print("3D transformer")
        page.goto(f"{BASE}/volume")
        page.get_by_role("button", name="3D SwinUNETR · transformer").click()
        page.locator("select").first.select_option("BraTS-GLI-00003-000")
        page.get_by_role("button", name="Load volume").click()
        page.get_by_text("Tumor volume").wait_for(timeout=300_000)
        shot(page, "swin3d")
        shot(page, "swin3d-full", full=True)
        viewer = page.locator(".cine-viewer")
        cine = page.locator(".cine")
        cine.focus()
        for _ in range(18):
            page.keyboard.press("ArrowLeft")
        frames = []
        for _ in range(36):
            frames.append(Image.open(io.BytesIO(viewer.screenshot())))
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(60)
        gif_from_frames(frames, OUT / "swin3d-cine.gif", width=420, ms=110)

        print("longitudinal")
        page.goto(f"{BASE}/longitudinal")
        page.get_by_role("button", name="Add timepoint").click()
        page.get_by_role("button", name="Measure & analyse").click()
        page.get_by_text("Volume over time").wait_for(timeout=400_000)
        shot(page, "longitudinal")
        shot(page, "longitudinal-full", full=True)

        print("performance pages")
        page.goto(f"{BASE}/model/classification")
        page.get_by_text("Confusion matrix").first.wait_for()
        shot(page, "classification-report", full=True)
        page.goto(f"{BASE}/model")
        page.get_by_text("Model performance").first.wait_for()
        shot(page, "performance")

        print("dashboard")
        page.goto(f"{BASE}/")
        page.get_by_text("Recent studies").wait_for()
        shot(page, "dashboard")
        dash = []
        for _ in range(20):
            dash.append(Image.open(io.BytesIO(page.screenshot())))
            page.wait_for_timeout(120)
        gif_from_frames(dash, OUT / "dashboard.gif", width=900, ms=120)
        page.get_by_role("button", name="Switch to light clinical theme").click()
        shot(page, "dashboard-light")
        page.get_by_role("button", name="Switch to dark reading-room theme").click()

        page.goto(f"{BASE}/history")
        shot(page, "history")

        video_path = page.video.path()
        ctx.close()
        browser.close()

    # Walkthrough video: webm from Playwright -> H.264 MP4 (plays everywhere, small).
    mp4 = OUT / "demo.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_path),
         "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "30", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(mp4)],
        check=True,
    )
    shutil.rmtree(video_dir, ignore_errors=True)
    print("  saved demo.mp4", f"{mp4.stat().st_size / 1e6:.1f} MB")

    # Compress PNG screenshots.
    for png in OUT.glob("*.png"):
        Image.open(png).convert("RGB").save(png, optimize=True)
    print("done:", ", ".join(sorted(f.name for f in OUT.iterdir())))


if __name__ == "__main__":
    sys.exit(main())
