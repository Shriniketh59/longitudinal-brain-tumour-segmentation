import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    classifier_weights: Path = Path(
        os.getenv("CLASSIFIER_WEIGHTS", BACKEND_DIR / "weights" / "vgg16_classifier.pt")
    )
    unet_weights: Path = Path(
        os.getenv("UNET_WEIGHTS", BACKEND_DIR / "weights" / "unet_vgg16.pt")
    )
    swin_weights: Path = Path(
        os.getenv("SWIN_WEIGHTS", BACKEND_DIR / "weights" / "swin_unetr_brats.pt")
    )
    device: str = os.getenv("DEVICE", "auto")
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _split_csv(os.getenv("CORS_ORIGINS", "http://localhost:3000"))
    )
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    max_volume_bytes: int = int(os.getenv("MAX_VOLUME_MB", "200")) * 1024 * 1024
    brats_dir: Path = Path(os.getenv("BRATS_DIR", BACKEND_DIR / "data" / "brats2023"))
    cache_dir: Path = Path(os.getenv("CACHE_DIR", BACKEND_DIR / "data" / "cache"))
    # Optional "64,64,64"-style override for the 3D sliding-window size.
    swin_roi: tuple[int, int, int] | None = field(
        default_factory=lambda: tuple(int(v) for v in os.getenv("SWIN_ROI").split(",")) if os.getenv("SWIN_ROI") else None
    )


settings = Settings()
