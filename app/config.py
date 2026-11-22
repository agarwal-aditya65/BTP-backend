"""Application configuration for the adverse-weather enhancement backend."""

import os
from pathlib import Path

import torch


class Config:
    """Central configuration for the application.

    Holds all paths, model settings, and device configuration used across
    the backend services.
    """

    # ── Base paths ──────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    WEIGHTS_DIR: Path = BASE_DIR / "app" / "weights"

    # ── Model settings ──────────────────────────────────────────────────
    IMAGE_SIZE: int = 512          # working size for frames/videos
    CLASSIFIER_SIZE: int = 224     # classifier input size (CenterCrop) – from Classification/inference.py
    CLASSIFIER_RESIZE: int = 256   # shorter side is resized to this before the crop
    NUM_CLASSES: int = 3

    # !! Must be in the SAME ORDER as the label indices used in training !!
    # torchvision ImageFolder sorts folders alphabetically, so folders named
    # fog/ low_light/ rain/ give index 0,1,2 = fog, low_light, rain.
    CLASS_NAMES: list[str] = ["fog", "low_light", "rain"]

    # Below this softmax confidence the classifier result is flagged as uncertain.
    CONFIDENCE_THRESHOLD: float = 0.6

    # Dev switch: set ALLOW_RANDOM_WEIGHTS=1 to start without trained weights
    # (outputs will be garbage). Default: refuse to start with missing weights.
    ALLOW_RANDOM_WEIGHTS: bool = os.getenv("ALLOW_RANDOM_WEIGHTS", "0") == "1"

    # ImageNet normalisation statistics used by the classifier
    IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

    # ── Weight file names ───────────────────────────────────────────────
    CLASSIFIER_WEIGHTS: str = "classifier.pth"
    # One entry per class. "norm" and "size" MUST match each model's train_opt.txt
    # (--norm and the resolution it was trained at):
    #   rain      : norm=batch,    trained at 512
    #   low_light : norm=batch,    trained at 512
    #   fog       : norm=instance, trained at 256
    ENHANCERS: dict[str, dict] = {
        "rain":      {"weights": "rain_enhancer.pth",     "norm": "batch",    "size": 512},
        "fog":       {"weights": "fog_enhancer.pth",      "norm": "instance", "size": 256},
        "low_light": {"weights": "lowlight_enhancer.pth", "norm": "batch",    "size": 512},
    }

    # ── Video processing ────────────────────────────────────────────────
    VIDEO_SAMPLE_FPS: float = 0.5   # frames per second extracted from uploaded videos (1 frame / 2 s)
    MAX_VIDEO_FRAMES: int = 60      # safety cap (= 2 minutes of video at 0.5 FPS)
    VIDEO_MAX_SIDE: int = 960       # longest side of processed video frames (keeps aspect ratio)
    IMAGE_MAX_SIDE: int = 1600      # longest side of a processed single image

    # ── Device auto-detection ───────────────────────────────────────────
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Server settings ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they do not already exist."""
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


config = Config()
