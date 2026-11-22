"""Classification service – wraps the EfficientNet-B0 weather classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.config import config
from app.models.classifier_model import WeatherClassifier
from app.utils.checkpoint import load_weights

logger = logging.getLogger(__name__)


class ClassifierService:
    """Loads the classifier weights and exposes a single ``classify`` method.

    Usage::

        svc = ClassifierService()
        svc.load()
        label = svc.classify(pil_image)  # → "rain" | "fog" | "low_light"
    """

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.weights_loaded: bool = False
        self._transform = transforms.Compose([
            transforms.Resize(config.CLASSIFIER_RESIZE),
            transforms.CenterCrop(config.CLASSIFIER_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])

    # ── lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights from disk and move the model to the configured device."""
        weights_path: Path = config.WEIGHTS_DIR / config.CLASSIFIER_WEIGHTS

        self.model = WeatherClassifier(num_classes=config.NUM_CLASSES)

        if weights_path.exists():
            load_weights(self.model, weights_path, config.DEVICE)
            self.weights_loaded = True
            logger.info("Classifier weights loaded from %s", weights_path)
        elif config.ALLOW_RANDOM_WEIGHTS:
            logger.warning("Classifier weights missing at %s – RANDOM weights (dev mode)", weights_path)
        else:
            raise FileNotFoundError(
                f"Classifier weights not found: {weights_path}. "
                "Copy your trained file there or set ALLOW_RANDOM_WEIGHTS=1 for dev."
            )

        self.model.to(config.DEVICE)
        self.model.eval()

    # ── inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def classify_with_confidence(self, image: Image.Image) -> tuple[str, float]:
        """Return ``(label, softmax_confidence)`` for a PIL image."""
        if self.model is None:
            raise RuntimeError("ClassifierService has not been loaded. Call load() first.")

        tensor: torch.Tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(config.DEVICE)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        idx = int(probs.argmax().item())
        return config.CLASS_NAMES[idx], float(probs[idx].item())

    def classify(self, image: Image.Image) -> str:
        """Return only the predicted class name: ``"rain"``, ``"fog"`` or ``"low_light"``."""
        return self.classify_with_confidence(image)[0]
