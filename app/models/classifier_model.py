"""EfficientNet-B0 classifier for 3-class weather degradation.

Your trained checkpoint (``best_weather_model.pth``) holds a *bare* torchvision
EfficientNet-B0 – keys look like ``features.0.0.weight`` and
``classifier.1.weight`` with shape (3, 1280) – so the model must NOT be wrapped
in another module, otherwise the keys get a prefix and loading fails.

Training class order in the checkpoint: ['foggy', 'lowLight', 'rainy']
(index 0, 1, 2) – matches ``config.CLASS_NAMES = ["fog", "low_light", "rain"]``.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def WeatherClassifier(num_classes: int = 3) -> nn.Module:  # noqa: N802 (kept for compatibility)
    """Return a torchvision EfficientNet-B0 whose last layer has *num_classes* outputs."""
    model = models.efficientnet_b0(weights=None)
    in_features: int = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
