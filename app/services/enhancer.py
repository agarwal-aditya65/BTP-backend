"""Enhancement service – wraps the three pix2pix ``unet_256`` generators."""

from __future__ import annotations

import logging

import torch
from PIL import Image

from app.config import config
from app.models.enhancement_model import UnetGenerator, build_generator
from app.utils.checkpoint import load_weights
from app.utils.image_utils import pil_to_tensor, tensor_to_pil

logger = logging.getLogger(__name__)


class EnhancerService:
    """Loads the per-class generators and exposes ``enhance(image, degradation_type)``."""

    def __init__(self) -> None:
        self.generators: dict[str, UnetGenerator] = {}
        self.weights_loaded: dict[str, bool] = {}

    def load(self) -> None:
        for name, cfg in config.ENHANCERS.items():
            path = config.WEIGHTS_DIR / cfg["weights"]
            gen = build_generator(cfg["norm"])
            self.weights_loaded[name] = False

            if path.exists():
                load_weights(gen, path, config.DEVICE)
                self.weights_loaded[name] = True
                logger.info("Loaded %s enhancer (%s norm, %dpx) from %s", name, cfg["norm"], cfg["size"], path)
            elif config.ALLOW_RANDOM_WEIGHTS:
                logger.warning("%s enhancer weights missing at %s – RANDOM weights (dev mode)", name, path)
            else:
                raise FileNotFoundError(
                    f"{name} enhancer weights not found: {path}. "
                    "Copy your trained *_net_G.pth there or set ALLOW_RANDOM_WEIGHTS=1 for dev."
                )

            gen.to(config.DEVICE).eval()
            self.generators[name] = gen

    @torch.no_grad()
    def enhance(self, image: Image.Image, degradation_type: str) -> Image.Image:
        """Enhance *image* with the generator for *degradation_type*.

        The image is run at that model's training resolution and the result is
        resized back to the input image's size.
        """
        if not self.generators:
            raise RuntimeError("EnhancerService has not been loaded. Call load() first.")
        if degradation_type not in self.generators:
            raise ValueError(
                f"Unknown degradation type '{degradation_type}'. Expected one of {list(self.generators)}."
            )

        size = config.ENHANCERS[degradation_type]["size"]
        x = pil_to_tensor(image, normalize=True, size=size)   # [-1, 1], like pix2pix
        out = tensor_to_pil(self.generators[degradation_type](x))
        if out.size != image.size:
            out = out.resize(image.size, Image.LANCZOS)
        return out
