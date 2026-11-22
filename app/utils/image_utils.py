"""Image utility helpers for tensor ↔ PIL conversions and base64 encoding."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.config import config


def pil_to_tensor(image: Image.Image, normalize: bool = True, size: int | None = None) -> torch.Tensor:
    """Convert a PIL Image to a float tensor on the configured device.

    Args:
        image: Input PIL Image (RGB).
        normalize: If ``True`` the pixel values are scaled to [-1, 1] which
            is the range expected by the Pix2Pix generators.  When ``False``
            the standard ``ToTensor`` [0, 1] range is used (useful before
            applying ImageNet normalisation in the classifier pipeline).

    Returns:
        A ``(1, 3, H, W)`` tensor on :pyattr:`config.DEVICE`.
    """
    transform_steps: list[transforms.transforms._BasicTransform] = [
        transforms.Resize((size or config.IMAGE_SIZE, size or config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
    if normalize:
        # Scale [0, 1] → [-1, 1]
        transform_steps.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    transform = transforms.Compose(transform_steps)
    tensor: torch.Tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).to(config.DEVICE)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [-1, 1] float tensor back to a PIL Image.

    Args:
        tensor: A ``(1, 3, H, W)`` or ``(3, H, W)`` tensor with values in
            [-1, 1].

    Returns:
        An 8-bit RGB PIL Image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # De-normalise [-1, 1] → [0, 1]
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0.0, 1.0)
    np_image: np.ndarray = tensor.detach().cpu().permute(1, 2, 0).numpy()
    np_image = (np_image * 255).astype(np.uint8)
    return Image.fromarray(np_image)


def encode_image_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 string.

    Args:
        image: The image to encode.
        fmt: Image format (e.g. ``"PNG"``, ``"JPEG"``).

    Returns:
        A base64-encoded string of the image bytes.
    """
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_base64(data: str) -> Image.Image:
    """Decode a base64 string into a PIL Image.

    Args:
        data: Base64-encoded image data (optionally with a
            ``data:image/...;base64,`` prefix).

    Returns:
        An RGB PIL Image.
    """
    # Strip optional data-URI prefix
    if "," in data:
        data = data.split(",", 1)[1]
    image_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def resize_for_model(image: Image.Image) -> Image.Image:
    """Resize a PIL Image to the model input resolution (512×512).

    Args:
        image: Input PIL Image.

    Returns:
        A resized RGB PIL Image.
    """
    return image.convert("RGB").resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE),
        Image.LANCZOS,
    )


def fit_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale *image* so its longest side is at most *max_side* (aspect ratio kept).

    Images that are already small enough are returned unchanged (RGB).
    """
    image = image.convert("RGB")
    longest = max(image.size)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.LANCZOS)
