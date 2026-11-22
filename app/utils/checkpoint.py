"""Helpers for loading trained weight files robustly."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Keys commonly used when a checkpoint wraps the real state_dict.
_WRAPPER_KEYS = (
    "state_dict",
    "model_state_dict",
    "model",
    "generator",
    "generator_state_dict",
    "G",
    "G_state_dict",
    "netG",
    "net_g",
)


def _extract_state_dict(obj: object) -> dict[str, torch.Tensor]:
    """Return a plain state_dict from whatever ``torch.save`` wrote."""
    if isinstance(obj, nn.Module):  # whole model pickled
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in _WRAPPER_KEYS:
            if key in obj and isinstance(obj[key], dict):
                logger.info("Checkpoint wraps weights under key '%s'", key)
                return obj[key]
        return obj
    raise TypeError(f"Unsupported checkpoint type: {type(obj)!r}")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state and all(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load *path* into *model* (strict), tolerating common checkpoint wrappers.

    Handles: wrapped dicts (``model_state_dict`` / ``generator`` …), a pickled
    full model, and the ``module.`` prefix left behind by ``DataParallel``.

    Raises:
        RuntimeError: with the missing / unexpected keys if the architecture in
            ``app/models`` does not match the one used for training.
    """
    try:
        raw = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Checkpoints that contain more than tensors (optimizer state, a full
        # pickled model, …) need weights_only=False. Only do this for files you
        # trained yourself.
        logger.warning("weights_only load failed for %s – retrying unrestricted", path.name)
        raw = torch.load(path, map_location=device, weights_only=False)

    state = _extract_state_dict(raw)
    state = _strip_prefix(state, "module.")

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        model_keys = list(model.state_dict().keys())[:3]
        file_keys = list(state.keys())[:3]
        raise RuntimeError(
            f"Weights in '{path.name}' do not match the model architecture.\n"
            f"  model expects keys like: {model_keys}\n"
            f"  file contains keys like: {file_keys}\n"
            f"Fix the architecture in app/models/ to match your training code.\n\n{exc}"
        ) from exc
