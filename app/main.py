"""FastAPI application entry point for the adverse-weather enhancement backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import config
from app.routers import enhance, realtime, upload
from app.services.classifier import ClassifierService
from app.services.enhancer import EnhancerService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ───────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load ML models into ``app.state`` on startup and clean up on shutdown."""
    config.ensure_dirs()

    # Classifier
    logger.info("Loading classifier model …")
    classifier = ClassifierService()
    classifier.load()
    app.state.classifier = classifier

    # Enhancers
    logger.info("Loading enhancer models …")
    enhancer = EnhancerService()
    enhancer.load()
    app.state.enhancer = enhancer

    logger.info("All models loaded on %s – server ready", config.DEVICE)

    yield  # ← app runs here

    logger.info("Shutting down – releasing resources")


# ── Application factory ────────────────────────────────────────────────────


app = FastAPI(
    title="Adverse-Weather Image/Video Enhancement API",
    description=(
        "Classifies weather degradation (rain / fog / low-light) using EfficientNet-B0 "
        "and enhances images via per-class Pix2Pix U-Net generators."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow everything during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(enhance.router)
app.include_router(realtime.router)

# ── Static file serving for uploads & outputs ──────────────────────────────
app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")


@app.get("/", tags=["health"])
async def root() -> dict:
    """Health-check / welcome endpoint."""
    return {
        "message": "Adverse-Weather Enhancement API is running",
        "device": str(config.DEVICE),
    }


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Report whether each trained weight file was actually loaded."""
    clf = app.state.classifier
    enh = app.state.enhancer
    return {
        "device": str(config.DEVICE),
        "class_order": config.CLASS_NAMES,
        "classifier_weights_loaded": clf.weights_loaded,
        "enhancer_weights_loaded": enh.weights_loaded,
    }
