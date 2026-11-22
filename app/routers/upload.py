"""Router for video upload, background processing, and result retrieval."""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from app.config import config
from app.services.video_processor import VideoProcessor
from app.utils.image_utils import encode_image_base64, fit_max_side

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/upload", tags=["upload"])

# ── In-memory job store (sufficient for a BTP demo) ────────────────────────
jobs: dict[str, dict[str, Any]] = {}


def _init_job(job_id: str) -> dict[str, Any]:
    """Create and return a fresh job record."""
    job: dict[str, Any] = {
        "status": "pending",
        "progress": 0.0,
        "error": None,
        "result": None,
    }
    jobs[job_id] = job
    return job


async def _process_video(
    job_id: str,
    video_path: Path,
    request_app: Any,
) -> None:
    """Background task: extract → classify → enhance → reconstruct."""
    job = jobs[job_id]
    job["status"] = "processing"
    job["progress"] = 0.0

    classifier = request_app.state.classifier
    enhancer = request_app.state.enhancer
    video_proc = VideoProcessor()

    try:
        # 1. Extract frames
        frames, original_fps, total_count = await asyncio.to_thread(
            video_proc.extract_frames, video_path, config.VIDEO_SAMPLE_FPS
        )
        if not frames:
            raise RuntimeError("No frames extracted from video.")
        frames = frames[: config.MAX_VIDEO_FRAMES]

        num_frames = len(frames)
        classifications: list[str] = []
        enhanced_frames: list = []
        original_b64: list[str] = []

        for i, frame in enumerate(frames):
            resized = fit_max_side(frame, config.VIDEO_MAX_SIDE)  # keeps aspect ratio

            # 2. Classify
            label = await asyncio.to_thread(classifier.classify, resized)
            classifications.append(label)

            # 3. Enhance
            enhanced = await asyncio.to_thread(enhancer.enhance, resized, label)
            enhanced_frames.append(enhanced)

            # Keep the (aspect-correct) original for the before/after view
            original_b64.append(encode_image_base64(resized, fmt="JPEG"))

            # Update progress  (frames account for 90 %)
            job["progress"] = round((i + 1) / num_frames * 90, 1)

        # 4. Reconstruct enhanced video
        output_filename = f"{job_id}_enhanced.mp4"
        output_path = config.OUTPUT_DIR / output_filename
        await asyncio.to_thread(
            video_proc.reconstruct_video, enhanced_frames, output_path, config.VIDEO_SAMPLE_FPS
        )

        # Build enhanced-frame base64 list
        enhanced_b64: list[str] = [
            encode_image_base64(ef, fmt="JPEG") for ef in enhanced_frames
        ]

        # Store the result FIRST, then flag the job as done – otherwise the
        # frontend can request /results between the two assignments and fail.
        job["result"] = {
            "original_frames": original_b64,
            "enhanced_frames": enhanced_b64,
            "classifications": classifications,
            "enhanced_video_url": f"/outputs/{output_filename}",
            "total_frames": num_frames,
            "fps": config.VIDEO_SAMPLE_FPS,
        }
        job["progress"] = 100.0
        job["status"] = "done"

    except Exception as exc:
        logger.exception("Video processing failed for job %s", job_id)
        job["status"] = "error"
        job["error"] = str(exc)


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/video")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    """Upload a video for background enhancement processing.

    Returns a ``job_id`` that can be polled via the status endpoint.
    """
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a video.")

    job_id = uuid.uuid4().hex
    _init_job(job_id)

    # Save the uploaded video
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        suffix = ".mp4"
    video_filename = f"{job_id}{suffix}"
    video_path = config.UPLOAD_DIR / video_filename
    try:
        contents = await file.read()
        video_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {exc}")

    # Kick off background processing
    background_tasks.add_task(_process_video, job_id, video_path, request.app)

    return {"job_id": job_id, "status": "pending"}


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> dict:
    """Return the current processing status and progress of a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
    }


@router.get("/results/{job_id}")
async def get_job_results(job_id: str) -> dict:
    """Return the full results of a completed job.

    Includes original frames, enhanced frames (base64), per-frame
    classifications, and the URL for the reconstructed enhanced video.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not complete yet. Current status: {job['status']}",
        )

    return {
        "job_id": job_id,
        **job["result"],
    }
