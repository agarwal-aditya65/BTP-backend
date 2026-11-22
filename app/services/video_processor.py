"""Video processing utilities – frame extraction & reconstruction via OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Stateless helper for video I/O operations.

    All methods are regular (non-async) and should be called via
    ``asyncio.to_thread`` from async code to keep the event loop responsive.
    """

    # ── frame extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_frames(
        video_path: str | Path,
        fps: float | None = None,
    ) -> tuple[list[Image.Image], float, int]:
        """Extract frames from a video file.

        Args:
            video_path: Path to the source video.
            fps: If provided, extract at this frame rate instead of the
                video's native rate.  ``None`` extracts every frame.

        Returns:
            A tuple of ``(frames, original_fps, total_frame_count)`` where
            *frames* is a list of RGB PIL Images.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If OpenCV cannot open the video.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        original_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute frame interval if a target fps is specified
        interval: int = 1
        if fps is not None and fps > 0:
            interval = max(1, round(original_fps / fps))

        frames: list[Image.Image] = []
        frame_idx: int = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                frame_idx += 1
        finally:
            cap.release()

        logger.info(
            "Extracted %d frames from %s (original fps=%.2f, total=%d)",
            len(frames),
            video_path.name,
            original_fps,
            total_frames,
        )
        return frames, original_fps, total_frames

    # ── video reconstruction ────────────────────────────────────────────

    @staticmethod
    def reconstruct_video(
        frames: list[Image.Image],
        output_path: str | Path,
        fps: float,
    ) -> Path:
        """Create an MP4 video from a list of PIL Images.

        Args:
            frames: Ordered list of RGB PIL Images to write.
            output_path: Destination file path (should end in ``.mp4``).
            fps: Frame rate of the output video.

        Returns:
            The resolved output path.

        Raises:
            ValueError: If *frames* is empty.
        """
        if not frames:
            raise ValueError("Cannot reconstruct a video from an empty frame list.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        first_np: np.ndarray = np.array(frames[0])
        height, width = first_np.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            for frame in frames:
                bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

        logger.info("Reconstructed video at %s (%d frames, %.1f fps)", output_path, len(frames), fps)
        return output_path

    # ── video info ──────────────────────────────────────────────────────

    @staticmethod
    def get_video_info(video_path: str | Path) -> dict:
        """Return metadata about a video file.

        Args:
            video_path: Path to the video.

        Returns:
            A dict with keys ``fps``, ``frame_count``, ``duration``
            (seconds), and ``resolution`` (``[width, height]``).

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If OpenCV cannot open the video.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        try:
            fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration: float = frame_count / fps if fps > 0 else 0.0
        finally:
            cap.release()

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": round(duration, 2),
            "resolution": [width, height],
        }
