"""WebSocket router for real-time frame-by-frame enhancement."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import config
from app.utils.image_utils import decode_image_base64, encode_image_base64, fit_max_side

logger = logging.getLogger(__name__)
router = APIRouter(tags=["realtime"])


@router.websocket("/ws/realtime")
async def realtime_enhance(websocket: WebSocket) -> None:
    """Accept a WebSocket connection for real-time enhancement.

    Protocol (JSON over text frames):

    **Client → Server** ::

        { "frame": "<base64-encoded image>" }

    **Server → Client** ::

        {
            "enhanced_frame": "<base64-encoded enhanced image>",
            "classification": "rain" | "fog" | "low_light"
        }

    The connection stays open until the client disconnects or an error
    occurs.
    """
    await websocket.accept()
    logger.info("WebSocket client connected for real-time enhancement")

    classifier = websocket.app.state.classifier
    enhancer = websocket.app.state.enhancer

    try:
        while True:
            # Receive a text frame
            raw = await websocket.receive_text()

            try:
                data = json.loads(raw)
                frame_b64: str = data.get("frame", "")
                if not frame_b64:
                    await websocket.send_json({"error": "Missing 'frame' field."})
                    continue
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON."})
                continue

            try:
                # Decode → resize → classify → enhance
                image = decode_image_base64(frame_b64)
                image = fit_max_side(image, config.VIDEO_MAX_SIDE)

                classification: str = await asyncio.to_thread(classifier.classify, image)
                enhanced = await asyncio.to_thread(enhancer.enhance, image, classification)

                enhanced_b64 = encode_image_base64(enhanced, fmt="JPEG")

                await websocket.send_json({
                    "enhanced_frame": enhanced_b64,
                    "classification": classification,
                })

            except Exception as exc:
                logger.error("Real-time processing error: %s", exc)
                await websocket.send_json({"error": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket unexpected error: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
