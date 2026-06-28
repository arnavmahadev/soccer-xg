"""FastAPI routes for the photo/video -> pitch feature.

Kept in its own router so the main app can include it conditionally: if the
vision extras (opencv / ultralytics / torch) aren't installed, the app still
boots and these routes report unavailable instead of crashing at import time.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .homography import Calibration
from .worker import run_frame, run_video


def _parse_calibration(calib_json: str) -> Calibration:
    """Build a Calibration from the frontend's list of point pairs.

    Expected JSON: ``[{"image": [x, y], "pitch": [x, y]}, ...]`` with >= 4 pairs.
    """
    try:
        pairs = json.loads(calib_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"calibration is not valid JSON: {exc}") from exc
    try:
        image_pts = [(float(p["image"][0]), float(p["image"][1])) for p in pairs]
        pitch_pts = [(float(p["pitch"][0]), float(p["pitch"][1])) for p in pairs]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(422, f"malformed calibration pairs: {exc}") from exc
    try:
        return Calibration.from_points(image_pts, pitch_pts)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


router = APIRouter(prefix="/vision", tags=["vision"])

# NOTE: the endpoints below are *synchronous* (`def`, not `async def`) so FastAPI
# runs each in its threadpool. The thread then blocks on a separate inference
# *process* (see worker.py) — keeping the event loop and /health responsive while
# detection runs, and isolating torch from the server's threads where it crashes.


@router.get("/health")
def vision_health() -> dict:
    return {"available": True, "detector": "yolov8n"}


@router.post("/photo")
def vision_photo(
    file: UploadFile = File(...),
    calib: str = Form(...),
) -> dict:
    """Extrapolate a single photo onto the pitch."""
    calibration = _parse_calibration(calib)
    try:
        return run_frame(file.file.read(), calibration.to_list())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/video")
def vision_video(
    file: UploadFile = File(...),
    calib: str = Form(...),
    every_seconds: float = Form(0.5),
) -> dict:
    """Sample a video and project every sampled frame into a scrub timeline."""
    calibration = _parse_calibration(calib)
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        try:
            frames = run_video(tmp.name, calibration.to_list(), every_seconds)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    return {"frames": frames, "count": len(frames)}
