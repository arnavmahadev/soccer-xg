"""Run detection in a separate process.

torch inference is unstable when launched from the server's asyncio/worker
threads (it deadlocks or crashes under this Python/torch build), but runs
cleanly on a fresh process's main thread. So all heavy work is funnelled through
a single persistent worker process: the server pickles the inputs over, the
worker runs the pipeline on its own main thread and returns plain dicts.

A ``max_workers=1`` pool keeps one warm process that reuses its cached model
across requests.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

_POOL: ProcessPoolExecutor | None = None


def pool() -> ProcessPoolExecutor:
    """The singleton inference process (spawned lazily, kept warm)."""
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=1)
    return _POOL


# --- functions that execute INSIDE the worker process ---------------------
# These must be importable top-level callables so ``spawn`` can pickle them by
# name. They rebuild a Calibration from its 9 flat matrix values.


def _run_frame(image_bytes: bytes, matrix: list[float]) -> dict:
    import cv2

    from .homography import Calibration
    from .pipeline import process_frame

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode the uploaded image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    result = process_frame(rgb, Calibration.from_list(matrix))
    return {"source": {"w": w, "h": h}, **asdict(result)}


def _run_video(path: str, matrix: list[float], every_seconds: float) -> list[dict]:
    from .homography import Calibration
    from .pipeline import process_video

    return process_video(path, Calibration.from_list(matrix), every_seconds=every_seconds)


def _run_warmup() -> bool:
    from .detect import warmup

    warmup()
    return True


# --- thin wrappers the server calls (block on the worker result) ----------
def run_frame(image_bytes: bytes, matrix: list[float]) -> dict:
    return pool().submit(_run_frame, image_bytes, matrix).result()


def run_video(path: str, matrix: list[float], every_seconds: float) -> list[dict]:
    return pool().submit(_run_video, path, matrix, every_seconds).result()


def warm() -> None:
    """Build + warm the model inside the worker process, off the request path."""
    pool().submit(_run_warmup).result()
