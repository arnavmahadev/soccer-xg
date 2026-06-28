"""Player + ball detection with a small, CPU-friendly YOLO model.

The model (yolov8n) is a COCO detector: class 0 is ``person`` and class 32 is
``sports ball``. We keep only those two classes. Weights auto-download on first
use (~6 MB) and the model is cached process-wide so video frames reuse it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

PERSON_CLASS = 0
BALL_CLASS = 32

# Conservative thresholds: players are large and reliable, the ball is tiny and
# easily confused, so it gets a separate (lower) bar applied downstream.
PERSON_CONF = 0.35
BALL_CONF = 0.15


@dataclass
class Detection:
    """An axis-aligned box in image pixel coordinates."""

    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box — where the player meets the ground.

        This is the point we project onto the pitch; a player's feet, not their
        centroid, sit on the playing surface.
        """
        return (0.5 * (self.x1 + self.x2), self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))


WEIGHTS = Path(__file__).resolve().parents[3] / "models" / "weights" / "yolov8n.pt"


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so the FastAPI app still boots if the (heavy) vision extras
    # are not installed in a given deployment. Weights live under models/weights/
    # (gitignored); ultralytics downloads them there on first use.
    import torch
    from ultralytics import YOLO

    # Pin to a single thread. FastAPI runs each request in a worker thread, and a
    # multi-threaded OpenMP forward pass launched off the main thread deadlocks
    # under this torch build; single-threaded inference sidesteps it. yolov8n on
    # CPU is fast enough that the throughput cost is negligible.
    torch.set_num_threads(1)

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS if WEIGHTS.exists() else "yolov8n.pt"
    return YOLO(str(target))


def warmup() -> None:
    """Build the model and run one inference on the main thread.

    Ultralytics/torch initialise their thread pools lazily on first use; doing
    that the first time inside a request (which FastAPI runs in a worker thread)
    can deadlock. Triggering a real forward pass here, at startup on the main
    thread, means later worker-thread inferences only do the forward pass.
    """
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _model().predict(dummy, verbose=False)


def detect(image: np.ndarray) -> list[Detection]:
    """Run detection on a single BGR/RGB image array.

    Returns every person above ``PERSON_CONF`` and every ball above
    ``BALL_CONF``. Coordinates are in the image's own pixel space.
    """
    result = _model().predict(
        image,
        classes=[PERSON_CLASS, BALL_CLASS],
        conf=min(PERSON_CONF, BALL_CONF),
        verbose=False,
    )[0]

    out: list[Detection] = []
    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if cls == PERSON_CLASS and conf < PERSON_CONF:
            continue
        if cls == BALL_CLASS and conf < BALL_CONF:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        out.append(Detection(cls=cls, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))
    return out
