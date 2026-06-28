"""Glue: an image (or video) + a calibration -> dots on the 2D pitch.

`process_frame` is the unit of work shared by photos and video. Video simply
samples frames with OpenCV and runs the same routine per frame, producing a
timeline the frontend scrubs through. A single calibration is reused for every
frame, which is correct for a fixed (sideline / tactical) camera; a panning
broadcast camera would need per-frame calibration, noted as future work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .detect import BALL_CLASS, detect
from .homography import Calibration, on_pitch
from .teams import assign_teams


@dataclass
class PitchPlayer:
    team: int | None  # 0, 1, or None when colour couldn't be sampled
    x: float
    y: float
    conf: float


@dataclass
class FrameResult:
    players: list[dict]
    ball: dict | None

    @classmethod
    def empty(cls) -> "FrameResult":
        return cls(players=[], ball=None)


def _round(v: float) -> float:
    return round(v, 1)


def process_frame(image: np.ndarray, calib: Calibration) -> FrameResult:
    """Detect, team-assign and project one image into pitch space."""
    dets = detect(image)
    if not dets:
        return FrameResult.empty()

    teams = assign_teams(image, dets)

    players: list[dict] = []
    ball: dict | None = None
    for i, det in enumerate(dets):
        if det.cls == BALL_CLASS:
            # Project the ball from its centre (it sits on the grass).
            (px, py), = calib.project([det.center])
            if on_pitch(px, py):
                cand = {"x": _round(px), "y": _round(py), "conf": round(det.conf, 2)}
                # Keep only the most confident ball if the model fires twice.
                if ball is None or cand["conf"] > ball["conf"]:
                    ball = cand
            continue

        (px, py), = calib.project([det.foot])
        if not on_pitch(px, py):
            continue
        players.append(
            asdict(
                PitchPlayer(
                    team=teams.get(i),
                    x=_round(px),
                    y=_round(py),
                    conf=round(det.conf, 2),
                )
            )
        )
    return FrameResult(players=players, ball=ball)


def process_video(
    path: str,
    calib: Calibration,
    every_seconds: float = 0.5,
    max_frames: int = 240,
) -> list[dict]:
    """Sample a video and project each sampled frame.

    Returns a list of ``{"t": seconds, "players": [...], "ball": {...}|None}``
    ordered in time. ``every_seconds`` controls the scrub resolution; capped by
    ``max_frames`` so a long clip can't blow up CPU time.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * every_seconds)))

    # Honour the frame cap by widening the step rather than truncating the clip.
    if total and total / step > max_frames:
        step = int(np.ceil(total / max_frames))

    timeline: list[dict] = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            # OpenCV decodes BGR; YOLO + colour sampling are channel-agnostic
            # enough, but convert so team colours read true to the source.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = process_frame(rgb, calib)
            timeline.append({"t": round(idx / fps, 2), **asdict(res)})
        idx += 1
    cap.release()
    return timeline
