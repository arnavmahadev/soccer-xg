"""Map image pixels to pitch coordinates via a planar homography.

The user pairs at least four points they can see in the photo with their known
location on the pitch (a corner, a penalty-box corner, the centre spot, ...).
Because the pitch is flat, a single 3x3 homography maps any ground point in the
image to pitch coordinates. We solve it once per calibration and reuse it for
every detection (and, for a fixed camera, every video frame).

Pitch coordinates use the same 120 x 80 grid as the rest of SoccerBoard, with x
running goal-to-goal and y across.
"""

from __future__ import annotations

import cv2
import numpy as np

PITCH_X = (0.0, 120.0)
PITCH_Y = (0.0, 80.0)


class Calibration:
    """A solved image->pitch transform."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    @classmethod
    def from_points(
        cls,
        image_pts: list[tuple[float, float]],
        pitch_pts: list[tuple[float, float]],
    ) -> "Calibration":
        if len(image_pts) != len(pitch_pts) or len(image_pts) < 4:
            raise ValueError("need at least four matched image/pitch point pairs")
        src = np.asarray(image_pts, dtype=np.float64).reshape(-1, 1, 2)
        dst = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 1, 2)
        # RANSAC tolerates a mis-clicked point when five or more pairs are given;
        # with exactly four it degenerates to a plain least-squares solve.
        matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if matrix is None:
            raise ValueError("could not solve a homography from these points")
        return cls(matrix)

    def project(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Image points -> pitch coordinates."""
        if not points:
            return []
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in out]

    def to_list(self) -> list[float]:
        return self.matrix.flatten().tolist()

    @classmethod
    def from_list(cls, values: list[float]) -> "Calibration":
        return cls(np.asarray(values, dtype=np.float64).reshape(3, 3))


def on_pitch(x: float, y: float, margin: float = 5.0) -> bool:
    """True if a projected point lands on (or just off) the pitch.

    A small margin keeps players standing on the touchline; points far outside
    are projection noise (crowd, dugout, bad homography) and get dropped.
    """
    return (
        PITCH_X[0] - margin <= x <= PITCH_X[1] + margin
        and PITCH_Y[0] - margin <= y <= PITCH_Y[1] + margin
    )
