"""Assign each player to one of two teams by jersey colour.

For every player box we sample the torso region, drop pixels that look like
grass (green-dominant) and skin, and take the median colour as that player's
shirt. A 2-means clustering over those shirt colours splits the outfield
players into two teams. It is deliberately simple: no per-match training, just
"which of two colours is this shirt closer to". Goalkeepers and referees are
outliers and may land on either side — acceptable for a first pass.
"""

from __future__ import annotations

import numpy as np

from .detect import BALL_CLASS, Detection


def _torso_color(image: np.ndarray, det: Detection) -> np.ndarray | None:
    """Median RGB of a player's torso, or None if nothing usable remains."""
    h, w = image.shape[:2]
    # Upper-middle of the box: skip the head, avoid the legs/shorts.
    bx1, by1 = int(det.x1), int(det.y1)
    bx2, by2 = int(det.x2), int(det.y2)
    bh = by2 - by1
    bw = bx2 - bx1
    if bh < 6 or bw < 4:
        return None
    y_lo = by1 + int(0.20 * bh)
    y_hi = by1 + int(0.55 * bh)
    x_lo = bx1 + int(0.20 * bw)
    x_hi = bx2 - int(0.20 * bw)
    y_lo, y_hi = max(0, y_lo), min(h, y_hi)
    x_lo, x_hi = max(0, x_lo), min(w, x_hi)
    if y_hi - y_lo < 2 or x_hi - x_lo < 2:
        return None

    patch = image[y_lo:y_hi, x_lo:x_hi].reshape(-1, 3).astype(np.float32)
    r, g, b = patch[:, 0], patch[:, 1], patch[:, 2]
    # Grass: green clearly dominant. Skin/very dark/very bright: low saturation
    # extremes. Keep the rest as candidate shirt pixels.
    is_grass = (g > r + 12) & (g > b + 12)
    keep = ~is_grass
    if keep.sum() < 8:
        keep = np.ones(len(patch), dtype=bool)  # fall back to the whole patch
    return np.median(patch[keep], axis=0)


def _kmeans2(points: np.ndarray, iters: int = 20) -> np.ndarray:
    """Tiny 2-means; returns a 0/1 label per row. Deterministic seeding."""
    if len(points) < 2:
        return np.zeros(len(points), dtype=int)
    # Seed with the two most separated points along the dominant colour axis.
    spread = points.std(axis=0)
    axis = int(np.argmax(spread))
    order = np.argsort(points[:, axis])
    centers = np.stack([points[order[0]], points[order[-1]]]).astype(np.float32)

    labels = np.zeros(len(points), dtype=int)
    for _ in range(iters):
        d0 = np.linalg.norm(points - centers[0], axis=1)
        d1 = np.linalg.norm(points - centers[1], axis=1)
        new = (d1 < d0).astype(int)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                centers[k] = points[labels == k].mean(axis=0)
    return labels


def assign_teams(image: np.ndarray, dets: list[Detection]) -> dict[int, int]:
    """Map the index of each *player* detection to a team id (0 or 1).

    Ball detections are ignored. Players whose torso colour can't be sampled are
    omitted from the returned map (caller treats them as team ``None``).
    """
    idx, colors = [], []
    for i, det in enumerate(dets):
        if det.cls == BALL_CLASS:
            continue
        c = _torso_color(image, det)
        if c is not None:
            idx.append(i)
            colors.append(c)

    if not idx:
        return {}
    labels = _kmeans2(np.stack(colors))
    return {i: int(lab) for i, lab in zip(idx, labels)}
