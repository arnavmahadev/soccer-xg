"""Scoring metrics for probabilistic predictions.

For xG we care about *probability quality*, not classification accuracy — a good
model says "0.1" for a chance that goes in 10% of the time. The two standard
measures:

- log loss: punishes confident wrong predictions hard (the primary xG metric).
- Brier score: mean squared error of the probability (0 = perfect, lower better).

Plain numpy so the numbers are inspectable and match the StatsBomb benchmark we
computed in Phase 1 (log loss 0.279, Brier 0.081).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-15


def log_loss(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y_true, p_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability_curve(y_true, p_pred, n_bins: int = 10):
    """Bin predictions into [0,1] slices and return, per non-empty bin, the mean
    predicted probability, the actual goal rate, and the count. A perfectly
    calibrated model has mean-predicted == actual-rate in every bin."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    pred_mean, actual_rate, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        pred_mean.append(p[mask].mean())
        actual_rate.append(y[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(pred_mean), np.array(actual_rate), np.array(counts)


def ece(y_true, p_pred, n_bins: int = 10) -> float:
    """Expected Calibration Error: count-weighted mean gap between predicted
    probability and actual rate across bins. 0 = perfectly calibrated."""
    pred_mean, actual_rate, counts = reliability_curve(y_true, p_pred, n_bins)
    if counts.sum() == 0:
        return 0.0
    return float(np.sum(counts * np.abs(pred_mean - actual_rate)) / counts.sum())


def summary(y_true, p_pred) -> dict[str, float]:
    """All the headline numbers for one set of predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    return {
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "pred_goals": float(p.sum()),   # summed xG ...
        "actual_goals": float(y.sum()), # ... vs reality (aggregate calibration)
    }
