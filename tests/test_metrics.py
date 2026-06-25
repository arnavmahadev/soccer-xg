"""Metrics behave correctly on inputs with known answers."""

import numpy as np

from xg.eval.metrics import brier, ece, log_loss, reliability_curve


def test_perfect_prediction_has_zero_loss():
    y = np.array([1, 0, 1, 0])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert log_loss(y, p) < 1e-6
    assert brier(y, p) == 0.0


def test_well_calibrated_data_has_low_ece():
    # 1000 shots all predicted 0.3, and exactly 30% are goals -> calibrated.
    rng = np.random.default_rng(0)
    p = np.full(1000, 0.3)
    y = (rng.random(1000) < 0.3).astype(int)
    assert ece(y, p) < 0.05


def test_reliability_curve_shapes_align():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.85, 0.9])
    pred, actual, counts = reliability_curve(y, p, n_bins=10)
    assert len(pred) == len(actual) == len(counts)
    assert counts.sum() == len(y)
