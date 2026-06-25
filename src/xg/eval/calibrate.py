"""Phase 5 evaluation: is our served xG model well calibrated?

Accuracy isn't the point of xG — calibration is. A model that says 0.3 should be
right 30% of the time. This script bins the served XGBoost model's predictions on
the held-out test shots, compares the predicted xG to the actual goal rate, and
plots it against StatsBomb's xG and the perfect-calibration diagonal.

Output: reports/calibration.png (committed) + printed log loss / Brier / ECE.

Run:  python -m xg.eval.calibrate
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: just write the file
import matplotlib.pyplot as plt
import numpy as np

from xg.eval.metrics import ece, reliability_curve, summary
from xg.features.build import build_dataset, test_mask_by_match
from xg.models.baseline import load_model

REPORTS = Path(__file__).resolve().parents[3] / "reports"


def evaluate():
    X, y, groups, sb_xg = build_dataset()
    is_test = test_mask_by_match(groups)
    Xte, yte, sb_te = X[is_test], y[is_test], sb_xg[is_test]

    model = load_model()["model"]
    p = model.predict_proba(Xte)[:, 1]

    ours, bench = summary(yte, p), summary(yte, sb_te)
    ours["ece"], bench["ece"] = ece(yte, p), ece(yte, sb_te)

    print(f"{'':18s}{'log_loss':>9s}{'brier':>8s}{'ece':>8s}{'pred_g':>8s}{'actual_g':>9s}")
    for label, m in [("our xG (XGBoost)", ours), ("StatsBomb xG", bench)]:
        print(f"{label:18s}{m['log_loss']:9.4f}{m['brier']:8.4f}{m['ece']:8.4f}"
              f"{m['pred_goals']:8.1f}{m['actual_goals']:9.0f}")

    _plot(yte, p, sb_te)
    return ours, bench


def _plot(yte, p, sb_te):
    op, oa, _ = reliability_curve(yte, p)
    bp, ba, _ = reliability_curve(yte, sb_te)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.plot(op, oa, "o-", c="crimson", label="our xG (XGBoost)")
    ax.plot(bp, ba, "s-", c="steelblue", alpha=0.7, label="StatsBomb xG")
    ax.set_xlabel("predicted xG (bin mean)")
    ax.set_ylabel("actual goal rate")
    ax.set_title("Calibration on held-out shots")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend()

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "calibration.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    evaluate()
