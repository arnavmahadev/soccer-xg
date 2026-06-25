# Phase 5 — Model evaluation

Held-out test set: 540 open-play shots, split by match (no leakage). Our served
model is XGBoost; StatsBomb's own xG is shown as a professional benchmark on the
identical shots.

| model | log loss | Brier | ECE | summed xG | actual goals |
|---|---|---|---|---|---|
| **our xG (XGBoost)** | 0.2631 | 0.0713 | 0.0271 | 55.2 | 57 |
| StatsBomb xG | 0.2444 | 0.0678 | 0.0186 | 51.1 | 57 |

![calibration](calibration.png)

## Reading the calibration curve

- **Well calibrated where it matters most.** The overwhelming majority of shots
  are low-xG (< 0.2), and there both curves hug the diagonal tightly. A predicted
  0.1 really does go in about 10% of the time.
- **Mildly under-confident in the mid-range (~0.3–0.6).** The red curve sits
  slightly *above* the diagonal: shots we rate ~0.45 actually convert a bit more
  often. The model nudges good chances toward the average — a known tendency of
  tree ensembles, amplified by how few high-quality chances exist to learn from.
- **High bins are noisy, not broken.** Past ~0.6 the curve jumps to 1.0 because
  only a handful of test shots land there; those points are small-sample artefacts,
  not systematic error.
- **Aggregate calibration is strong.** Summed xG (55.2) closely tracks actual
  goals (57) — the model gets the big picture right.

**One-sentence summary:** our positions-only model is well calibrated (ECE 0.027,
vs StatsBomb's 0.019), excellent in the dense low-xG region and only mildly
under-confident on rarer high-quality chances.

Reproduce: `python -m xg.eval.calibrate`
