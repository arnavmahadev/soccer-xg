"""Lightweight serving predictor — the only model code the deployed app imports.

Loads a raw XGBoost Booster (exported by `baseline.export_for_serving`) plus a
tiny JSON of metadata. Imports just xgboost + numpy (no scikit-learn, no joblib,
no torch, and not the training module), which keeps the container image small.

The trained classifier and this predictor produce identical numbers — this only
changes the *packaging* of the model, not the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from xg.data.schema import GameState
from xg.features.build import state_to_features

_MODELS = Path(__file__).resolve().parents[3] / "models"
BOOSTER_PATH = _MODELS / "baseline.ubj"
META_PATH = _MODELS / "serve_meta.json"

_booster: xgb.Booster | None = None
_meta: dict | None = None


def load():
    global _booster, _meta
    if _booster is None:
        if not (BOOSTER_PATH.exists() and META_PATH.exists()):
            raise FileNotFoundError(
                f"No serving model at {BOOSTER_PATH}. Run: python -m xg.models.baseline"
            )
        _booster = xgb.Booster()
        _booster.load_model(str(BOOSTER_PATH))
        _meta = json.loads(META_PATH.read_text())
    return _booster, _meta


def model_info() -> dict:
    _, meta = load()
    return {"name": meta["model"], "n_features": len(meta["features"])}


def predict(state: GameState, shot_type: str = "open_play") -> float:
    booster, meta = load()
    if shot_type == "penalty":
        return float(meta["penalty_xg"])
    x = state_to_features(state).reshape(1, -1)
    dm = xgb.DMatrix(x, feature_names=meta["features"])
    return float(booster.predict(dm)[0])
