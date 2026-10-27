"""Prediction, input validation and monitoring logic used by app.py.

Kept separate from the Streamlit UI so it can be unit-tested (tests/).
"""
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_LABELS, INPUT_FEATURES, NUMERIC_FEATURES, TYPE_CATEGORIES, add_engineered_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "machine_failure_model.joblib"
LOG_PATH = ROOT / "logs" / "predictions.csv"

logger = logging.getLogger("machine_failure_app")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Physically plausible limits. Values outside these are rejected as invalid.
# (Values inside these limits but outside the training range only raise a warning.)
HARD_LIMITS = {
    "air_temp_k": (250.0, 400.0),
    "process_temp_k": (250.0, 450.0),
    "rot_speed_rpm": (1.0, 5000.0),
    "torque_nm": (0.0, 150.0),
    "tool_wear_min": (0.0, 400.0),
}

# Monitoring review triggers
MIN_RECORDS_FOR_REVIEW = 30
FAILURE_RATE_MULTIPLIER = 3.0   # predicted failure rate > 3x training rate
OUT_OF_RANGE_SHARE = 0.10       # >10% of inputs outside the training range
MEAN_SHIFT_Z = 1.0              # recent mean differs from training mean by > 1 training std


def load_artifact(path: Path = MODEL_PATH) -> dict:
    artifact = joblib.load(path)
    required = {"pipeline", "threshold", "input_features", "feature_stats"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Model file is missing keys: {missing}")
    if list(artifact["input_features"]) != INPUT_FEATURES:
        raise ValueError("Feature order in model file does not match features.py")
    return artifact


def validate_inputs(inputs: dict, artifact: dict) -> tuple[dict, list, list]:
    """Return (clean_inputs, errors, warnings).

    errors   -> prediction must not be made
    warnings -> prediction is made but flagged as less reliable
    """
    errors, warnings, clean = [], [], {}

    t = inputs.get("type")
    t = str(t).strip().upper() if t is not None and not (isinstance(t, float) and math.isnan(t)) else ""
    if t not in TYPE_CATEGORIES:
        errors.append(f"{FEATURE_LABELS['type']} must be one of {TYPE_CATEGORIES} (got '{inputs.get('type')}').")
    clean["type"] = t

    stats = artifact["feature_stats"]
    for c in NUMERIC_FEATURES:
        label = FEATURE_LABELS[c]
        raw = inputs.get(c)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number (got '{raw}').")
            continue
        if not math.isfinite(v):
            errors.append(f"{label} is missing or not a finite number.")
            continue
        lo, hi = HARD_LIMITS[c]
        if v < lo or v > hi:
            errors.append(f"{label} = {v:g} is outside the valid range {lo:g} to {hi:g}.")
            continue
        if v < stats[c]["min"] or v > stats[c]["max"]:
            warnings.append(f"{label} = {v:g} is outside the training range "
                            f"{stats[c]['min']:g} to {stats[c]['max']:g}; the prediction is an extrapolation.")
        clean[c] = v

    if not errors and clean["process_temp_k"] <= clean["air_temp_k"]:
        warnings.append("Process temperature is not above air temperature, which never occurs in the training data.")
    return clean, errors, warnings


def predict(artifact: dict, clean_inputs: dict) -> dict:
    X = pd.DataFrame([clean_inputs], columns=INPUT_FEATURES)
    proba = float(artifact["pipeline"].predict_proba(X)[0, 1])
    threshold = float(artifact["threshold"])
    label = "Failure" if proba >= threshold else "No Failure"
    eng = add_engineered_features(X).iloc[0]
    return {
        "failure_probability": proba,
        "threshold": threshold,
        "prediction": label,
        "temp_diff_k": float(eng["temp_diff_k"]),
        "power_w": float(eng["power_w"]),
        "wear_torque": float(eng["wear_torque"]),
    }


def predict_batch(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Validate and predict every row; invalid rows get an error message instead."""
    rows = []
    for i, rec in df.iterrows():
        clean, errors, warnings = validate_inputs(rec.to_dict(), artifact)
        out = {"row": i, **{c: rec.get(c) for c in INPUT_FEATURES}}
        if errors:
            out.update(status="REJECTED", prediction="", failure_probability=np.nan, message=" | ".join(errors))
        else:
            res = predict(artifact, clean)
            out.update(status="OK" if not warnings else "OK (warning)", prediction=res["prediction"],
                       failure_probability=res["failure_probability"], message=" | ".join(warnings))
        rows.append(out)
    return pd.DataFrame(rows)


def log_prediction(clean_inputs: dict, result: dict, warnings: list, source: str = "single",
                   path: Path = LOG_PATH) -> dict:
    """Append a non-sensitive record (machine readings + prediction only)."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        **{c: clean_inputs[c] for c in INPUT_FEATURES},
        "failure_probability": round(result["failure_probability"], 6),
        "prediction": result["prediction"],
        "threshold": result["threshold"],
        "out_of_training_range": any("training range" in w for w in warnings),
        "n_warnings": len(warnings),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record]).to_csv(path, mode="a", header=not path.exists(), index=False)
    logger.info("prediction %s", json.dumps(record))  # visible in Streamlit Cloud "Manage app" logs
    return record


def read_log(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarise_log(log: pd.DataFrame, artifact: dict, window: int = 50) -> dict:
    """Monitoring summary for the most recent `window` predictions plus review triggers."""
    if log.empty:
        return {"n_total": 0, "n_window": 0, "triggers": [], "feature_shift": {}}
    recent = log.tail(window)
    n = len(recent)
    pred_rate = float((recent["prediction"] == "Failure").mean())
    oor_share = float(recent["out_of_training_range"].astype(bool).mean())
    train_rate = float(artifact["train_failure_rate"])
    shift = {}
    for c in NUMERIC_FEATURES:
        s = artifact["feature_stats"][c]
        shift[c] = (float(recent[c].mean()) - s["mean"]) / s["std"] if s["std"] > 0 else 0.0

    triggers = []
    if n >= MIN_RECORDS_FOR_REVIEW:
        if pred_rate > FAILURE_RATE_MULTIPLIER * train_rate:
            triggers.append(f"Predicted failure rate {pred_rate:.1%} is more than {FAILURE_RATE_MULTIPLIER:g}x "
                            f"the training failure rate {train_rate:.1%}.")
        if oor_share > OUT_OF_RANGE_SHARE:
            triggers.append(f"{oor_share:.0%} of recent inputs are outside the training range "
                            f"(limit {OUT_OF_RANGE_SHARE:.0%}).")
        for c, z in shift.items():
            if abs(z) > MEAN_SHIFT_Z:
                triggers.append(f"Mean {FEATURE_LABELS[c]} has shifted {z:+.2f} training std from training data.")
    return {
        "n_total": int(len(log)),
        "n_window": n,
        "predicted_failure_rate": pred_rate,
        "train_failure_rate": train_rate,
        "mean_probability": float(recent["failure_probability"].mean()),
        "out_of_range_share": oor_share,
        "feature_shift": shift,
        "triggers": triggers,
        "enough_data": n >= MIN_RECORDS_FOR_REVIEW,
    }
