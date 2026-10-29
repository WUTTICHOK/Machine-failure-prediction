"""Automated tests for validation, prediction and monitoring.

Run after train.py:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import INPUT_FEATURES  # noqa: E402
from predictor import (  # noqa: E402
    MODEL_PATH,
    load_artifact,
    log_prediction,
    predict,
    predict_batch,
    summarise_log,
    validate_inputs,
)

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run train.py first")

TYPICAL = {"air_temperature_c": 28.0, "process_temperature_c": 72.5, "rotational_speed_rpm": 1700,
           "torque_nm": 50.0, "tool_wear_min": 120, "vibration_mm_s": 2.4, "machine_age_years": 7.5}
RISKY = {**TYPICAL, "torque_nm": 82.0, "tool_wear_min": 245, "vibration_mm_s": 5.5, "machine_age_years": 14.0}


@pytest.fixture(scope="module")
def artifact():
    return load_artifact()


# ---------------------------------------------------------------- normal
def test_typical_input_is_valid_and_predicts(artifact):
    clean, errors, warnings = validate_inputs(TYPICAL, artifact)
    assert errors == [] and warnings == []
    res = predict(artifact, clean)
    assert 0.0 <= res["failure_probability"] <= 1.0
    assert res["prediction"] in {"Failure", "No Failure"}


def test_numeric_strings_are_accepted(artifact):
    _, errors, _ = validate_inputs({k: str(v) for k, v in TYPICAL.items()}, artifact)
    assert errors == []


def test_high_wear_vibration_case_more_risky_than_typical(artifact):
    p_typ = predict(artifact, validate_inputs(TYPICAL, artifact)[0])["failure_probability"]
    p_risk = predict(artifact, validate_inputs(RISKY, artifact)[0])["failure_probability"]
    assert p_risk > p_typ


# ---------------------------------------------------------------- missing values (imputation)
@pytest.mark.parametrize("col", ["air_temperature_c", "process_temperature_c"])
@pytest.mark.parametrize("blank", [None, np.nan, ""])
def test_blank_temperature_is_imputed_with_warning(artifact, col, blank):
    clean, errors, warnings = validate_inputs({**TYPICAL, col: blank}, artifact)
    assert errors == []
    assert any("is missing" in w for w in warnings)
    res = predict(artifact, clean)
    assert 0.0 <= res["failure_probability"] <= 1.0


@pytest.mark.parametrize("col", ["rotational_speed_rpm", "torque_nm", "tool_wear_min",
                                 "vibration_mm_s", "machine_age_years"])
def test_blank_non_temperature_input_is_rejected(artifact, col):
    _, errors, _ = validate_inputs({**TYPICAL, col: None}, artifact)
    assert errors


# ---------------------------------------------------------------- boundary
def test_training_min_and_max_are_valid_without_range_warning(artifact):
    s = artifact["feature_stats"]
    for key in ("min", "max"):
        inp = {c: s[c][key] for c in INPUT_FEATURES}
        _, errors, warnings = validate_inputs(inp, artifact)
        assert errors == []
        assert not any("training range" in w for w in warnings)


def test_outside_training_range_gives_warning_not_error(artifact):
    _, errors, warnings = validate_inputs({**TYPICAL, "rotational_speed_rpm": 3500}, artifact)
    assert errors == []
    assert any("training range" in w for w in warnings)


def test_process_not_above_air_temperature_warns(artifact):
    _, errors, warnings = validate_inputs({**TYPICAL, "process_temperature_c": 28.0}, artifact)
    assert errors == [] and any("Process temperature" in w for w in warnings)


# ---------------------------------------------------------------- invalid
@pytest.mark.parametrize("bad", [
    {"torque_nm": -5},
    {"rotational_speed_rpm": 0},
    {"tool_wear_min": 1000},
    {"vibration_mm_s": "abc"},
    {"machine_age_years": -1},
    {"air_temperature_c": 500},
    {"torque_nm": float("inf")},
])
def test_invalid_inputs_are_rejected(artifact, bad):
    _, errors, _ = validate_inputs({**TYPICAL, **bad}, artifact)
    assert errors, f"expected rejection for {bad}"


def test_batch_rejects_bad_rows_without_crashing(artifact):
    df = pd.DataFrame([TYPICAL, {**TYPICAL, "torque_nm": -1}, {**TYPICAL, "tool_wear_min": None},
                       {**TYPICAL, "air_temperature_c": None}])
    out = predict_batch(artifact, df)
    assert list(out["status"]) == ["OK", "REJECTED", "REJECTED", "OK (warning)"]


# ---------------------------------------------------------------- agreement with training script
def test_reference_predictions_match_training_script(artifact):
    ref = pd.read_csv(ROOT / "reports" / "reference_predictions.csv")
    for _, r in ref.iterrows():
        clean, errors, _ = validate_inputs(r.to_dict(), artifact)
        assert errors == []
        res = predict(artifact, clean)
        assert res["failure_probability"] == pytest.approx(r["notebook_proba"], abs=1e-9)
        assert res["prediction"] == r["notebook_prediction"]


# ---------------------------------------------------------------- monitoring
def test_monitoring_trigger_fires_on_drift(artifact, tmp_path):
    log_file = tmp_path / "log.csv"
    for _ in range(40):
        clean, _, w = validate_inputs(RISKY, artifact)
        log_prediction(clean, predict(artifact, clean), w, path=log_file)
    summ = summarise_log(pd.read_csv(log_file), artifact)
    assert summ["n_total"] == 40
    assert summ["triggers"], "expected a review trigger for shifted inputs"


def test_monitoring_trigger_fires_on_missing_sensor(artifact, tmp_path):
    log_file = tmp_path / "log.csv"
    for _ in range(40):
        clean, _, w = validate_inputs({**TYPICAL, "air_temperature_c": None}, artifact)
        log_prediction(clean, predict(artifact, clean), w, path=log_file)
    summ = summarise_log(pd.read_csv(log_file), artifact)
    assert any("missing temperature" in t for t in summ["triggers"])


def test_monitoring_handles_empty_log(artifact):
    assert summarise_log(pd.DataFrame(), artifact)["n_total"] == 0
