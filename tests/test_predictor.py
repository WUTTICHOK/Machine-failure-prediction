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

TYPICAL = {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0,
           "rot_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}


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


def test_lowercase_type_is_accepted(artifact):
    _, errors, _ = validate_inputs({**TYPICAL, "type": " l "}, artifact)
    assert errors == []


def test_high_strain_case_more_risky_than_typical(artifact):
    risky = {**TYPICAL, "type": "L", "rot_speed_rpm": 1200, "torque_nm": 70, "tool_wear_min": 240}
    p_typ = predict(artifact, validate_inputs(TYPICAL, artifact)[0])["failure_probability"]
    p_risk = predict(artifact, validate_inputs(risky, artifact)[0])["failure_probability"]
    assert p_risk > p_typ


# ---------------------------------------------------------------- boundary
def test_training_min_and_max_are_valid_without_warning(artifact):
    s = artifact["feature_stats"]
    for key in ("min", "max"):
        inp = {"type": "L", **{c: s[c][key] for c in s}}
        _, errors, warnings = validate_inputs(inp, artifact)
        assert errors == []
        assert not any("training range" in w for w in warnings)


def test_outside_training_range_gives_warning_not_error(artifact):
    _, errors, warnings = validate_inputs({**TYPICAL, "rot_speed_rpm": 3500}, artifact)
    assert errors == []
    assert any("training range" in w for w in warnings)


def test_process_not_above_air_temperature_warns(artifact):
    _, errors, warnings = validate_inputs({**TYPICAL, "process_temp_k": 300.0}, artifact)
    assert errors == [] and any("Process temperature" in w for w in warnings)


# ---------------------------------------------------------------- invalid
@pytest.mark.parametrize("bad", [
    {"type": "X"},
    {"type": None},
    {"torque_nm": -5},
    {"rot_speed_rpm": 0},
    {"tool_wear_min": 1000},
    {"air_temp_k": "abc"},
    {"air_temp_k": None},
    {"process_temp_k": np.nan},
    {"torque_nm": float("inf")},
])
def test_invalid_inputs_are_rejected(artifact, bad):
    _, errors, _ = validate_inputs({**TYPICAL, **bad}, artifact)
    assert errors, f"expected rejection for {bad}"


def test_batch_rejects_bad_rows_without_crashing(artifact):
    df = pd.DataFrame([TYPICAL, {**TYPICAL, "type": "X"}, {**TYPICAL, "torque_nm": -1}])
    out = predict_batch(artifact, df)
    assert list(out["status"]) == ["OK", "REJECTED", "REJECTED"]


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
    risky = {**TYPICAL, "type": "L", "rot_speed_rpm": 1200, "torque_nm": 75, "tool_wear_min": 240}
    for _ in range(40):
        clean, _, w = validate_inputs(risky, artifact)
        log_prediction(clean, predict(artifact, clean), w, path=log_file)
    summ = summarise_log(pd.read_csv(log_file), artifact)
    assert summ["n_total"] == 40
    assert summ["triggers"], "expected a review trigger for shifted inputs"


def test_monitoring_handles_empty_log(artifact):
    assert summarise_log(pd.DataFrame(), artifact)["n_total"] == 0
