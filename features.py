"""Feature definitions shared by train.py and app.py.

Keeping these in one module guarantees that the dashboard applies exactly the
same column names, feature order and engineered features as training. The
saved pipeline references `add_engineered_features`, so this file must be
deployed next to app.py.
"""
import numpy as np
import pandas as pd

DATA_FILE = "07_Machine_Failure.xlsx"   # lecturer-provided dataset (sheet "Data")
DATA_SHEET = "Data"

ID_COLUMN = "record_id"                 # identifier - excluded from inputs
TARGET = "failure_next_shift"           # "Failure" / "No_Failure"
POSITIVE_RAW = "Failure"
TARGET_NAMES = {0: "No Failure", 1: "Failure"}
POSITIVE_LABEL = 1  # "Failure" is the positive class for precision/recall/F1

INPUT_FEATURES = [  # order expected by the pipeline
    "air_temperature_c",
    "process_temperature_c",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_mm_s",
    "machine_age_years",
]
# Inputs that may be missing (blank sensor readings in the data); filled by the
# pipeline's median imputer, so the dashboard also accepts them as blank.
IMPUTABLE_FEATURES = ["air_temperature_c", "process_temperature_c"]

ENGINEERED_FEATURES = ["temp_diff_c", "power_w", "wear_torque"]
MODEL_FEATURES = INPUT_FEATURES + ENGINEERED_FEATURES

FEATURE_LABELS = {
    "air_temperature_c": "Air temperature (°C)",
    "process_temperature_c": "Process temperature (°C)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (N·m)",
    "tool_wear_min": "Tool wear (min)",
    "vibration_mm_s": "Vibration (mm/s)",
    "machine_age_years": "Machine age (years)",
}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add physically motivated features computed only from the inputs.

    Runs AFTER imputation inside the pipeline, so no value is missing here.
    - temp_diff_c : process minus air temperature (thermal stress)
    - power_w     : mechanical power = torque x angular speed (W)
    - wear_torque : tool wear x torque (load on a worn tool, min*N*m)
    """
    X = pd.DataFrame(X, columns=INPUT_FEATURES).copy()
    X["temp_diff_c"] = X["process_temperature_c"] - X["air_temperature_c"]
    X["power_w"] = X["torque_nm"] * X["rotational_speed_rpm"] * 2 * np.pi / 60
    X["wear_torque"] = X["tool_wear_min"] * X["torque_nm"]
    return X


def encode_target(series: pd.Series) -> pd.Series:
    """'Failure' -> 1, 'No_Failure' -> 0."""
    return (series.astype(str).str.strip() == POSITIVE_RAW).astype(int)
