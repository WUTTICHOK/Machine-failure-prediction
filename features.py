"""Feature definitions shared by train.py and app.py.

Keeping these in one module guarantees that the dashboard applies exactly the
same column names, feature order and engineered features as training. The
saved pipeline references `add_engineered_features`, so this file must be
deployed next to app.py.
"""
import numpy as np
import pandas as pd

# Original AI4I 2020 column name -> clean name used in code
RAW_TO_CLEAN = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

TARGET = "Machine failure"
TARGET_NAMES = {0: "No Failure", 1: "Failure"}
POSITIVE_LABEL = 1  # "Failure" is the positive class for precision/recall/F1

# Columns that must NOT be used as inputs
ID_COLUMNS = ["UDI", "Product ID"]                 # identifiers
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # failure-mode flags reveal the outcome

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]
INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # order expected by the pipeline

ENGINEERED_FEATURES = ["temp_diff_k", "power_w", "wear_torque"]
MODEL_NUMERIC_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES

TYPE_CATEGORIES = ["L", "M", "H"]  # product quality variant: Low / Medium / High

FEATURE_LABELS = {
    "type": "Product quality type",
    "air_temp_k": "Air temperature (K)",
    "process_temp_k": "Process temperature (K)",
    "rot_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add physically motivated features computed only from the inputs.

    - temp_diff_k : process minus air temperature (heat dissipation)
    - power_w     : mechanical power = torque x angular speed (W)
    - wear_torque : tool wear x torque (overstrain indicator, min*Nm)
    """
    X = pd.DataFrame(X).copy()
    X["temp_diff_k"] = X["process_temp_k"] - X["air_temp_k"]
    X["power_w"] = X["torque_nm"] * X["rot_speed_rpm"] * 2 * np.pi / 60
    X["wear_torque"] = X["tool_wear_min"] * X["torque_nm"]
    return X


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns to the clean names used in the code."""
    return df.rename(columns=RAW_TO_CLEAN)
