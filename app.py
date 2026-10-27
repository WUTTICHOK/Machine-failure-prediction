"""Streamlit dashboard - Case 7 Machine failure prediction.

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURE_LABELS, INPUT_FEATURES, NUMERIC_FEATURES, TYPE_CATEGORIES
from predictor import (
    HARD_LIMITS,
    LOG_PATH,
    MODEL_PATH,
    load_artifact,
    log_prediction,
    predict,
    predict_batch,
    read_log,
    summarise_log,
    validate_inputs,
)

ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / "reports" / "reference_predictions.csv"

st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️", layout="wide")


@st.cache_resource
def get_artifact():
    return load_artifact(MODEL_PATH)


try:
    artifact = get_artifact()
except FileNotFoundError:
    st.error("Model file not found. Run `python train.py` first to create models/machine_failure_model.joblib.")
    st.stop()
except Exception as exc:  # e.g. scikit-learn version mismatch
    st.error(f"Could not load the model: {exc}")
    st.stop()

stats = artifact["feature_stats"]
threshold = artifact["threshold"]

st.title("⚙️ Machine Failure Prediction")
st.markdown(
    "Predicts whether a milling machine will **fail** under the current operating conditions, "
    "using a model trained on the AI4I 2020 Predictive Maintenance dataset. "
    "Intended for maintenance staff as an early-warning aid, not a replacement for inspection."
)

with st.sidebar:
    st.header("Model")
    st.write(f"**Algorithm:** {artifact['model_name']}")
    st.write(f"**Decision threshold:** {threshold:.2f}")
    tm = artifact.get("test_metrics", {})
    if tm:
        st.write(f"**Test recall (Failure):** {tm['recall']:.2%}")
        st.write(f"**Test precision (Failure):** {tm['precision']:.2%}")
        st.write(f"**Test F1 (Failure):** {tm['f1']:.3f}")
    st.caption(f"Trained {artifact.get('trained_at', '?')} | scikit-learn {artifact.get('sklearn_version', '?')}")
    st.header("How to use")
    st.markdown(
        "1. Enter the machine readings in **Predict**.\n"
        "2. Press **Predict failure**.\n"
        "3. Values outside the training range are allowed but flagged.\n"
        "4. Check **Monitoring** for logged predictions and review alerts."
    )

tab_pred, tab_batch, tab_check, tab_mon, tab_about = st.tabs(
    ["🔮 Predict", "📄 Batch / invalid-input test", "✅ Notebook agreement", "📈 Monitoring", "ℹ️ About"])

# --------------------------------------------------------------------------- Predict
with tab_pred:
    st.subheader("Enter operating conditions")
    defaults = {"air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 1500.0,
                "torque_nm": 40.0, "tool_wear_min": 100.0}
    steps = {"air_temp_k": 0.1, "process_temp_k": 0.1, "rot_speed_rpm": 10.0, "torque_nm": 0.5, "tool_wear_min": 1.0}
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        inputs = {"type": c1.selectbox(FEATURE_LABELS["type"], TYPE_CATEGORIES,
                                       help="L = low (50% of products), M = medium (30%), H = high quality (20%)")}
        cols = [c1, c2, c2, c3, c3]
        for col, c in zip(cols, NUMERIC_FEATURES):
            lo, hi = HARD_LIMITS[c]
            inputs[c] = col.number_input(
                FEATURE_LABELS[c], min_value=lo, max_value=hi, value=defaults[c], step=steps[c],
                help=f"Training range: {stats[c]['min']:g} to {stats[c]['max']:g}. Valid range: {lo:g} to {hi:g}.")
        st.caption(f"Air ≈ {inputs['air_temp_k'] - 273.15:.1f} °C, process ≈ {inputs['process_temp_k'] - 273.15:.1f} °C")
        submitted = st.form_submit_button("Predict failure", type="primary")

    if submitted:
        clean, errors, warnings = validate_inputs(inputs, artifact)
        if errors:
            for e in errors:
                st.error(e)
        else:
            res = predict(artifact, clean)
            log_prediction(clean, res, warnings)
            for w in warnings:
                st.warning(w)
            p = res["failure_probability"]
            if res["prediction"] == "Failure":
                st.error(f"### Prediction: **FAILURE** likely\nEstimated failure probability **{p:.1%}** "
                         f"(threshold {res['threshold']:.0%}). Recommend inspection before continuing.")
            else:
                st.success(f"### Prediction: **NO FAILURE**\nEstimated failure probability **{p:.1%}** "
                           f"(threshold {res['threshold']:.0%}).")
            st.progress(min(max(p, 0.0), 1.0))
            m1, m2, m3 = st.columns(3)
            m1.metric("Temperature difference", f"{res['temp_diff_k']:.1f} K")
            m2.metric("Mechanical power", f"{res['power_w']:,.0f} W")
            m3.metric("Tool wear × torque", f"{res['wear_torque']:,.0f} min·Nm")
            st.caption("The probability is a model score calibrated on historical data; it is not a guarantee. "
                       "Logged predictions cannot confirm accuracy until the true outcome is recorded.")

# --------------------------------------------------------------------------- Batch
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.markdown(f"Upload a CSV with columns `{', '.join(INPUT_FEATURES)}`. Each row is validated; "
                "invalid rows are **rejected with a reason** instead of crashing the app.")
    example = pd.DataFrame([
        {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 100},
        {"type": "L", "air_temp_k": 302.0, "process_temp_k": 311.0, "rot_speed_rpm": 1250, "torque_nm": 70, "tool_wear_min": 220},
        {"type": "X", "air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 100},
        {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 1500, "torque_nm": -5, "tool_wear_min": 100},
        {"type": "H", "air_temp_k": "abc", "process_temp_k": 310.0, "rot_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 100},
        {"type": "H", "air_temp_k": 300.0, "process_temp_k": None, "rot_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 100},
        {"type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 3500, "torque_nm": 90, "tool_wear_min": 300},
    ])
    st.download_button("Download example CSV (includes invalid rows)", example.to_csv(index=False),
                       "example_inputs.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df_in = pd.read_csv(up)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            df_in = None
        if df_in is not None:
            missing = [c for c in INPUT_FEATURES if c not in df_in.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            elif len(df_in) > 5000:
                st.error("Please upload at most 5,000 rows.")
            else:
                out = predict_batch(artifact, df_in)
                st.dataframe(out, width="stretch")
                st.write(out["status"].value_counts())
                if st.checkbox("Add valid rows to the monitoring log"):
                    for _, r in out[out.status.str.startswith("OK")].iterrows():
                        clean, _, w = validate_inputs(r.to_dict(), artifact)
                        log_prediction(clean, predict(artifact, clean), w, source="batch")
                    st.success("Logged.")
                st.download_button("Download results", out.to_csv(index=False), "predictions.csv", "text/csv")

# --------------------------------------------------------------------------- Agreement
with tab_check:
    st.subheader("Dashboard vs training notebook")
    st.markdown("Re-predicts the reference cases saved by `train.py` and compares the probabilities. "
                "Agreement shows that the saved preprocessing and feature order are applied identically.")
    if not REFERENCE_PATH.exists():
        st.info("reports/reference_predictions.csv not found - run train.py.")
    else:
        ref = pd.read_csv(REFERENCE_PATH)
        app_rows = []
        for _, r in ref.iterrows():
            clean, errors, _ = validate_inputs(r.to_dict(), artifact)
            app_rows.append(predict(artifact, clean) if not errors else {"failure_probability": np.nan, "prediction": "REJECTED"})
        ref["app_proba"] = [a["failure_probability"] for a in app_rows]
        ref["app_prediction"] = [a["prediction"] for a in app_rows]
        ref["abs_diff"] = (ref["app_proba"] - ref["notebook_proba"]).abs()
        ref["match"] = (ref["app_prediction"] == ref["notebook_prediction"]) & (ref["abs_diff"] < 1e-9)
        st.dataframe(ref, width="stretch")
        if ref["match"].all():
            st.success(f"All {len(ref)} reference cases match (max |difference| = {ref['abs_diff'].max():.2e}).")
        else:
            st.error(f"{(~ref['match']).sum()} case(s) do not match - check model/feature versions.")

# --------------------------------------------------------------------------- Monitoring
with tab_mon:
    st.subheader("Monitoring")
    log = read_log()
    window = st.slider("Window (most recent predictions)", 10, 500, 50, step=10)
    summ = summarise_log(log, artifact, window=window)
    if summ["n_total"] == 0:
        st.info("No predictions logged yet.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Logged predictions", summ["n_total"])
        k2.metric("Predicted failure rate", f"{summ['predicted_failure_rate']:.1%}",
                  f"training rate {summ['train_failure_rate']:.1%}", delta_color="off")
        k3.metric("Mean failure probability", f"{summ['mean_probability']:.1%}")
        k4.metric("Out-of-training-range inputs", f"{summ['out_of_range_share']:.0%}")

        if not summ["enough_data"]:
            st.info(f"Review triggers are evaluated once at least 30 predictions are in the window "
                    f"(currently {summ['n_window']}).")
        elif summ["triggers"]:
            st.error("**Model review recommended:**\n\n" + "\n".join(f"- {t}" for t in summ["triggers"]))
        else:
            st.success("No review trigger in the current window.")

        shift = pd.DataFrame({"feature": [FEATURE_LABELS[c] for c in summ["feature_shift"]],
                              "mean shift (training std)": list(summ["feature_shift"].values())})
        st.bar_chart(shift.set_index("feature"))
        st.caption("Failure probability of each logged prediction (oldest → newest)")
        st.line_chart(log["failure_probability"].reset_index(drop=True))
        st.dataframe(log.tail(window), width="stretch")
        st.download_button("Download prediction log", log.to_csv(index=False), "predictions_log.csv", "text/csv")
    st.caption("The log stores machine readings and predictions only (no personal data). On Streamlit Community "
               "Cloud the file is reset when the app restarts, so download it regularly; every prediction is also "
               "written to the app logs (Manage app → Logs). Predictions alone cannot show real accuracy - "
               "actual failure outcomes must be recorded and compared.")

# --------------------------------------------------------------------------- About
with tab_about:
    st.subheader("About the model")
    st.markdown(f"""
- **Data:** AI4I 2020 Predictive Maintenance Dataset (S. Matzka, UCI Machine Learning Repository, CC BY 4.0), 10,000 records.
- **Inputs:** product type, air & process temperature, rotational speed, torque, tool wear.
  Engineered inside the pipeline: temperature difference, mechanical power, wear × torque.
- **Excluded:** identifiers (UDI, Product ID) and failure-mode flags (TWF, HDF, PWF, OSF, RNF) because they reveal the outcome.
- **Target:** Machine failure (Failure / No Failure); positive class = Failure.
- **Selected model:** {artifact['model_name']} with decision threshold {threshold:.2f} (tuned for F1 using cross-validation).
- **Limitations:** synthetic dataset; random failures (RNF) are unpredictable by design; failures are rare (≈3.4%),
  so false alarms and missed failures should both be expected.
""")
    st.json(json.loads(json.dumps({k: v for k, v in artifact.items() if k != "pipeline"}, default=str)),
            expanded=False)
