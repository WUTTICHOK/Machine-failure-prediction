"""Streamlit dashboard - Case 7 Machine failure prediction.

Run locally:  streamlit run app.py
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURE_LABELS, IMPUTABLE_FEATURES, INPUT_FEATURES
from predictor import (
    HARD_LIMITS,
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

LOGO_PATH = ROOT / "assets" / "group_logo.png"
GROUP_NAME = "MADE IN THAILAND (Walailak)"

if LOGO_PATH.exists():
    logo_col, title_col = st.columns([1, 5], vertical_alignment="center")
    logo_col.image(str(LOGO_PATH), width=170)
    with title_col:
        st.title("⚙️ Machine Failure Prediction")
        st.caption(f"Group {GROUP_NAME}")
else:
    st.title("⚙️ Machine Failure Prediction")
st.markdown(
    "Predicts whether a machine will **fail during the next operating shift** from readings taken at inspection. "
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
        "1. Enter the inspection readings in **Predict**.\n"
        "2. Tick *reading unavailable* if a temperature sensor gave no value.\n"
        "3. Press **Predict failure**.\n"
        "4. Values outside the training range are allowed but flagged.\n"
        "5. Check **Monitoring** for logged predictions and review alerts."
    )

tab_pred, tab_batch, tab_check, tab_mon, tab_about = st.tabs(
    ["🔮 Predict", "📄 Batch / invalid-input test", "✅ Notebook agreement", "📈 Monitoring", "ℹ️ About"])

# --------------------------------------------------------------------------- Predict
with tab_pred:
    st.subheader("Enter inspection readings")
    steps = {"air_temperature_c": 0.1, "process_temperature_c": 0.1, "rotational_speed_rpm": 10.0,
             "torque_nm": 0.5, "tool_wear_min": 1.0, "vibration_mm_s": 0.01, "machine_age_years": 0.1}
    with st.form("predict_form"):
        cols = st.columns(3)
        inputs, unavailable = {}, {}
        for k, c in enumerate(INPUT_FEATURES):
            col = cols[k % 3]
            lo, hi = HARD_LIMITS[c]
            default = float(round(stats[c]["median"], 2))
            inputs[c] = col.number_input(
                FEATURE_LABELS[c], min_value=lo, max_value=hi, value=default, step=steps[c],
                help=f"Training range: {stats[c]['min']:g} to {stats[c]['max']:g}. Valid range: {lo:g} to {hi:g}.")
            if c in IMPUTABLE_FEATURES:
                unavailable[c] = col.checkbox("Reading unavailable", key=f"na_{c}",
                                              help="The model fills a missing temperature with the training median.")
        submitted = st.form_submit_button("Predict failure", type="primary")

    if submitted:
        for c, na in unavailable.items():
            if na:
                inputs[c] = None
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
                st.error(f"### Prediction: **FAILURE** likely in the next shift\nEstimated failure probability "
                         f"**{p:.1%}** (threshold {res['threshold']:.0%}). Recommend inspection before the next shift.")
            else:
                st.success(f"### Prediction: **NO FAILURE** in the next shift\nEstimated failure probability "
                           f"**{p:.1%}** (threshold {res['threshold']:.0%}).")
            st.progress(min(max(p, 0.0), 1.0))
            m1, m2, m3 = st.columns(3)
            m1.metric("Process − air temperature", f"{res['temp_diff_c']:.1f} °C")
            m2.metric("Mechanical power", f"{res['power_w']:,.0f} W")
            m3.metric("Tool wear × torque", f"{res['wear_torque']:,.0f} min·N·m")
            st.caption("The probability is a model score learned from simulated data; it is not a guarantee. "
                       "Logged predictions cannot confirm accuracy until the true outcome is recorded.")

# --------------------------------------------------------------------------- Batch
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.markdown(f"Upload a CSV with columns `{', '.join(INPUT_FEATURES)}`. Each row is validated; "
                "invalid rows are **rejected with a reason** instead of crashing the app. "
                "Blank temperatures are allowed (median-filled); other blanks are rejected.")
    base = {"air_temperature_c": 28.0, "process_temperature_c": 72.5, "rotational_speed_rpm": 1700,
            "torque_nm": 50.0, "tool_wear_min": 120, "vibration_mm_s": 2.4, "machine_age_years": 7.5}
    example = pd.DataFrame([
        base,
        {**base, "torque_nm": 80.0, "tool_wear_min": 240, "vibration_mm_s": 4.8, "machine_age_years": 13.0},
        {**base, "air_temperature_c": None},
        {**base, "torque_nm": -5},
        {**base, "vibration_mm_s": "abc"},
        {**base, "tool_wear_min": None},
        {**base, "rotational_speed_rpm": 3500},
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
                st.dataframe(out.astype({c: str for c in INPUT_FEATURES}), width="stretch")
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
                "Agreement shows that the saved imputation, features, scaling and feature order are applied identically.")
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
        # Display copy: short number formats; boundary cases have no true outcome
        shown = ref.copy()
        shown["actual"] = shown["actual"].fillna("- (boundary test)").replace("", "- (boundary test)")
        shown["notebook_proba"] = shown["notebook_proba"].round(4)
        shown["app_proba"] = shown["app_proba"].round(4)
        shown["abs_diff"] = shown["abs_diff"].map(lambda v: "0" if v == 0 else f"{v:.0e}")
        st.dataframe(shown, width="stretch", hide_index=True)
        st.caption("abs_diff = |app probability - notebook probability|. Values such as 2e-17 (0.00000000000000002) "
                   "are floating-point rounding from saving to CSV, i.e. identical predictions. "
                   "Blank temperatures (missing_temperatures case) are filled by the same median imputer.")
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
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Logged predictions", summ["n_total"])
        k2.metric("Predicted failure rate", f"{summ['predicted_failure_rate']:.1%}",
                  f"training rate {summ['train_failure_rate']:.1%}", delta_color="off")
        k3.metric("Mean failure probability", f"{summ['mean_probability']:.1%}")
        k4.metric("Out-of-training-range inputs", f"{summ['out_of_range_share']:.0%}")
        k5.metric("Missing temperature readings", f"{summ['missing_share']:.0%}")

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
    split = artifact.get("split", {})
    st.markdown(f"""
- **Data:** `07_Machine_Failure.xlsx` provided by the course lecturer — 2,000 simulated machines, each once
  (synthetic teaching data, not real machines).
- **Inputs:** air & process temperature (°C), rotational speed (rpm), torque (N·m), tool wear (min),
  vibration (mm/s), machine age (years). Engineered inside the pipeline: process − air temperature,
  mechanical power, tool wear × torque. Blank temperatures are filled with the training median.
- **Excluded:** `record_id` (identifier).
- **Target:** `failure_next_shift` (Failure / No Failure in the next operating shift); positive class = Failure.
- **Split:** stratified train / validation / test = {split.get('train', {}).get('rows', '?')} /
  {split.get('validation', {}).get('rows', '?')} / {split.get('test', {}).get('rows', '?')} machines.
- **Selected model:** {artifact['model_name']} with decision threshold {threshold:.2f}
  (hyper-parameters by 5-fold CV, threshold and model choice on the validation set).
- **Limitations:** simulated data with a probabilistic target, so some failures look like normal machines and
  cannot be predicted; failures are about 16% of machines, so false alarms and missed failures should both be expected.
""")
    st.json(json.loads(json.dumps({k: v for k, v in artifact.items() if k != "pipeline"}, default=str)),
            expanded=False)

# --------------------------------------------------------------------------- Footer (every tab)
st.divider()
footer_logo = ""
if LOGO_PATH.exists():  # small logo inline, in front of the footer text
    logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    footer_logo = (f"<img src='data:image/png;base64,{logo_b64}' alt='{GROUP_NAME} logo' "
                   "style='height:56px; vertical-align:middle; margin-right:12px;'>")
st.markdown(
    f"<div style='display:flex; align-items:center; justify-content:center; flex-wrap:wrap; opacity:0.85;'>"
    f"{footer_logo}<span>Developed by group <b>{GROUP_NAME}</b> · "
    "IFJC4002 Practical Artificial Intelligence with Python · Case 7 Machine Failure Prediction</span></div>",
    unsafe_allow_html=True,
)
