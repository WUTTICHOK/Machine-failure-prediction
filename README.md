# Case 7 – Machine Failure Prediction (IFJC4002)

Binary classification system that predicts whether a machine will **fail during the next operating shift**
(`Failure` / `No Failure`) from readings taken at inspection, deployed as a Streamlit dashboard.

| Item | Link |
|---|---|
| Streamlit app | `https://<your-app>.streamlit.app` ← fill in |
| GitHub repository | `https://github.com/<user>/<repo>` ← fill in |
| Dataset | `data/07_Machine_Failure.xlsx` – provided by the course lecturer |

## Dataset
* `07_Machine_Failure.xlsx`, sheet `Data`: **2,000 simulated machines**, each occurring once
  (synthetic teaching data generated for IFJC4002; no real machines or outcomes). Sheet `Guide` documents the columns.
* Target `failure_next_shift`: Failure 319 (16.0 %), No_Failure 1,681. Positive class = Failure.
* Missing values: 16 blanks in `air_temperature_c` and 16 in `process_temperature_c`; target and IDs complete.
* No filtering or sampling applied.

| Used as input | Excluded |
|---|---|
| air_temperature_c, process_temperature_c (°C), rotational_speed_rpm, torque_nm (N·m), tool_wear_min, vibration_mm_s, machine_age_years | record_id (identifier) |

Engineered inside the pipeline: process − air temperature, mechanical power (torque × speed), tool wear × torque.

## Project structure
```
features.py        shared feature names + engineered features (used by training AND app)
train.py           EDA, train/validation/test split, CV tuning, threshold tuning, evaluation, saves model
predictor.py       input validation, prediction, logging, monitoring triggers
app.py             Streamlit dashboard
tests/             pytest tests (normal, boundary, missing, invalid, agreement, monitoring)
data/              07_Machine_Failure.xlsx
models/            machine_failure_model.joblib (pipeline + threshold + metadata)
reports/           results.md, model_comparison.csv, figures/, reference_predictions.csv
docs/              test log, prompt record, contribution record
logs/              predictions.csv (created by the app; not committed)
```

## Method summary
1. Stratified split: **train 60 % / validation 20 % / test 20 %** (seed 42).
2. Pipeline fitted on training data only: median imputation → engineered features → standard scaling → classifier.
3. Hyper-parameters by stratified 5-fold CV on the training set; decision threshold and final model chosen on the
   validation set; test set used once for the final evaluation.
4. Models: majority-class baseline, Logistic Regression, Decision Tree, Random Forest (class-weighted).

## Reproduce (Google Colab – recommended)
Open `Case7_Machine_Failure_Colab.ipynb` in Colab, **Runtime → Run all**, and upload `07_Machine_Failure.xlsx`
when asked. The notebook writes all files, trains, tests and downloads the complete project as a zip.

## Reproduce (local)
Use the same Python version as the training environment (printed by the notebook).
```bash
pip install -r requirements.txt
python train.py                 # creates models/ and reports/
pytest -q                       # all tests should pass
streamlit run app.py
```

## Deploy (GitHub + Streamlit Community Cloud)
1. Upload the **contents** of the project folder to GitHub, including `models/machine_failure_model.joblib`,
   `reports/reference_predictions.csv` and the `requirements.txt` written by the notebook.
2. https://share.streamlit.io → **Create app** → repo, branch `main`, main file `app.py`.
   Under *Advanced settings* choose the **same Python version** that trained the model (printed by the notebook).
3. Deploy, then open **✅ Notebook agreement** – all reference cases must show *match*.
4. To update the model: re-run the notebook, upload the new `models/` and `reports/` files, push –
   Streamlit Cloud redeploys automatically.

> The model file must be loaded with the same scikit-learn version it was trained with (the notebook writes the
> exact versions into requirements.txt).

## Monitoring
Every prediction is appended to `logs/predictions.csv` (machine readings + prediction only) and written to the app
log (Streamlit Cloud → *Manage app* → *Logs*). The **📈 Monitoring** tab raises a **review trigger** when, with ≥ 30
records in the window:
* predicted failure rate > 2 × the training failure rate (≈ 16 %), or
* > 10 % of inputs fall outside the training range, or
* > 10 % of inputs have a missing temperature reading (sensor problem), or
* the mean of any input shifts by > 1 training standard deviation.

Streamlit Cloud storage is temporary – download the log from the Monitoring tab for your evidence.
Logged predictions alone cannot prove real-world accuracy; true outcomes must be recorded and compared.
