# Case 7 – Machine Failure Prediction (IFJC4002)

Binary classification system that predicts whether a milling machine will **fail** (`Failure` / `No Failure`)
from its operating conditions, deployed as a Streamlit dashboard.

| Item | Link |
|---|---|
| Streamlit app | `https://<your-app>.streamlit.app` ← fill in after deployment |
| GitHub repository | `https://github.com/<user>/<repo>` ← fill in |
| Dataset | AI4I 2020 Predictive Maintenance Dataset – https://archive.ics.uci.edu/dataset/601 |

## Dataset
* **Source:** S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications", 2020;
  UCI Machine Learning Repository, DOI 10.24432/C5HS5C. Licence **CC BY 4.0** (redistribution allowed with citation).
* 10,000 rows, 14 columns, synthetic data reflecting real industrial machining. No filtering or sampling applied.
* `train.py` downloads the file automatically to `data/ai4i2020.csv`. If the download fails, download the zip
  from the link above, extract `ai4i2020.csv` and place it in `data/`.

| Used as input | Excluded (reason) |
|---|---|
| Type (L/M/H), Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] | UDI, Product ID (identifiers); TWF, HDF, PWF, OSF, RNF (failure-mode flags – reveal the outcome) |

Target: `Machine failure` (1 = Failure, positive class).

## Project structure
```
features.py        shared feature names + engineered features (used by training AND app)
train.py           EDA, training, CV tuning, threshold tuning, evaluation, saves model
predictor.py       input validation, prediction, logging, monitoring triggers
app.py             Streamlit dashboard
tests/             pytest tests (normal, boundary, invalid, agreement, monitoring)
notebooks/         Google Colab notebook that runs the whole workflow
models/            machine_failure_model.joblib (pipeline + threshold + metadata)
reports/           results.md, model_comparison.csv, figures/, reference_predictions.csv
docs/              report draft, test log, prompt record, contribution record
logs/              predictions.csv (created by the app; not committed)
```

## Reproduce (local)
Use the same Python version as the training environment (printed by the Colab notebook).
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python train.py                 # ~2-5 min; creates models/ and reports/
pytest -q                       # all tests should pass
streamlit run app.py
```
Random seed is fixed (`SEED = 42`), so results are identical between runs with the same library versions.

## Reproduce (Google Colab)
Open `notebooks/Case7_Machine_Failure_Colab.ipynb` in Colab and run all cells. It clones the repository,
installs the pinned requirements, runs `train.py` and the tests and shows the results.

## Deploy (GitHub + Streamlit Community Cloud)
1. Run `python train.py` and commit **including** `models/machine_failure_model.joblib` and
   `reports/reference_predictions.csv` (the app needs both).
2. Push to a public GitHub repository (or private and give the lecturer access).
3. Go to https://share.streamlit.io → **Create app** → choose the repo, branch `main`, main file `app.py`.
   Under *Advanced settings* choose the **same Python version** that trained the model (printed by the notebook).
4. Deploy, then open the **✅ Notebook agreement** tab – all reference cases must show *match*.
5. To update the model: re-run `train.py`, commit the new `models/` and `reports/` files, push –
   Streamlit Cloud redeploys automatically.

> The model file must be loaded with the same scikit-learn version it was trained with (the notebook writes the exact versions into
> requirements.txt). If you change the version, retrain.

## Monitoring
Every prediction is appended to `logs/predictions.csv` (machine readings + prediction only, no personal data)
and written to the app log (Streamlit Cloud → *Manage app* → *Logs*). The **📈 Monitoring** tab shows the
predicted failure rate, mean probability, out-of-range inputs and feature drift, and raises a **review trigger** when
(with ≥ 30 records in the window):
* predicted failure rate > 3 × the training failure rate (≈ 3.4 %), or
* > 10 % of inputs fall outside the training range, or
* the mean of any input shifts by > 1 training standard deviation.

Streamlit Cloud storage is temporary – download the log from the Monitoring tab for your evidence.
Logged predictions alone cannot prove real-world accuracy; true failure outcomes must be recorded and compared.
