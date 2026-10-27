# Machine Failure Prediction for Predictive Maintenance
**IFJC4002 Practical Artificial Intelligence with Python – Case 7**
Group: ____ | Members: (see Appendix A) | App: <streamlit link> | Repository: <github link>

> **How to use this draft:** text in `[[...]]` must be replaced with numbers from `reports/results.md`,
> `reports/eda_summary.json` and `reports/model_comparison.csv` after you run `train.py`.
> Do not submit numbers you did not obtain yourselves. Rewrite in your own words where needed.

---

## 1. Problem definition and dataset analysis (CO1)

### 1.1 User, problem and benefit
The intended users are **maintenance planners and machine operators** in a machining workshop. Unplanned machine
failures stop production, damage workpieces and tools, and are more expensive to repair than planned maintenance.
The system predicts, from readings that are available *before or during* an operation, whether the machine is
likely to fail, so that staff can inspect the machine, replace the tool or reduce the load in advance.

Because a missed failure (false negative) is usually more costly than an unnecessary inspection (false positive),
**recall on the Failure class** is the most important single metric, balanced against precision through the F1-score
so that the system does not raise so many false alarms that users ignore it.

### 1.2 Target, inputs and task
* **Task:** binary classification.
* **Target:** `Machine failure` – class labels *Failure* (1, positive class) and *No Failure* (0).
* **Inputs (6):** product quality `Type` (L/M/H, categorical), air temperature [K], process temperature [K],
  rotational speed [rpm], torque [Nm], tool wear [min] (all numeric, continuous or integer).
* **Excluded:** `UDI` and `Product ID` (identifiers with no predictive meaning, would allow memorisation) and the five
  failure-mode flags `TWF, HDF, PWF, OSF, RNF`. These flags are set *only when* a failure occurs, so they reveal the
  outcome and are not available at prediction time (target leakage). They are kept aside only for error analysis.

### 1.3 Dataset
AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020; UCI Machine Learning Repository, CC BY 4.0). It is a
synthetic dataset designed to reflect real predictive-maintenance data from industry. No filtering or sampling was
applied.

| Property | Value |
|---|---|
| Rows × columns | [[10,000 × 14]] |
| Missing values | [[0]] |
| Duplicate rows / Product IDs | [[0 / 0]] |
| Failure / No Failure | [[339 (3.39 %) / 9,661]] |
| Type L / M / H | [[6,000 / 2,997 / 1,003]] |
| Failure modes TWF / HDF / PWF / OSF / RNF | [[46 / 115 / 95 / 98 / 19]] |

(Values in brackets are from the dataset documentation – confirm against `reports/eda_summary.json`.)

**Feature distributions** (Figure 1 – `eda_distributions.png`, `eda_boxplots.png`): air and process temperature
are approximately normal and strongly correlated ([[r ≈ 0.88]]); rotational speed is right-skewed with
[[n]] IQR outliers at high speed, and torque is approximately normal, strongly negatively correlated with speed
([[r ≈ −0.88]]). Tool wear is roughly uniform from 0 to ≈ 250 min. The outliers are physically plausible operating
points rather than errors, so they were **kept**; removing them would remove many of the failure cases.

**Class imbalance** (Figure 2): only ≈ 3.4 % of records are failures. A model that always predicts *No Failure*
reaches ≈ 96.6 % accuracy while detecting nothing, so **accuracy is misleading** and the evaluation focuses on
precision, recall, F1 and PR-AUC for the Failure class.

**Relationship with the target:** failures are associated with high torque, low rotational speed, high tool wear
and a small difference between process and air temperature ([[quote class means from eda_summary.json → mean_by_class]]).
Failure rate differs by type ([[L x %, M y %, H z %]]), consistent with lower-quality products tolerating less strain.

**Label quality:** [[9]] records are labelled *Failure* without any failure-mode flag and [[n]] records have a mode
flag (mostly RNF) but are labelled *No Failure*. These inconsistencies, and the random failures (RNF, 0.1 %) which
by design do not depend on the inputs, set an upper limit on achievable recall.

### 1.4 Suitable AI approaches and implications
* **Logistic regression** – interpretable linear model with probability output; needs scaling and benefits from
  engineered features because the failure mechanisms are non-linear thresholds.
* **Decision tree** – captures threshold rules (e.g. *power < 3,500 W → failure*), very interpretable, but unstable.
* **Random forest** – ensemble of trees; captures non-linear interactions (speed × torque, wear × torque) and is
  robust to outliers, at the cost of interpretability and model size.

Implications: (1) imbalance → class weighting, stratified splitting and threshold tuning; (2) failure mechanisms are
interactions between inputs → engineered physical features (power, temperature difference, wear × torque) and
tree ensembles are expected to perform best; (3) the small number of positives (≈ 68 in the test set) means test
metrics have high variance, so cross-validation is used for model selection.

---

## 2. Solution design and preprocessing (CO2)

### 2.1 Workflow
`Raw CSV → drop identifiers & leakage columns → stratified 80/20 split → [Pipeline: feature engineering →
StandardScaler (numeric) + OneHotEncoder (Type) → classifier] → 5-fold stratified CV grid search on training set →
decision-threshold tuning on out-of-fold training predictions → model selection by CV F1 → single evaluation on the
test set → save pipeline + threshold → Streamlit app.`

(Insert a workflow diagram here.)

### 2.2 Preprocessing decisions
| Step | Decision | Justification |
|---|---|---|
| Feature selection | 6 raw inputs; drop UDI, Product ID, TWF–RNF | identifiers carry no causal information; mode flags leak the target |
| Feature engineering | `temp_diff_k = process − air`, `power_w = torque × rpm × 2π/60`, `wear_torque = wear × torque` | encode the physical mechanisms of heat-dissipation, power and overstrain failure; computed only from inputs, so available at prediction time |
| Missing values | none present; app rejects missing inputs | no imputation needed; avoids silent guessing |
| Encoding | One-hot encoding of Type with fixed categories L/M/H, `handle_unknown="ignore"` | nominal variable; fixed order guarantees identical columns in the app |
| Scaling | StandardScaler on numeric features | required for logistic regression; harmless for trees |
| Fit on training only | all transformers are inside a scikit-learn `Pipeline`, fitted inside each CV fold | prevents information from validation/test data leaking into scaling |
| Split | stratified 80/20, `random_state=42` | keeps the 3.4 % failure rate in both sets; data is not a time series, and each Product ID appears once, so no entity appears in both sets |

### 2.3 Evaluation design
Stratified 5-fold CV on the training set (8,000 rows) for hyper-parameter tuning and model selection; the
test set (2,000 rows, [[68]] failures) is used **once** for final reporting. Metrics: confusion matrix, accuracy,
precision, recall, F1 (positive class = Failure, binary averaging), plus macro-F1, balanced accuracy, ROC-AUC and
PR-AUC. The decision threshold is tuned to maximise F1 on out-of-fold probabilities of the training set (not the test set).

---

## 3. Model development and comparison (CO2)

### 3.1 Models and settings
| Model | Key settings | Grid searched (5-fold CV, scoring = F1) |
|---|---|---|
| Baseline | `DummyClassifier(strategy="most_frequent")` – always predicts No Failure | – |
| Logistic Regression | `class_weight="balanced"`, `max_iter=5000`, L2 penalty | C ∈ {0.01, 0.1, 1, 10} |
| Decision Tree | `class_weight="balanced"` | max_depth ∈ {4, 6, 8, 12}, min_samples_leaf ∈ {1, 5, 20} |
| Random Forest | 300 trees, `class_weight="balanced_subsample"` | max_depth ∈ {None, 12}, min_samples_leaf ∈ {1, 3}, max_features ∈ {sqrt, 0.5} |

`class_weight` makes errors on the rare Failure class cost more during training. All models share the same
preprocessing pipeline, split, folds and seed, so the comparison is fair. Best parameters: [[from results.md]].

### 3.2 Results
**Table – Cross-validation and test results (positive class = Failure)**

| Model | CV F1 (tuned thr.) | Threshold | Test Accuracy | Precision | Recall | F1 | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | [[ ]] | 0.50 | [[≈0.966]] | 0 | 0 | 0 | [[ ]] | | | | |
| Logistic Regression | | | | | | | | | | | |
| Decision Tree | | | | | | | | | | | |
| Random Forest | | | | | | | | | | | |

Insert: confusion matrices (`cm_*.png`), precision-recall curves (`pr_curves.png`).

### 3.3 Interpretation
[[Write 1–2 paragraphs, e.g.:]] The baseline confirms that accuracy alone is uninformative ([[96.6 %]] accuracy,
zero recall). Logistic regression with engineered features reaches [[F1]], showing that the physical features make
the problem partly linearly separable, but it produces [[n]] false positives because failure regions are bounded
thresholds rather than a single linear boundary. Tree-based models capture these thresholds; the random forest
achieves the highest CV F1 ([[ ]]) and test F1 ([[ ]]) with [[n]] missed failures out of [[68]].

**Error analysis** (`reports/error_analysis.csv`, *Detection by failure mode* table): [[e.g. heat-dissipation and
power failures are detected almost perfectly; most missed failures are tool-wear failures (TWF), which occur at
random times between 200–240 min of wear, and random failures (RNF) which are unpredictable by design]].
Practical consequence: a missed TWF means a tool breaks during machining; therefore operators should still follow
a tool-replacement schedule near 200 min of wear.

**Feature importance** (`permutation_importance.png`): [[torque, rotational speed and tool wear are most important;
type contributes little]], which matches the known failure mechanisms and increases trust in the model.

---

## 4. Final model selection, limitations and improvements (CO2)

### 4.1 Selection
The final model is **[[Random Forest]]** with threshold [[ ]], chosen because it has the highest cross-validated F1
on the training data, and its test performance confirms this ([[F1, recall, precision]]). Practical constraints:

| Criterion | LR | DT | RF |
|---|---|---|---|
| Test F1 / Recall | | | |
| Training time (s) | | | |
| Single prediction (ms) | | | |
| Model size (KB) | | | |
| Interpretability | high (coefficients) | high (rules) | medium (importance only) |

Prediction time of [[x ms]] and size of [[x MB]] are acceptable for a dashboard on Streamlit Community Cloud
(1 GB RAM). If interpretability were required by auditors, the decision tree would be the fallback with [[x]] lower F1.

### 4.2 Limitations
1. **Synthetic data** – real machines have sensor noise, drift and missing readings not present in the data.
2. **Few positive cases** ([[68]] in the test set) – one more or fewer detected failure changes recall by ≈ 1.5 %.
3. **Unpredictable failures** – RNF and part of TWF cannot be predicted from the given inputs.
4. **Snapshot data** – no time history per machine; trends (e.g. rising temperature) cannot be used.
5. **Probability calibration** – class weighting shifts probabilities upward; the displayed probability is a risk score.
6. **Extrapolation** – inputs outside the training range (e.g. speed > [[2,886]] rpm) are flagged as less reliable.

### 4.3 Improvements (linked to findings)
1. Collect **time-series sensor data** per machine to predict tool-wear failures earlier (addresses the main source of FN).
2. Try **gradient boosting** (e.g. HistGradientBoosting/XGBoost) and **probability calibration** (CalibratedClassifierCV).
3. Choose the threshold with **maintenance costs** (cost of FN vs FP) instead of F1 once costs are known.
4. Predict the **failure mode** (multi-class) so the dashboard can recommend a specific action.
5. Record **actual outcomes** of logged predictions to measure real-world precision/recall and retrain periodically.

---

## 5. Implementation, dashboard, deployment, testing and monitoring (CO3)

### 5.1 Implementation
* `features.py` – single definition of feature names, order and engineered features shared by training and app.
* `train.py` – reproducible script (fixed seed 42, pinned library versions, automatic dataset download).
* `predictor.py` – validation, prediction, logging and monitoring logic (unit-tested).
* `app.py` – Streamlit UI. `models/machine_failure_model.joblib` stores the **whole pipeline** (feature engineering,
  scaler, encoder, model) plus the threshold, feature order and training statistics, so the app applies exactly the
  same transformations.

### 5.2 Dashboard
Insert screenshots: Predict tab (Failure and No Failure examples), warning for out-of-range input, batch tab with
rejected rows, agreement tab, monitoring tab with a review alert. Describe input controls (select box for Type,
number inputs with valid limits and training-range help text) and the labelled output (class name, probability,
threshold, derived power/temperature difference).

### 5.3 Deployment
GitHub repository → Streamlit Community Cloud (same Python version as training, `requirements.txt` with exact library versions). Link: [[ ]]. Update procedure:
retrain → commit new model files → push → automatic redeploy → re-check the agreement tab.

### 5.4 Testing
* **Notebook–dashboard agreement:** [[14/14]] reference cases (10 test-set cases + 4 boundary cases) give identical
  probabilities (max difference [[0]]). Screenshot of the agreement tab.
* **Automated tests:** `pytest` – [[n passed]] (normal, boundary, invalid, batch, agreement, monitoring).
* **Manual test log:** see `docs/test_log.md` (Appendix C).

### 5.5 Monitoring
Each prediction is logged (inputs, probability, prediction, out-of-range flag – no personal data) to a CSV and to the
Streamlit application log. The Monitoring tab shows predicted failure rate, mean probability, share of out-of-range
inputs and feature-mean drift. **Review trigger:** with ≥ 30 recent predictions, if the predicted failure rate exceeds
3 × the training failure rate (≈ 10 %), more than 10 % of inputs are outside the training range, or any input mean
shifts by > 1 standard deviation. Observation record: [[describe what you logged, e.g. 40 high-wear inputs →
predicted failure rate 85 %, trigger fired for tool wear and rotational speed shift]]. Logged predictions cannot
confirm accuracy without true outcomes; the workshop must record whether each machine actually failed.

---

## 6. Conclusion
[[3–5 sentences: problem, best model and key metric, what the dashboard offers, main limitation, next step.]]

## References
* Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* 2020 Third
  International Conference on Artificial Intelligence for Industries (AI4I), pp. 69–74. doi:10.1109/AI4I49448.2020.00023
* Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5HS5C (CC BY 4.0)
* Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
* Streamlit documentation. https://docs.streamlit.io

## Appendices
* A – Contribution record (`docs/contribution_record.md`)
* B – Prompt record (`docs/prompt_record.md`)
* C – Test log (`docs/test_log.md`) and monitoring observations
