"""Case 7 - Machine failure prediction (binary classification).

Reproducible end-to-end workflow:
  1. download / load the AI4I 2020 Predictive Maintenance dataset
  2. exploratory data analysis (saved to reports/)
  3. stratified train/test split (test set kept aside until the end)
  4. baseline + Logistic Regression + Decision Tree + Random Forest,
     tuned with stratified 5-fold cross-validation on the training set only
  5. decision-threshold tuning on out-of-fold training predictions
  6. model selection by cross-validated F1 (not by the test set)
  7. final test evaluation, error analysis, feature importance
  8. save the full pipeline (preprocessing + model) and reference
     predictions used to check that the dashboard agrees with this script

Run:  python train.py
"""
import io
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    INPUT_FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    TARGET_NAMES,
    TYPE_CATEGORIES,
    add_engineered_features,
    clean_columns,
)

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "machine_failure_model.joblib"

DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
]


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        for url in DATA_URLS:
            try:
                print(f"Downloading dataset from {url}")
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                        DATA_PATH.write_bytes(zf.read(csv_name))
                else:
                    DATA_PATH.write_bytes(resp.content)
                break
            except Exception as exc:  # try the next mirror
                print(f"  failed: {exc}")
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                "Could not download the dataset. Download ai4i2020.csv manually from "
                "https://archive.ics.uci.edu/dataset/601 and place it in data/."
            )
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------------------------------
# 2. Exploratory data analysis
# --------------------------------------------------------------------------
def run_eda(raw: pd.DataFrame) -> dict:
    df = clean_columns(raw)
    y = df[TARGET]
    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(raw.columns),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "missing_values": {c: int(v) for c, v in raw.isna().sum().items()},
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_product_ids": int(raw["Product ID"].duplicated().sum()),
        "class_counts": {TARGET_NAMES[k]: int(v) for k, v in y.value_counts().sort_index().items()},
        "failure_rate": float(y.mean()),
        "failure_mode_counts": {c: int(df[c].sum()) for c in LEAKAGE_COLUMNS},
        # label quality checks
        "failure_without_mode_flag": int(((y == 1) & (df[LEAKAGE_COLUMNS].sum(axis=1) == 0)).sum()),
        "mode_flag_without_failure": int(((y == 0) & (df[LEAKAGE_COLUMNS].sum(axis=1) > 0)).sum()),
        "type_counts": {k: int(v) for k, v in df["type"].value_counts().items()},
        "failure_rate_by_type": {k: float(v) for k, v in df.groupby("type")[TARGET].mean().items()},
        "numeric_describe": df[NUMERIC_FEATURES].describe().round(3).to_dict(),
    }
    # unusual values: IQR rule
    iqr_out = {}
    for c in NUMERIC_FEATURES:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_out[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    summary["iqr_outliers"] = iqr_out
    summary["mean_by_class"] = df.groupby(TARGET)[NUMERIC_FEATURES].mean().round(3).to_dict()

    eng = add_engineered_features(df[INPUT_FEATURES])
    summary["correlation"] = eng[MODEL_NUMERIC_FEATURES].corr().round(3).to_dict()

    # --- figures ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = y.value_counts().sort_index()
    ax.bar([TARGET_NAMES[i] for i in counts.index], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(y):.1%})", ha="center", va="bottom")
    ax.set_title("Class balance")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_class_balance.png", dpi=150)
    plt.close(fig)

    plot_cols = NUMERIC_FEATURES + ["temp_diff_k", "power_w", "wear_torque"]
    eng[TARGET] = y.values
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.ravel(), plot_cols):
        for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
            ax.hist(eng.loc[eng[TARGET] == label, c], bins=40, alpha=0.6, density=True,
                    color=color, label=TARGET_NAMES[label])
        ax.set_title(c)
    axes.ravel()[0].legend()
    fig.suptitle("Feature distributions by class (density)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(18, 4))
    for ax, c in zip(axes, NUMERIC_FEATURES):
        ax.boxplot([df.loc[y == 0, c], df.loc[y == 1, c]])
        ax.set_xticks([1, 2], ["No Failure", "Failure"])
        ax.set_title(c)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_boxplots.png", dpi=150)
    plt.close(fig)

    corr = eng[MODEL_NUMERIC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_correlation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    modes = summary["failure_mode_counts"]
    ax.bar(list(modes.keys()), list(modes.values()), color="#DD8452")
    ax.set_title("Failure modes (excluded from inputs - leakage)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_failure_modes.png", dpi=150)
    plt.close(fig)

    with open(REPORT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def make_preprocessor() -> Pipeline:
    column_tf = ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUMERIC_FEATURES),
        ("cat", OneHotEncoder(categories=[TYPE_CATEGORIES], handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
        ("columns", column_tf),
    ])


def model_specs() -> dict:
    """Candidate models and the hyper-parameter grids searched by CV."""
    return {
        "Baseline (majority class)": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [1, 5, 20]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED),
            {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 3], "clf__max_features": ["sqrt", 0.5]},
        ),
    }


def best_threshold(y_true, proba) -> tuple[float, float]:
    """Threshold that maximises F1 on out-of-fold training probabilities."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def test_metrics(y_true, y_pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(proba)) > 1
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, proba) if has_both else 0.5,
        "pr_auc": average_precision_score(y_true, proba) if has_both else float(np.mean(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    np.random.seed(SEED)
    for d in (MODEL_DIR, REPORT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")
    eda = run_eda(raw)
    print(f"Failure rate: {eda['failure_rate']:.2%}  |  missing values: {sum(eda['missing_values'].values())}")

    df = clean_columns(raw)
    # Drop identifiers and outcome-revealing failure-mode flags.
    X = df[INPUT_FEATURES].copy()
    y = df[TARGET].astype(int)
    modes = df[LEAKAGE_COLUMNS]  # kept only for error analysis, never used as input

    # Each Product ID appears once, so no entity can fall into both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Train {len(X_train)} (failures {y_train.sum()}) | Test {len(X_test)} (failures {y_test.sum()})")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results, fitted = [], {}

    for name, (clf, grid) in model_specs().items():
        print(f"\n=== {name} ===")
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        t0 = time.perf_counter()
        search = GridSearchCV(pipe, grid or {}, cv=cv, n_jobs=-1, refit="f1",
                              scoring={"f1": "f1", "pr_auc": "average_precision", "recall": "recall"})
        search.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        best = search.best_estimator_
        i = search.best_index_
        cv_res = search.cv_results_

        # Threshold tuning on out-of-fold probabilities (training data only).
        if name.startswith("Baseline"):
            threshold, cv_f1_tuned = 0.5, float(cv_res["mean_test_f1"][i])
        else:
            oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            threshold, cv_f1_tuned = best_threshold(y_train, oof)

        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        t0 = time.perf_counter()
        best.predict_proba(X_test.iloc[:1])
        latency_ms = (time.perf_counter() - t0) * 1000
        buf = io.BytesIO()
        joblib.dump(best, buf)
        size_kb = buf.getbuffer().nbytes / 1024

        m = test_metrics(y_test, pred, proba)
        row = {
            "model": name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1@0.5": float(cv_res["mean_test_f1"][i]),
            "cv_f1_std": float(cv_res["std_test_f1"][i]),
            "cv_pr_auc": float(cv_res["mean_test_pr_auc"][i]),
            "cv_recall@0.5": float(cv_res["mean_test_recall"][i]),
            "threshold": threshold,
            "cv_f1@threshold": cv_f1_tuned,
            "fit_seconds": fit_s,
            "single_prediction_ms": latency_ms,
            "model_size_kb": size_kb,
            **{f"test_{k}": v for k, v in m.items()},
        }
        results.append(row)
        fitted[name] = (best, threshold, proba, pred)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, default=str))

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=[0, 1]),
                               display_labels=["No Failure", "Failure"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{name}\n(threshold {threshold:.2f})")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"cm_{name.split(' (')[0].replace(' ', '_').lower()}.png", dpi=150)
        plt.close(fig)

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # PR curves for all non-baseline models on the test set
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_, _, proba, _) in fitted.items():
        if not name.startswith("Baseline"):
            PrecisionRecallDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.axhline(y_test.mean(), ls="--", c="grey", label="Baseline (prevalence)")
    ax.legend()
    ax.set_title("Precision-Recall curves (test set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- selection
    # Choose by cross-validated F1 on training data; the test set is only reported.
    candidates = res_df[~res_df["model"].str.startswith("Baseline")]
    final_name = candidates.sort_values("cv_f1@threshold", ascending=False).iloc[0]["model"]
    final_pipe, final_t, final_proba, final_pred = fitted[final_name]
    print(f"\nSelected model: {final_name} (threshold {final_t:.2f})")

    # ---------------------------------------------------------------- error analysis
    err = X_test.copy()
    err["actual"] = y_test.values
    err["proba"] = final_proba
    err["predicted"] = final_pred
    err = err.join(modes)
    err["error_type"] = np.select(
        [(err.actual == 1) & (err.predicted == 0), (err.actual == 0) & (err.predicted == 1)],
        ["False negative (missed failure)", "False positive (false alarm)"], default="Correct")
    err[err.error_type != "Correct"].to_csv(REPORT_DIR / "error_analysis.csv")
    fail = err[err.actual == 1]
    recall_by_mode = {
        c: {"n": int(fail[c].sum()),
            "detected": int(((fail[c] == 1) & (fail.predicted == 1)).sum())}
        for c in LEAKAGE_COLUMNS}
    recall_by_mode["no mode flag"] = {
        "n": int((fail[LEAKAGE_COLUMNS].sum(axis=1) == 0).sum()),
        "detected": int(((fail[LEAKAGE_COLUMNS].sum(axis=1) == 0) & (fail.predicted == 1)).sum())}
    report_txt = classification_report(y_test, final_pred, target_names=["No Failure", "Failure"], digits=4)

    # ---------------------------------------------------------------- importance
    perm = permutation_importance(final_pipe, X_test, y_test, scoring="f1", n_repeats=10,
                                  random_state=SEED, n_jobs=-1)
    perm_df = pd.DataFrame({"feature": INPUT_FEATURES, "importance_mean": perm.importances_mean,
                            "importance_std": perm.importances_std}).sort_values("importance_mean")
    perm_df.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(perm_df.feature, perm_df.importance_mean, xerr=perm_df.importance_std, color="#55A868")
    ax.set_title(f"Permutation importance (test F1 drop)\n{final_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)

    names_out = final_pipe.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    clf = final_pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=names_out).sort_values()
    else:
        imp = pd.Series(clf.coef_[0], index=names_out).sort_values()
    imp.to_csv(REPORT_DIR / "model_feature_importance.csv", header=["value"])

    # LR coefficients are useful for interpretation even if LR is not selected
    lr = fitted["Logistic Regression"][0]
    lr_names = lr.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    pd.Series(lr.named_steps["clf"].coef_[0], index=lr_names).sort_values().to_csv(
        REPORT_DIR / "logreg_coefficients.csv", header=["coefficient"])

    # ---------------------------------------------------------------- save artefact
    stats = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                 "mean": float(X_train[c].mean()), "std": float(X_train[c].std()),
                 "p01": float(X_train[c].quantile(0.01)), "p99": float(X_train[c].quantile(0.99))}
             for c in NUMERIC_FEATURES}
    final_row = res_df[res_df.model == final_name].iloc[0].to_dict()
    artifact = {
        "pipeline": final_pipe,
        "threshold": final_t,
        "model_name": final_name,
        "input_features": INPUT_FEATURES,
        "type_categories": TYPE_CATEGORIES,
        "target_names": TARGET_NAMES,
        "feature_stats": stats,
        "train_failure_rate": float(y_train.mean()),
        "test_metrics": {k.replace("test_", ""): v for k, v in final_row.items() if k.startswith("test_")},
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
    }
    joblib.dump(artifact, MODEL_PATH)
    meta = {k: v for k, v in artifact.items() if k != "pipeline"}
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")

    # ---------------------------------------------------------------- reference cases
    # Typical cases from the test set (both classes) plus boundary cases at the
    # training-range limits. The dashboard re-predicts these to prove agreement.
    rng = np.random.RandomState(SEED)
    pos_idx = rng.choice(X_test.index[y_test == 1], 5, replace=False)
    neg_idx = rng.choice(X_test.index[y_test == 0], 5, replace=False)
    ref = X_test.loc[list(pos_idx) + list(neg_idx)].copy()
    ref["case"] = [f"test_failure_{i + 1}" for i in range(5)] + [f"test_normal_{i + 1}" for i in range(5)]
    ref["actual"] = y_test.loc[ref.index].map(TARGET_NAMES).values
    lo = {c: stats[c]["min"] for c in NUMERIC_FEATURES}
    hi = {c: stats[c]["max"] for c in NUMERIC_FEATURES}
    boundary = pd.DataFrame([
        {"type": "L", **lo, "case": "boundary_all_min", "actual": ""},
        {"type": "H", **hi, "case": "boundary_all_max", "actual": ""},
        {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 1500,
         "torque_nm": 40.0, "tool_wear_min": 0, "case": "boundary_new_tool", "actual": ""},
        {"type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0, "rot_speed_rpm": 1500,
         "torque_nm": 40.0, "tool_wear_min": hi["tool_wear_min"], "case": "boundary_max_wear", "actual": ""},
    ])
    ref = pd.concat([ref.reset_index(drop=True), boundary], ignore_index=True)
    ref_proba = final_pipe.predict_proba(ref[INPUT_FEATURES])[:, 1]
    ref["notebook_proba"] = ref_proba
    ref["notebook_prediction"] = np.where(ref_proba >= final_t, "Failure", "No Failure")
    ref[["case"] + INPUT_FEATURES + ["actual", "notebook_proba", "notebook_prediction"]].to_csv(
        REPORT_DIR / "reference_predictions.csv", index=False)

    write_results_md(eda, res_df, final_name, final_t, report_txt, recall_by_mode, perm_df, ref)
    print("\nDone. See reports/results.md")


def write_results_md(eda, res_df, final_name, final_t, report_txt, recall_by_mode, perm_df, ref):
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    cols = ["model", "cv_f1@0.5", "cv_f1@threshold", "threshold", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_macro_f1", "test_roc_auc", "test_pr_auc",
            "test_tn", "test_fp", "test_fn", "test_tp", "fit_seconds", "single_prediction_ms", "model_size_kb"]
    lines = [
        "# Case 7 - Results (auto-generated by train.py)",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | scikit-learn {sklearn.__version__} | seed {SEED}_",
        "",
        "## Dataset",
        f"- Rows: {eda['n_rows']}, columns: {eda['n_columns']}, missing values: {sum(eda['missing_values'].values())}, "
        f"duplicate rows: {eda['duplicate_rows']}",
        f"- Class counts: {eda['class_counts']} (failure rate {eda['failure_rate']:.2%})",
        f"- Failure modes (excluded as leakage): {eda['failure_mode_counts']}",
        f"- Failures with no mode flag: {eda['failure_without_mode_flag']}; "
        f"mode flag but no failure: {eda['mode_flag_without_failure']}",
        f"- Failure rate by type: { {k: round(v, 4) for k, v in eda['failure_rate_by_type'].items()} }",
        f"- IQR outliers per feature: {eda['iqr_outliers']}",
        "",
        "## Model comparison (positive class = Failure, binary averaging)",
        "| " + " | ".join(cols) + " | best params |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for _, r in res_df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + f" | {r['best_params']} |")
    lines += [
        "",
        f"## Selected model: **{final_name}** (decision threshold {final_t:.2f})",
        "Selected by highest cross-validated F1 on the training set; test set used only for final reporting.",
        "",
        "```", report_txt, "```",
        "",
        "### Detection of failures by failure mode (test set)",
        "| mode | failures | detected |", "|---|---|---|",
    ]
    for k, v in recall_by_mode.items():
        lines.append(f"| {k} | {v['n']} | {v['detected']} |")
    lines += ["", "### Permutation importance (drop in test F1)", "| feature | mean | std |", "|---|---|---|"]
    for _, r in perm_df.sort_values("importance_mean", ascending=False).iterrows():
        lines.append(f"| {r.feature} | {r.importance_mean:.4f} | {r.importance_std:.4f} |")
    lines += ["", "## Reference predictions (use to check the dashboard)", "",
              "```", ref.round(4).to_string(index=False), "```"]
    (REPORT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
