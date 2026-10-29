"""Case 7 - Machine failure prediction (binary classification).

Reproducible end-to-end workflow on the lecturer-provided dataset
data/07_Machine_Failure.xlsx (sheet "Data", 2,000 simulated machines):
  1. load the data, exploratory data analysis (saved to reports/)
  2. stratified train / validation / test split (60 / 20 / 20)
  3. baseline + Logistic Regression + Decision Tree + Random Forest,
     hyper-parameters tuned by stratified 5-fold CV on the training set
  4. decision threshold tuned and final model selected on the validation set
  5. one final evaluation on the test set, error analysis, feature importance
  6. save the full pipeline (imputation + features + scaling + model) and
     reference predictions used to check that the dashboard agrees

Run:  python train.py
"""
import io
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import (
    DATA_FILE,
    DATA_SHEET,
    ENGINEERED_FEATURES,
    ID_COLUMN,
    IMPUTABLE_FEATURES,
    INPUT_FEATURES,
    MODEL_FEATURES,
    TARGET,
    TARGET_NAMES,
    add_engineered_features,
    encode_target,
)

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20   # of the whole dataset
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / DATA_FILE
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_PATH = MODEL_DIR / "machine_failure_model.joblib"


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Put the lecturer-provided file {DATA_FILE} in {DATA_PATH.parent}/")
    df = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)
    missing_cols = [c for c in [ID_COLUMN, TARGET] + INPUT_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")
    return df


# --------------------------------------------------------------------------
# 2. Exploratory data analysis
# --------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> dict:
    y = encode_target(df[TARGET])
    X = df[INPUT_FEATURES]
    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "rows_with_missing_input": int(X.isna().any(axis=1).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[ID_COLUMN].duplicated().sum()),
        "target_values": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
        "class_counts": {TARGET_NAMES[k]: int(v) for k, v in y.value_counts().sort_index().items()},
        "failure_rate": float(y.mean()),
        "process_not_above_air": int((df["process_temperature_c"] <= df["air_temperature_c"]).sum()),
        "numeric_describe": X.describe().round(3).to_dict(),
        "mean_by_class": X.groupby(y.map(TARGET_NAMES)).mean().round(3).to_dict(),
    }
    iqr_out = {}
    for c in INPUT_FEATURES:
        q1, q3 = X[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_out[c] = int(((X[c] < q1 - 1.5 * iqr) | (X[c] > q3 + 1.5 * iqr)).sum())
    summary["iqr_outliers"] = iqr_out

    # failure rate across quintiles of each input: shows which inputs drive failure
    by_band = {}
    for c in INPUT_FEATURES:
        bands = pd.qcut(X[c], 5, duplicates="drop")
        rate = y.groupby(bands, observed=True).mean()
        by_band[c] = {str(k): round(float(v), 4) for k, v in rate.items()}
    summary["failure_rate_by_quintile"] = by_band

    eng = add_engineered_features(X)
    summary["correlation"] = eng[MODEL_FEATURES].corr().round(3).to_dict()
    summary["correlation_with_target"] = {
        c: round(float(eng[c].corr(y)), 3) for c in MODEL_FEATURES}

    # --- figures ---
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = y.value_counts().sort_index()
    ax.bar([TARGET_NAMES[i] for i in counts.index], counts.values, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / len(y):.1%})", ha="center", va="bottom")
    ax.set_title("Class balance (failure_next_shift)")
    ax.set_ylabel("Machines")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_class_balance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    miss = df[INPUT_FEATURES].isna().sum()
    ax.barh(miss.index, miss.values, color="#8172B2")
    ax.set_title("Missing values per input")
    ax.set_xlabel("Blank cells")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_missing_values.png", dpi=150)
    plt.close(fig)

    eng["_y"] = y.values
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, c in zip(axes.ravel(), MODEL_FEATURES):
        for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
            ax.hist(eng.loc[eng["_y"] == label, c].dropna(), bins=30, alpha=0.6, density=True,
                    color=color, label=TARGET_NAMES[label])
        ax.set_title(c)
    axes.ravel()[0].legend()
    fig.suptitle("Feature distributions by class (density)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(INPUT_FEATURES), figsize=(22, 4))
    for ax, c in zip(axes, INPUT_FEATURES):
        ax.boxplot([X.loc[y == 0, c].dropna(), X.loc[y == 1, c].dropna()])
        ax.set_xticks([1, 2], ["No Failure", "Failure"])
        ax.set_title(c)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_boxplots.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(INPUT_FEATURES), figsize=(22, 3.8), sharey=True)
    for ax, c in zip(axes, INPUT_FEATURES):
        rates = list(by_band[c].values())
        ax.plot(range(1, len(rates) + 1), rates, marker="o", color="#C44E52")
        ax.axhline(float(y.mean()), ls="--", c="grey", lw=1)
        ax.set_xticks(range(1, len(rates) + 1))
        ax.set_xlabel("quintile (low → high)")
        ax.set_title(c, fontsize=10)
    axes[0].set_ylabel("Failure rate")
    fig.suptitle("Failure rate by quintile of each input (dashed = overall rate)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_failure_rate_by_quintile.png", dpi=150)
    plt.close(fig)

    corr = eng[MODEL_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Correlation matrix (inputs + engineered features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_correlation.png", dpi=150)
    plt.close(fig)

    with open(REPORT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def make_preprocessor() -> Pipeline:
    """Median imputation -> engineered features -> scaling (all fitted on training data only)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
        ("scale", StandardScaler()),
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
            {"clf__max_depth": [3, 4, 6, 8], "clf__min_samples_leaf": [5, 10, 20]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED),
            {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", 0.5]},
        ),
    }


def best_threshold(y_true, proba) -> tuple[float, float]:
    """Threshold that maximises F1 on the validation set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def metrics(y_true, y_pred, proba) -> dict:
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

    df = load_data()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    eda = run_eda(df)
    print(f"Failure rate: {eda['failure_rate']:.2%}  |  rows with a missing input: {eda['rows_with_missing_input']}")

    X = df[INPUT_FEATURES].copy()      # record_id excluded
    y = encode_target(df[TARGET])

    # Each machine occurs once, so a stratified random split keeps machines apart.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_tmp, random_state=SEED)
    split = {name: {"rows": int(len(t)), "failures": int(t.sum())}
             for name, t in [("train", y_train), ("validation", y_val), ("test", y_test)]}
    print("Split:", split)

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

        # Threshold tuning and model selection on the validation set
        val_proba = best.predict_proba(X_val)[:, 1]
        if name.startswith("Baseline"):
            threshold = 0.5
            val_f1 = f1_score(y_val, (val_proba >= threshold).astype(int), zero_division=0)
        else:
            threshold, val_f1 = best_threshold(y_val, val_proba)
        val_m = metrics(y_val, (val_proba >= threshold).astype(int), val_proba)

        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        t0 = time.perf_counter()
        best.predict_proba(X_test.iloc[:1])
        latency_ms = (time.perf_counter() - t0) * 1000
        buf = io.BytesIO()
        joblib.dump(best, buf)
        size_kb = buf.getbuffer().nbytes / 1024

        m = metrics(y_test, pred, proba)
        row = {
            "model": name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_f1@0.5": float(cv_res["mean_test_f1"][i]),
            "cv_f1_std": float(cv_res["std_test_f1"][i]),
            "cv_pr_auc": float(cv_res["mean_test_pr_auc"][i]),
            "threshold": threshold,
            "val_f1@threshold": float(val_f1),
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_pr_auc": val_m["pr_auc"],
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
    # Highest validation F1 at the tuned threshold (ties -> higher CV F1); test set only reported.
    candidates = res_df[~res_df["model"].str.startswith("Baseline")]
    final_name = candidates.sort_values(["val_f1@threshold", "cv_f1@0.5"], ascending=False).iloc[0]["model"]
    final_pipe, final_t, final_proba, final_pred = fitted[final_name]
    print(f"\nSelected model: {final_name} (threshold {final_t:.2f})")

    # ---------------------------------------------------------------- error analysis
    err = X_test.copy()
    err["actual"] = y_test.values
    err["proba"] = final_proba
    err["predicted"] = final_pred
    err["outcome"] = np.select(
        [(err.actual == 1) & (err.predicted == 1), (err.actual == 1) & (err.predicted == 0),
         (err.actual == 0) & (err.predicted == 1)],
        ["True positive", "False negative (missed failure)", "False positive (false alarm)"],
        default="True negative")
    err[~err.outcome.str.startswith("True")].to_csv(REPORT_DIR / "error_analysis.csv")
    outcome_means = err.groupby("outcome")[INPUT_FEATURES].mean().round(2)
    outcome_counts = err["outcome"].value_counts()
    outcome_means.insert(0, "count", outcome_counts)
    outcome_means.to_csv(REPORT_DIR / "error_profile.csv")

    wear_band = pd.cut(err["tool_wear_min"], [-1, 100, 150, 200, 250, 1000],
                       labels=["0-100", "101-150", "151-200", "201-250", ">250"])
    by_wear = (err.assign(band=wear_band,
                          detected=(err.actual == 1) & (err.predicted == 1),
                          false_alarm=(err.actual == 0) & (err.predicted == 1))
               .groupby("band", observed=True)
               .agg(machines=("actual", "size"), failures=("actual", "sum"),
                    detected=("detected", "sum"), false_alarms=("false_alarm", "sum")))
    by_wear.to_csv(REPORT_DIR / "errors_by_tool_wear.csv")
    missing_rows = err[INPUT_FEATURES].isna().any(axis=1)
    missing_info = {"test_rows_with_missing_input": int(missing_rows.sum()),
                    "correct_on_those_rows": int((err.loc[missing_rows, "actual"] == err.loc[missing_rows, "predicted"]).sum())}
    report_txt = classification_report(y_test, final_pred, target_names=["No Failure", "Failure"], digits=4)

    # ---------------------------------------------------------------- importance
    perm = permutation_importance(final_pipe, X_test, y_test, scoring="average_precision", n_repeats=10,
                                  random_state=SEED, n_jobs=-1)
    perm_df = pd.DataFrame({"feature": INPUT_FEATURES, "importance_mean": perm.importances_mean,
                            "importance_std": perm.importances_std}).sort_values("importance_mean")
    perm_df.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(perm_df.feature, perm_df.importance_mean, xerr=perm_df.importance_std, color="#55A868")
    ax.set_title(f"Permutation importance (drop in test PR-AUC)\n{final_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)

    names_out = final_pipe.named_steps["prep"].named_steps["scale"].get_feature_names_out()
    clf = final_pipe.named_steps["clf"]
    imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else clf.coef_[0]
    pd.Series(imp, index=names_out).sort_values().to_csv(REPORT_DIR / "model_feature_importance.csv", header=["value"])
    lr = fitted["Logistic Regression"][0]
    pd.Series(lr.named_steps["clf"].coef_[0], index=names_out).sort_values().to_csv(
        REPORT_DIR / "logreg_coefficients.csv", header=["coefficient"])

    # ---------------------------------------------------------------- save artefact
    stats = {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                 "mean": float(X_train[c].mean()), "std": float(X_train[c].std()),
                 "median": float(X_train[c].median())}
             for c in INPUT_FEATURES}
    final_row = res_df[res_df.model == final_name].iloc[0].to_dict()
    artifact = {
        "pipeline": final_pipe,
        "threshold": final_t,
        "model_name": final_name,
        "input_features": INPUT_FEATURES,
        "imputable_features": IMPUTABLE_FEATURES,
        "target_names": TARGET_NAMES,
        "feature_stats": stats,
        "train_failure_rate": float(y_train.mean()),
        "split": split,
        "test_metrics": {k.replace("test_", ""): v for k, v in final_row.items() if k.startswith("test_")},
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    meta = {k: v for k, v in artifact.items() if k != "pipeline"}
    with open(MODEL_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")

    # ---------------------------------------------------------------- reference cases
    rng = np.random.RandomState(SEED)
    pos_idx = rng.choice(X_test.index[y_test == 1], 5, replace=False)
    neg_idx = rng.choice(X_test.index[y_test == 0], 5, replace=False)
    ref = X_test.loc[list(pos_idx) + list(neg_idx)].copy()
    ref["case"] = [f"test_failure_{i + 1}" for i in range(5)] + [f"test_normal_{i + 1}" for i in range(5)]
    ref["actual"] = y_test.loc[ref.index].map(TARGET_NAMES).values
    lo = {c: stats[c]["min"] for c in INPUT_FEATURES}
    hi = {c: stats[c]["max"] for c in INPUT_FEATURES}
    med = {c: stats[c]["median"] for c in INPUT_FEATURES}
    boundary = pd.DataFrame([
        {**lo, "case": "boundary_all_min", "actual": ""},
        {**hi, "case": "boundary_all_max", "actual": ""},
        {**med, "tool_wear_min": 0.0, "case": "boundary_new_tool", "actual": ""},
        {**med, "tool_wear_min": hi["tool_wear_min"], "case": "boundary_max_wear", "actual": ""},
        {**med, "air_temperature_c": np.nan, "process_temperature_c": np.nan,
         "case": "missing_temperatures", "actual": ""},
    ])
    ref = pd.concat([ref.reset_index(drop=True), boundary], ignore_index=True)
    ref_proba = final_pipe.predict_proba(ref[INPUT_FEATURES])[:, 1]
    ref["notebook_proba"] = ref_proba
    ref["notebook_prediction"] = np.where(ref_proba >= final_t, "Failure", "No Failure")
    ref[["case"] + INPUT_FEATURES + ["actual", "notebook_proba", "notebook_prediction"]].to_csv(
        REPORT_DIR / "reference_predictions.csv", index=False)

    write_results_md(eda, split, res_df, final_name, final_t, report_txt, by_wear, outcome_means,
                     missing_info, perm_df, ref)
    print("\nDone. See reports/results.md")


def write_results_md(eda, split, res_df, final_name, final_t, report_txt, by_wear, outcome_means,
                     missing_info, perm_df, ref):
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    cols = ["model", "cv_f1@0.5", "threshold", "val_f1@threshold", "val_recall", "val_precision",
            "test_accuracy", "test_precision", "test_recall", "test_f1", "test_macro_f1", "test_roc_auc",
            "test_pr_auc", "test_tn", "test_fp", "test_fn", "test_tp",
            "fit_seconds", "single_prediction_ms", "model_size_kb"]
    lines = [
        "# Case 7 - Results (auto-generated by train.py)",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | scikit-learn {sklearn.__version__} | seed {SEED}_",
        "",
        "## Dataset (data/07_Machine_Failure.xlsx, sheet Data)",
        f"- Rows: {eda['n_rows']}, columns: {eda['n_columns']}, duplicate rows: {eda['duplicate_rows']}, "
        f"duplicate IDs: {eda['duplicate_ids']}",
        f"- Missing values: { {k: v for k, v in eda['missing_values'].items() if v} } "
        f"(rows with any missing input: {eda['rows_with_missing_input']})",
        f"- Class counts: {eda['class_counts']} (failure rate {eda['failure_rate']:.2%})",
        f"- Process temperature not above air temperature: {eda['process_not_above_air']} rows",
        f"- IQR outliers per input: {eda['iqr_outliers']}",
        f"- Correlation with target: {eda['correlation_with_target']}",
        "",
        "### Mean by class", "```", pd.DataFrame(eda["mean_by_class"]).T.to_string(), "```",
        "",
        "### Failure rate by quintile (low -> high)", "```",
        pd.DataFrame({c: list(v.values()) for c, v in eda["failure_rate_by_quintile"].items()
                      if len(v) == 5}).T.rename(columns=lambda i: f"Q{i + 1}").to_string(), "```",
        "",
        f"## Split (stratified, seed {SEED}): {split}",
        "",
        "## Model comparison (positive class = Failure, binary averaging)",
        "Hyper-parameters: 5-fold CV on training set. Threshold + model selection: validation set. "
        "Test set: final evaluation only.",
        "",
        "| " + " | ".join(cols) + " | best params |",
        "|" + "---|" * (len(cols) + 1),
    ]
    for _, r in res_df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + f" | {r['best_params']} |")
    lines += [
        "",
        f"## Selected model: **{final_name}** (decision threshold {final_t:.2f})",
        "Selected by highest validation F1 at its tuned threshold; the test set was used only for final reporting.",
        "",
        "```", report_txt, "```",
        "",
        "### Test-set outcomes by tool wear (min)", "```", by_wear.to_string(), "```",
        "",
        "### Mean inputs by outcome (test set)", "```", outcome_means.to_string(), "```",
        "",
        f"### Rows with a missing input in the test set: {missing_info}",
        "",
        "### Permutation importance (drop in test PR-AUC)", "| feature | mean | std |", "|---|---|---|",
    ]
    for _, r in perm_df.sort_values("importance_mean", ascending=False).iterrows():
        lines.append(f"| {r.feature} | {r.importance_mean:.4f} | {r.importance_std:.4f} |")
    lines += ["", "## Reference predictions (use to check the dashboard)", "",
              "```", ref.round(4).to_string(index=False), "```"]
    (REPORT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
