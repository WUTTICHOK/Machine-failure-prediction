# AI-assisted development – prompt record

The brief requires **at least five representative prompts covering different development stages**, with an
explanation of how the generated code was checked, corrected or adapted. Record the prompts your group actually
used (copy them exactly). The stages below are a guide; one row per prompt.

| # | Stage | Member | Prompt (exact text) | What the AI produced | How we checked it | What we changed and why |
|---|---|---|---|---|---|---|
| 1 | Data analysis / EDA | | | | e.g. compared counts with UCI documentation; re-ran `df.describe()` | |
| 2 | Preprocessing & leakage | | | | e.g. checked the imputer is fitted after the split, inside the pipeline; record_id excluded | |
| 3 | Model training / tuning | | | | e.g. checked scaler fitted inside Pipeline only on training folds | |
| 4 | Evaluation / threshold | | | | e.g. verified the threshold was chosen on out-of-fold training data, not test | |
| 5 | Streamlit app / validation | | | | e.g. typed invalid values; compared app vs notebook reference cases | |
| 6 | Deployment / monitoring | | | | e.g. fixed version mismatch error by pinning scikit-learn | |

## Verification practices used (examples to describe in the report)
* Re-running code and comparing outputs with independent calculations (e.g. class counts, power formula).
* Unit tests in `tests/test_predictor.py` (invalid inputs, boundary values, agreement with training script).
* Reading scikit-learn documentation to confirm parameters (`class_weight`, `handle_unknown`, `stratify`).
* Checking that every member can explain each function in `train.py`, `predictor.py` and `app.py`.
