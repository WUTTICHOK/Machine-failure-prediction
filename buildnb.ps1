param($proj, $out)
$u = New-Object Text.UTF8Encoding($false)
$cells = New-Object System.Collections.ArrayList

function AddMd($text) { [void]$cells.Add([ordered]@{ cell_type = "markdown"; metadata = @{}; source = $text }) }
function Code($text) { [void]$cells.Add([ordered]@{ cell_type = "code"; execution_count = $null; metadata = @{}; outputs = @(); source = $text }) }
function WriteFileCell($rel) {
  $content = [IO.File]::ReadAllText("$proj\$($rel.Replace('/', '\'))", $u).TrimEnd() + "`n"
  Code ("%%writefile $rel`n" + $content)
}

AddMd @'
# IFJC4002 – Case 7: Machine Failure Prediction
**Machine Learning Prediction Project – Google Colab version (self-contained)**

This notebook builds the full project with no other files needed:
1. writes all project files (`features.py`, `predictor.py`, `train.py`, `app.py`, tests, README, docs)
2. loads the lecturer-provided dataset **`07_Machine_Failure.xlsx`** (you upload it in step 0)
3. EDA → trains & compares **Baseline / Logistic Regression / Decision Tree / Random Forest** → saves the model
4. reloads the model, runs automated tests, previews the Streamlit app inside Colab
5. zips the whole project for upload to GitHub → Streamlit Community Cloud

**How to run:** `Runtime → Run all`, then choose `07_Machine_Failure.xlsx` when the upload button appears. Training takes about 1–3 minutes.
'@

AddMd "## 0. Setup and dataset upload"
Code @'
import os, platform, shutil
PROJECT = "/content/case7-machine-failure"
for d in ["", "tests", "docs", "data", "models", "reports", "logs"]:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)
%cd {PROJECT}
!pip install -q -U streamlit pytest openpyxl
print("Python", platform.python_version())

DATA = "data/07_Machine_Failure.xlsx"
if not os.path.exists(DATA):
    from google.colab import files
    print("Choose the file 07_Machine_Failure.xlsx")
    uploaded = files.upload()
    name = next(iter(uploaded))
    shutil.move(name, DATA)
print("Dataset ready:", DATA, os.path.getsize(DATA), "bytes")
'@

AddMd @'
## 1. Project files
Each cell below writes one file of the project. `features.py` is shared by training and the dashboard so both use the same feature order and engineered features.
'@
foreach ($f in @("features.py", "predictor.py", "train.py", "app.py", "tests/test_predictor.py")) { WriteFileCell $f }

AddMd "### Group logo (assets/group_logo.png, embedded as base64 so this notebook stays self-contained)"
$logoB64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("$proj\assets\group_logo.png"))
Code ("import base64, os`nos.makedirs(""assets"", exist_ok=True)`nLOGO_B64 = """ + $logoB64 + """`nwith open(""assets/group_logo.png"", ""wb"") as f:`n    f.write(base64.b64decode(LOGO_B64))`nprint(""assets/group_logo.png"", os.path.getsize(""assets/group_logo.png""), ""bytes"")")

AddMd "### Requirements (exact versions of this Colab environment, so Streamlit Cloud loads the model with the same library versions)"
Code @'
import importlib.metadata as md
pkgs = ["streamlit", "scikit-learn", "pandas", "numpy", "joblib", "matplotlib", "openpyxl", "pytest"]
with open("requirements.txt", "w") as f:
    f.write("\n".join(f"{p}=={md.version(p)}" for p in pkgs) + "\n")
print(open("requirements.txt").read())
print("Choose this Python version on Streamlit Cloud (Advanced settings):", platform.python_version())
'@

AddMd "### README and documentation templates"
foreach ($f in @("README.md", ".gitignore", "docs/test_log.md", "docs/prompt_record.md", "docs/contribution_record.md")) { WriteFileCell $f }
Code @'
open("logs/.gitkeep", "w").close()
'@

AddMd "## 2. Train and evaluate all models"
Code "!python train.py"
Code @'
# Stop "Run all" here with a clear message if train.py failed (a failing ! command does not stop Colab by itself)
import os
assert os.path.exists("models/machine_failure_model.joblib"), "train.py did not finish - scroll up to the output of the cell above and read the error (Traceback)."
print("Training finished. Model saved to models/machine_failure_model.joblib")
'@

AddMd "## 3. Dataset analysis (Task 1)"
Code @'
import json, glob
import pandas as pd
from IPython.display import Image, Markdown, display

df = pd.read_excel("data/07_Machine_Failure.xlsx", sheet_name="Data")
display(df.head())
df.info()
display(df.describe().T.round(3))
print("Missing values per column:\n", df.isna().sum(), sep="")
print("\nClass balance:\n", df["failure_next_shift"].value_counts(), "\n", df["failure_next_shift"].value_counts(normalize=True).round(4), sep="")
print("\nMean of each input by class:")
display(df.drop(columns="record_id").groupby("failure_next_shift").mean().T.round(2))
eda = json.load(open("reports/eda_summary.json"))
print("Correlation with target (Failure = 1):", eda["correlation_with_target"])
'@
Code @'
for f in ["eda_class_balance", "eda_missing_values", "eda_distributions", "eda_boxplots", "eda_failure_rate_by_quintile", "eda_correlation"]:
    display(Image(f"reports/figures/{f}.png"))
'@

AddMd "## 4. Model comparison and final model selection (Task 2)"
Code @'
display(pd.read_csv("reports/model_comparison.csv").round(4))
display(Markdown(open("reports/results.md", encoding="utf-8").read()))
'@
Code @'
for f in sorted(glob.glob("reports/figures/cm_*.png")) + ["reports/figures/pr_curves.png", "reports/figures/permutation_importance.png"]:
    display(Image(f))
print("Test-set outcomes by tool wear:"); display(pd.read_csv("reports/errors_by_tool_wear.csv"))
print("Mean inputs by outcome:"); display(pd.read_csv("reports/error_profile.csv"))
display(pd.read_csv("reports/error_analysis.csv").head(20))
'@

AddMd "## 5. Reload the saved model and predict (same code path as the dashboard)"
Code @'
from predictor import load_artifact, validate_inputs, predict
art = load_artifact()
print("Model:", art["model_name"], "| threshold:", art["threshold"])

typical = {"air_temperature_c": 28.0, "process_temperature_c": 72.5, "rotational_speed_rpm": 1700,
           "torque_nm": 50.0, "tool_wear_min": 120, "vibration_mm_s": 2.4, "machine_age_years": 7.5}
for sample in [
    typical,                                                                              # normal
    {**typical, "torque_nm": 82.0, "tool_wear_min": 245, "vibration_mm_s": 5.5, "machine_age_years": 14.0},  # high risk
    {**typical, "air_temperature_c": None},                                                # missing sensor -> imputed
    {**typical, "torque_nm": -5},                                                          # invalid -> rejected
]:
    clean, errors, warnings = validate_inputs(sample, art)
    print("\nInput:", sample)
    print("  errors:", errors, "| warnings:", warnings)
    if not errors:
        print("  ->", predict(art, clean))

display(pd.read_csv("reports/reference_predictions.csv"))
'@

AddMd "## 6. Automated tests (normal, boundary, invalid, notebook–dashboard agreement, monitoring)"
Code "!python -m pytest -v"

AddMd @'
## 7. Download the complete project
Extract the zip, then upload **the contents of** the `case7-machine-failure` folder to GitHub (it must include `models/machine_failure_model.joblib` and `reports/reference_predictions.csv`), then deploy `app.py` on https://share.streamlit.io.

If the download does not start, open the **Files** panel (folder icon on the left) → `/content` → right-click `case7-machine-failure.zip` → **Download**.
'@
Code @'
import os, shutil
PROJECT = "/content/case7-machine-failure"
model_file = os.path.join(PROJECT, "models", "machine_failure_model.joblib")
assert os.path.exists(model_file), "No trained model yet - train.py failed. Send the output of the '!python train.py' cell."
for junk in ["__pycache__", "tests/__pycache__", ".pytest_cache", "streamlit.log"]:
    p = os.path.join(PROJECT, junk)
    if os.path.isdir(p): shutil.rmtree(p)
    elif os.path.exists(p): os.remove(p)
zip_path = shutil.make_archive("/content/case7-machine-failure", "zip", "/content", "case7-machine-failure")
print(f"Model file: {os.path.getsize(model_file) / 1e6:.1f} MB (GitHub web upload limit: 25 MB per file)")
print(f"Zip file:   {os.path.getsize(zip_path) / 1e6:.1f} MB -> {zip_path}")
try:
    from google.colab import files
    files.download(zip_path)
except Exception as exc:
    print("Automatic download failed:", exc, "- download it from the Files panel instead.")
'@

AddMd @'
## 8. Save everything to Google Drive
Copies the whole project (code, data, trained model, reports, figures, docs) and the zip to
`My Drive/IFJC4002_Case7/`. Colab asks you to allow access to Google Drive the first time.
To keep this notebook itself, use **File → Save a copy in Drive**.
'@
Code @'
import os, shutil, datetime
from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/case7-machine-failure"
DRIVE_DIR = "/content/drive/MyDrive/IFJC4002_Case7"
os.makedirs(DRIVE_DIR, exist_ok=True)
shutil.copytree(PROJECT, os.path.join(DRIVE_DIR, "case7-machine-failure"), dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git", "streamlit.log"))
if os.path.exists("/content/case7-machine-failure.zip"):
    shutil.copy("/content/case7-machine-failure.zip", DRIVE_DIR)
with open(os.path.join(DRIVE_DIR, "last_saved.txt"), "w") as f:
    f.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
print("Saved to Google Drive: My Drive/IFJC4002_Case7/")
for root, dirs, fs in os.walk(os.path.join(DRIVE_DIR, "case7-machine-failure")):
    dirs[:] = [d for d in dirs if d != "figures"]
    depth = root.count(os.sep) - DRIVE_DIR.count(os.sep)
    print("  " * depth + os.path.basename(root) + "/", "" if not fs else f"({len(fs)} files)")
'@

AddMd @'
## 9. Push the project to GitHub
Uploads the project to your GitHub repository so Streamlit Community Cloud redeploys automatically.

**One-time setup (do not paste the token into the notebook or the chat):**
1. GitHub → *Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token*.
   *Repository access*: only your project repository. *Permissions → Contents*: **Read and write**. Copy the token.
2. In Colab click the **🔑 key icon (Secrets)** on the left → *Add new secret* → Name `GITHUB_TOKEN`, Value = the token,
   and switch on **Notebook access**.
3. Check `GITHUB_USER`, `GITHUB_REPO` and `BRANCH` below.

The repository is made identical to this project: files that are no longer part of the project (e.g. the old
AI4I files) are removed. Notebooks (`*.ipynb`) already in the repository are kept.
'@
Code @'
import os, subprocess
from google.colab import userdata

GITHUB_USER = "wuttichok"                      # <-- your GitHub user name
GITHUB_REPO = "machine-failure-prediction"     # <-- your repository name
BRANCH = "main"
COMMIT_MESSAGE = "Update Case 7 project (07_Machine_Failure dataset)"

PROJECT = "/content/case7-machine-failure"
REPO_DIR = "/content/github-repo"
assert os.path.exists(f"{PROJECT}/models/machine_failure_model.joblib"), "Run the training cells first."

token = userdata.get("GITHUB_TOKEN")           # read from Colab Secrets, never printed
remote = f"https://{GITHUB_USER}:{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"

def git(*args, cwd=REPO_DIR):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(token, "***")
    if out.strip():
        print(out.strip())
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed (see message above)")

subprocess.run(["rm", "-rf", REPO_DIR])
git("clone", "--quiet", "--branch", BRANCH, remote, REPO_DIR, cwd="/content")
git("config", "user.name", GITHUB_USER)
git("config", "user.email", f"{GITHUB_USER}@users.noreply.github.com")

# Make the repository match the project folder (keep .git and notebooks)
subprocess.run(["rsync", "-a", "--delete", "--exclude", ".git", "--exclude", "*.ipynb",
                "--exclude", "__pycache__", "--exclude", ".pytest_cache", "--exclude", "streamlit.log",
                PROJECT + "/", REPO_DIR + "/"], check=True)

git("add", "-A")
status = subprocess.run(["git", "status", "--short"], cwd=REPO_DIR, capture_output=True, text=True).stdout
if not status.strip():
    print("Nothing to commit - GitHub is already up to date.")
else:
    print("Changes:\n" + status)
    git("commit", "-q", "-m", COMMIT_MESSAGE)
    git("push", "-q", "origin", BRANCH)
    print(f"\nPushed to https://github.com/{GITHUB_USER}/{GITHUB_REPO} (branch {BRANCH}).")
    print("Streamlit Community Cloud will redeploy in about 1-2 minutes.")
'@

AddMd @'
## 10. Preview the Streamlit dashboard inside Colab (optional)
Opens the app through Colab's own port proxy. For the real submission, deploy to Streamlit Community Cloud (see README).
'@
Code @'
import subprocess, time
os.chdir(PROJECT)
streamlit_proc = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true",
     "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
    stdout=open("/content/streamlit.log", "w"), stderr=subprocess.STDOUT)
time.sleep(10)
print(open("/content/streamlit.log").read())
from google.colab import output
output.serve_kernel_port_as_window(8501)
'@

$nb = [ordered]@{
  cells = $cells
  metadata = [ordered]@{
    colab = @{ provenance = @(); name = "Case7_Machine_Failure_Colab.ipynb" }
    kernelspec = @{ display_name = "Python 3"; language = "python"; name = "python3" }
    language_info = @{ name = "python" }
  }
  nbformat = 4
  nbformat_minor = 0
}
$json = $nb | ConvertTo-Json -Depth 20
[IO.File]::WriteAllText($out, $json, $u)
