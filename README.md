---
title: Agentic Regression Analysis
emoji: 📈
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---

# 📈 Regression Crew

End-to-end agentic regression analysis built with **CrewAI** and **Gradio**.

Upload one CSV (or three: train / val / test) → the pipeline profiles and cleans the
data, trains and tunes up to 15 models plus a diverse ensemble **on one shared set of
folds with preprocessing fitted inside every fold**, diagnoses and fixes over/underfitting,
and picks a winner with a paired statistical rule. If nothing beats predicting the mean,
it says so. An optional advisor proposes improvements and keeps only those that
cross-validation confirms; 8 agents explain the results; you download a script, a
notebook and a model file that reproduce the run exactly.

---

## 🎯 How it chooses a model honestly

| Stage | What happens |
| --- | --- |
| **Same folds for everyone** | Folds are built once; every model (tuned or not, and the baseline) is scored on exactly those folds, so comparisons are *paired* |
| **No leakage into CV** | Imputation, scaling and target encoding are fitted inside each training fold |
| **Tuning ≠ scoring** | Hyperparameter search uses reshuffled folds; the reported CV score comes from the shared folds (optional nested CV for fully unbiased estimates) |
| **Relative diagnosis** | Each model is labelled *good*, *harmless gap*, *overfit*, *underfit*, *unstable* or *low-signal* from training CV only. "Underfit" requires that another model (or a flexible reference model) does clearly better; if nothing does, the data are *low-signal* and no capacity is added |
| **Fixes must be real** | A remedy is kept only if it lowers CV error by more than the corrected fold-to-fold noise (Nadeau–Bengio corrected resampled t-test) |
| **Diverse ensemble** | Averages the best model of three *different* families, scored from stored fold predictions |
| **Selection** | Lowest CV (or validation) error, with a paired one-standard-error rule preferring simpler models; **the baseline wins when the best model is not reliably better than the mean** |
| **Failures are visible** | A model whose CV fails is excluded and labelled — selection never silently switches to the test set |

## 🤖 Models

Baseline (mean, in original units), Linear, Ridge, Lasso, ElasticNet, Huber, Poisson (non-negative targets), Decision Tree, Random Forest,
Extra Trees, Gradient Boosting, HistGradientBoosting, KNN, SVR, XGBoost, LightGBM (+ ensemble).
XGBoost and LightGBM use **early stopping** on an internal validation slice (the most recent rows
in time-ordered data); in time mode HistGradientBoosting does too, and Gradient Boosting uses a
fixed budget because sklearn's internal validation split is random.

## ⚙️ Main options (Setup tab)

- **Split:** random, time-aware (sorts every file by the time column; a time split is always
  validated with time-series CV), or group-aware (no group on both sides)
- **Target transform:** *auto* compares raw vs `log1p` by cross-validation in original units;
  log-target predictions get a smearing bias correction when that lowers out-of-fold RMSE
- **Categoricals:** cross-fitted target encoding for > 20 levels (or frequency encoding);
  "treat as categorical" for integer codes; "drop columns" for suspected leakage
- **Tuning:** Off / Fast / Thorough; **selection metric:** auto (MAE when the target has heavy
  outliers), RMSE or MAE
- **Saved model:** fitted on training rows, training + validation rows (default when a validation
  file is given), or all labelled rows
- **Auto-refine:** the advisor proposes whitelisted changes (tuning, interactions, encodings,
  target transform, categorical overrides; optionally LLM-suggested), re-runs, and keeps a change
  only if the winner's CV error improves beyond noise. Column drops are never automatic — they are
  listed for your confirmation

## 📊 Charts and audits

Predicted vs actual, residuals, residual distribution, Q-Q, CV box (fold-consistent R²), feature
importance (held-out permutation per **original** column, in original units, for the winner),
learning curve, adaptive conformal prediction intervals. The quality review checks for target
leakage (single-feature model R² and rank correlation, so non-linear leaks such as `log(target)`
are caught), rows or groups shared between your train and test files, time-order violations,
kept-but-suspicious ID-like columns, fit problems and interval coverage.

## 📁 Generated artifacts

Each run writes to its own temporary directory (concurrent users never share files):

- `regression_pipeline.py` / `.ipynb` — reproduce the run by calling **the same library code as
  the app** with the exact options and winner settings the app used, then compare the test
  metrics with the app's (they match exactly). Needs this project's `utils/` importable: run it
  from the project folder or set `REGRESSION_CREW_DIR`
- `best_model.joblib` — `format_version: 3`: a raw-row feature builder + the fitted model
  pipeline (encoding + model), smearing factor, interval model, metrics

```python
import joblib, pandas as pd
from utils.io import read_table
from utils.modeling import predict_with_interval

bundle = joblib.load("best_model.joblib")
pred, low, high = predict_with_interval(bundle, read_table("new_rows.csv"))
```

---

## 🚀 Quick start

```bash
git clone https://github.com/navid015/Agentic_Regression_Analysis_8_Agents_1_Pipeline.git
cd Agentic_Regression_Analysis_8_Agents_1_Pipeline
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip && pip install -r requirements.txt

# optional: agent narration / LLM advisor
export OPENAI_API_KEY="sk-..."          # model: $OPENAI_MODEL (default gpt-4o-mini)
# or: export ANTHROPIC_API_KEY="..."   # model: $ANTHROPIC_MODEL (default anthropic/claude-sonnet-4-6)
python app.py                           # http://localhost:7860
```

Tests: `pip install pytest && python -m pytest -q` (83 tests, about a minute).

## 🧩 How it works

Deterministic utilities own execution; the advisor owns *proposals* that cross-validation
accepts or rejects; agents own narration. Agent tools are built **per run** from that run's
output, so every agent reports the same numbers as the app, and two users never see each
other's results. Without an LLM key everything except narration and LLM proposals still runs.

## 📁 Project structure

```
├── app.py                    # Gradio UI
├── crew/
│   ├── orchestrator.py       # runs everything; per-run output directory
│   ├── advisor.py            # propose -> re-run -> keep only CV-confirmed improvements
│   ├── tools.py              # per-run report functions + CrewAI tools
│   ├── agents.py / tasks.py  # 8 agents, each grounded in tools
├── utils/
│   ├── io.py                 # encoding- and delimiter-aware CSV reading
│   ├── preprocessing.py      # cleaning, typing, splits, encoders, time features
│   ├── modeling.py           # zoo, shared folds, tuning, diagnosis, selection, intervals
│   ├── diagnostics.py        # leakage audit, residual diagnostics
│   ├── visualization.py      # Plotly charts
│   └── code_generator.py     # drift-free .py / .ipynb
└── tests/                    # 83 tests
```

## ⚠️ Scope

- **Good for:** tabular regression — ordinary, skewed, count and outlier-heavy targets; grouped
  rows; time-ordered data with a trend; small and wide datasets
- **Caution:** true forecasting (no lag features or seasonality models — use sktime / Prophet);
  tiny datasets (< 50 rows); nested CV, Thorough tuning and auto-refine multiply run time
- **Not designed for:** classification, multi-output, quantile or survival regression,
  images / text / audio, deployment infrastructure, causal inference
