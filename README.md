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

Upload one CSV (or three: train / val / test) → the pipeline profiles, cleans, trains
and tunes up to 15 models, **diagnoses every model for over- and underfitting, fixes
the ones that need it**, builds an ensemble and picks a winner → 8 agents explain the
results → download a script, notebook and model file that reproduce the exact winner.

---

## 🎯 How it guards against overfitting and underfitting

| Stage | What happens |
| --- | --- |
| **Prevent** | Regularised defaults (leaf sizes, depth limits, early stopping for boosting); target scaling for SVR and penalised linear models; hyperparameter tuning (Fast by default) over the knobs that control capacity; automatic log transform for skewed targets; optional interaction features for linear models that underfit |
| **Detect** | Every model gets a label — *good*, *overfit*, *underfit*, *unstable*, or *harmless gap* — computed from cross-validation on the **training rows only** (train-fold vs held-out-fold score, error vs a mean baseline, fold-to-fold spread). The test set is never used for a decision |
| **Fix** | Over- or underfitting models are retrained with more / less regularisation; the change is kept only if cross-validated error improves by ≥ 1 % |
| **Choose** | Winner by validation file (if supplied) or cross-validated error, with the **one-standard-error rule** preferring the simplest model that is statistically as good; a top-3 averaging ensemble competes too; the metric switches to MAE automatically when the target has heavy outliers |
| **Check** | Learning curve for the winner (train vs held-out score as data grows, with a plain-English reading) and a 90 % conformal prediction interval whose coverage is verified on the test set |

## 🤖 Models

Baseline (mean) · LinearRegression · Ridge · Lasso · ElasticNet · **Huber** (outliers) ·
**PoissonRegressor** (counts / non-negative targets) · DecisionTree · RandomForest ·
**ExtraTrees** · GradientBoosting · **HistGradientBoosting** · KNN · SVR · XGBoost · LightGBM ·
**Ensemble (top-3 average)**

Models that cannot work on a dataset are skipped with a reason (Poisson on negative
targets, SVR above 15,000 rows). A model that crashes is reported and left out instead
of stopping the run.

## ⚙️ Main options (Setup tab)

| Option | Default | Notes |
| --- | --- | --- |
| Hyperparameter tuning | Fast (10 trials/model) | Off · Fast · Thorough (40) |
| Selection metric | Auto | RMSE, or MAE when > 2 % of targets are extreme outliers |
| Automatic over/underfitting fixes | On | |
| Top-3 ensemble | On | |
| One-standard-error rule | On | |
| Interaction features | Off | pairwise products of up to 15 numeric columns |
| Nested CV for tuned models | Off | honest CV scores for tuned models; much slower |
| Log-transform target | Auto | Off · Auto (skew > 1.5, non-negative, not a count) · On |

## 📊 Charts

**Per model:** Predicted vs Actual · Residuals vs Predicted · Residual Distribution ·
Q-Q Plot · CV R² Box · Feature Importance (held-out permutation importance for the winner)
· **Learning Curve** (winner)

**Comparison:** RMSE / MAE / R² bars · grouped error metrics · CV R² box · training time ·
predicted-vs-actual overlay. The metrics table shows each model's fit label, cross-validated
error, skill vs the mean, tuned settings and any automatic fix.

## 📁 Generated artifacts

- `regression_pipeline.py` — standalone script; rebuilds the app's **exact** winner
  (tuned / remediated settings, or the ensemble members) and compares it with the zoo
- `regression_pipeline.ipynb` — the same, as a notebook
- `best_model.joblib` — fitted preprocessor + winner, plus its metrics, fit label and
  prediction interval

```python
import joblib, pandas as pd
from utils.modeling import predict_with_interval

bundle = joblib.load("best_model.joblib")
pred, low, high = predict_with_interval(bundle, pd.read_csv("new_rows.csv"))
```

The bundle's preprocessor accepts **raw rows** (date columns included), but it uses
custom transformers from `utils/preprocessing.py`, so load it where this project's
`utils` folder is importable. Manual equivalent: `X = bundle["preprocessor"].transform(df)`,
`p = bundle["model"].predict(X)`, then `np.expm1(p)` if `bundle["target_transform"] == "log1p"`.

---

## 🚀 Quick start

```bash
git clone https://github.com/navid015/Agentic_Regression_Analysis_8_Agents_1_Pipeline.git
cd Agentic_Regression_Analysis_8_Agents_1_Pipeline

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# optional: enable agent narration (set in the same terminal before starting)
export OPENAI_API_KEY="sk-..."     # or ANTHROPIC_API_KEY=...
                                   # PowerShell: $env:OPENAI_API_KEY="sk-..."
python app.py                      # open http://localhost:7860
```

Tests (run from the project folder):

```bash
pip install pytest
python -m pytest -q                # 59 tests
```

---

## ✅ Correctness guarantees

| Guard | What it prevents |
| --- | --- |
| Whole-token identifier detection | Deleting real features (`fixed_acidity`, `humidity`) while catching `Unnamed: 0`, `customer_id` |
| Row-index / ID auto-drop | A leftover counter on a target-sorted CSV scoring R² ≈ 1.0 |
| Group-aware split and CV | The same patient / store on both sides of the split |
| Time-aware split with trend features | Training on the future; losing the trend (elapsed time is kept; calendar parts only when the training period covers two full cycles) |
| Exact-duplicate removal before splitting | Identical rows straddling the split |
| `fit` on train, `transform` elsewhere | Test statistics leaking into imputation, scaling or encoding |
| Decisions from training CV / validation only | The test set doubling as a selection set |
| Fold-consistent R² for diagnosis | Misleading per-fold R² on short time-series folds |
| Original-unit scorers under `log1p` | Comparing log-scale CV scores with original-scale test scores |
| Two-signal leakage audit | Crying wolf on datasets that are simply very predictable |

## 🧩 How it works

Agents own **strategy and narration**; deterministic utilities own **execution**. CrewAI
tools wrap the utilities and read results from a shared state, so every agent reports the
same numbers — and the same winner — as the app. You get identical numbers with or
without an LLM key.

## 📁 Project structure

```
├── app.py                       # Gradio UI
├── requirements.txt
├── crew/
│   ├── agents.py                # 8 CrewAI agents
│   ├── tasks.py                 # task descriptions
│   ├── tools.py                 # tools + quality audit (leakage, fit labels, outliers, coverage)
│   └── orchestrator.py          # runs everything
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py         # leakage / metric / selection tests
│   └── test_fit_quality.py      # over/underfitting prevention, detection, fixing
└── utils/
    ├── preprocessing.py         # cleaning, splits, auto log, interactions, time features
    ├── modeling.py              # zoo, tuning, diagnosis, remediation, ensemble, selection
    ├── visualization.py         # Plotly charts incl. learning curve
    └── code_generator.py        # .py / .ipynb that rebuild the exact winner
```

## ⚠️ Scope

- **Good for:** tabular regression — ordinary, skewed, count and outlier-heavy targets;
  grouped rows; time-ordered data with a trend; small and wide datasets
- **Caution:** true forecasting (lags, seasonality — use sktime / Prophet); tiny datasets
  (< 50 rows); tuning on very large data is slow (turn it off or use Fast)
- **Not designed for:** classification, multi-output targets, quantile or survival
  regression, images / text / audio, deployment infrastructure, causal inference
