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

Upload one CSV (or three: train / val / test) → 8 specialized agents plan, profile, preprocess, train, visualize, audit, and report → explore models in a UI with a per-model dropdown and full comparison views → download the standalone Python script and Jupyter notebook that reproduces everything.

---


## 🤖 8 CrewAI agents

| Agent | Role |
| --- | --- |
| 🗺️ Planner | Sequences the project |
| 🔬 EDA Analyst | Profiles dataset, names risky columns with percentages |
| 🧹 Preprocessor | Imputes, encodes, scales, splits |
| 🤖 Modeler & Evaluator | Trains models, ranks them, calls out overfitting |
| 📊 Visualization | Confirms chart inventory |
| 🧐 Quality Reviewer | Audits for leakage / overfitting → GO / CAUTION / NO-GO |
| 💻 Code Generator | Describes the reproducible artifacts |
| 📝 Insight Reporter | Plain-language executive summary |

## 🤖 Up to 11 regression models + a baseline

LinearRegression · Ridge · Lasso · ElasticNet · DecisionTree · RandomForest · GradientBoosting · KNN · SVR · *(optional)* XGBoost · LightGBM

A **mean-prediction baseline** is always trained alongside them. It is never selected as the winner — it exists so every other score is interpretable (“is this model actually better than doing nothing?”) and so a negative R² is obvious rather than mysterious.

## 📊 Charts

**Per model** (selected via dropdown):
Predicted vs Actual · Residuals vs Predicted · Residual Distribution · Q-Q Plot · CV R² Box · Feature Importance

**Comparison** across all models:
RMSE / MAE / R² bars · grouped error metrics · CV R² box · training time · predicted-vs-actual overlay

## 📁 Generated artifacts

All three are downloadable from the **Code** tab:

- `regression_pipeline.py` — single-file reproducible script
- `regression_pipeline.ipynb` — same pipeline as a Jupyter notebook
- `best_model.joblib` — fitted preprocessor + winning model in one object

The bundle is genuinely self-contained: every learned mapping (imputation,
scaling, one-hot **and** frequency encoding) lives inside the preprocessor, so
it transforms raw data without any extra state.

```python
import joblib, numpy as np, pandas as pd

bundle = joblib.load("best_model.joblib")
X = bundle["preprocessor"].transform(pd.read_csv("new_rows.csv"))
pred = bundle["model"].predict(X)
if bundle["target_transform"] == "log1p":
    pred = np.expm1(pred)
```

---

## 🚀 Quick start

```bash
unzip regression_crew.zip
cd regression_crew

# Python 3.10+
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# (optional) enable XGBoost & LightGBM
pip install xgboost lightgbm

# (optional) configure an LLM for agent narration
cp .env.example .env
# edit .env to set OPENAI_API_KEY or ANTHROPIC_API_KEY

# run
python app.py
```

Open `http://localhost:7860`.

---

## ✅ Correctness guarantees

The pipeline is written to avoid the leakage and evaluation mistakes that make
AutoML output look better than it is:

| Guard | What it prevents |
| --- | --- |
| Whole-token identifier detection | Deleting real features (`fixed_acidity`, `humidity`, `width`) while still catching `Unnamed: 0` and `customer_id` |
| Row-index / ID auto-drop | A leftover counter on a target-sorted CSV scoring R² ≈ 1.0 on nothing |
| Group-aware **split** and CV | The same patient / store appearing on both sides of the split |
| Time-aware split (refuses to run without a time column) | Training on the future to predict the past |
| Exact-duplicate removal before splitting | Identical rows straddling the split, leaking the answer |
| `fit` on train, `transform` elsewhere | Test statistics informing imputation, scaling or encoding |
| Validation-based model selection | The test set doubling as a selection set |
| Original-unit CV scorers under `log1p` | Comparing a log-scale CV R² against an original-scale test R² |
| Two-signal leakage audit | Crying wolf on datasets that are simply very predictable |

Run the suite with:

```bash
pip install pytest
pytest -q
```

---

## 🧩 How it works

The system separates **what to do** (agents) from **how to do it** (deterministic utilities):

- **Agents own strategy and narration.** They decide which steps to run and explain results.
- **Utilities own execution.** Preprocessing, training, evaluation, and chart generation are pure Python — fast, reproducible, no LLM in the hot path.
- **CrewAI tools** are the bridge: each tool wraps one utility function.

Whether you set an LLM key or not, you'll get the same numbers from the same data.

---

## 📁 Project structure

```
regression_crew/
├── app.py                       # Gradio UI
├── requirements.txt
├── .env.example
├── README.md
├── crew/
│   ├── __init__.py
│   ├── agents.py                # 8 CrewAI agents
│   ├── tasks.py                 # Task descriptions with concrete expected outputs
│   ├── tools.py                 # CrewAI tools wrapping the utilities
│   └── orchestrator.py          # Main runner
├── tests/
│   └── test_pipeline.py      # leakage / metric / selection regression tests
└── utils/
    ├── __init__.py
    ├── preprocessing.py         # Multi-file, datetime, log-transform, time/group splits
    ├── modeling.py              # Up to 11 models, CV strategies, optional tuning
    ├── visualization.py         # Plotly per-model + comparison charts
    └── code_generator.py        # Emits .py and .ipynb that reflect every option
```

---

## ⚠️ Scope

See the **Scope & Disclaimer** tab inside the app for the full breakdown. Quick version:

- **Good for:** standard tabular regression with independent rows
- **Caution for:** time series (use Prophet / sktime), grouped data (use the group-CV option), heavily skewed targets (use log-transform option), tiny datasets (< 50 rows)
- **Not designed for:** classification, multi-output, survival, quantile regression, images / text / audio, deployment infrastructure, causal inference
