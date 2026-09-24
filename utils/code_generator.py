"""
Generate a self-contained Python script and Jupyter notebook that
reproduce the entire pipeline, reflecting every option chosen in the UI.
"""

from __future__ import annotations

import json
import re
from typing import Any


# Constructor text for every model in the zoo, matching utils/modeling.py.
_MODEL_TEMPLATES = {
    "Baseline (mean)":  "DummyRegressor(strategy='mean')",
    "LinearRegression": "LinearRegression()",
    "Ridge":            "scaled(Ridge(alpha=1.0, random_state=RANDOM_STATE))",
    "Lasso":            "scaled(Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=20000))",
    "ElasticNet":       "scaled(ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=20000))",
    "Huber":            "scaled(HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000))",
    "PoissonRegressor": "PoissonRegressor(alpha=1e-3, max_iter=2000)",
    "DecisionTree":     "DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE)",
    "RandomForest":     "RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)",
    "ExtraTrees":       "ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)",
    "GradientBoosting": "GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=3, subsample=0.8, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)",
    "HistGradientBoosting": "HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=RANDOM_STATE)",
    "KNN":              "KNeighborsRegressor(n_neighbors=7, n_jobs=-1)",
    "SVR":              "scaled(SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'))",
    "XGBoost":          "XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)",
    "LightGBM":         "LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)",
}


def _model_construction_lines(model_names: list[str], use_xgb: bool = False,
                              use_lgbm: bool = False) -> str:
    return "\n".join(f'    "{n}": {_MODEL_TEMPLATES[n]},'
                     for n in model_names if n in _MODEL_TEMPLATES)


def _with_params_expr(name: str, params: dict) -> str:
    """Python text that rebuilds a model with its exact final settings."""
    base = _MODEL_TEMPLATES[name]
    if base.startswith("scaled("):
        inner = base[len("scaled("):-1]
        return f"scaled({inner}.set_params(**{params!r}))"
    return f"{base}.set_params(**{params!r})"


def _winner_block(winner_spec: dict | None) -> str:
    if not winner_spec:
        return "WINNER_NAME = None\nWINNER = None"
    name = winner_spec["name"]
    if winner_spec.get("members"):
        members = ",\n".join(
            f"    ({re.sub(r'[^A-Za-z0-9]+', '_', m['name']).strip('_')!r}, "
            f"{_with_params_expr(m['name'], m['params'])})"
            for m in winner_spec["members"] if m["name"] in _MODEL_TEMPLATES)
        return f"WINNER_NAME = {name!r}\nWINNER = VotingRegressor([\n{members},\n])"
    if name not in _MODEL_TEMPLATES:
        return "WINNER_NAME = None\nWINNER = None"
    return f"WINNER_NAME = {name!r}\nWINNER = {_with_params_expr(name, winner_spec.get('params', {}))}"


def _imports_block(model_names: list[str]) -> str:
    extra = []
    if "XGBoost"  in model_names: extra.append("from xgboost import XGBRegressor")
    if "LightGBM" in model_names: extra.append("from lightgbm import LGBMRegressor")
    return "\n".join(extra)


def generate_python_script(
    *,
    target: str,
    model_names: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    cv_strategy: str = "kfold",
    csv_path: str = "your_dataset.csv",
    val_csv_path: str | None = None,
    test_csv_path: str | None = None,
    log_transform_target: bool = False,
    auto_datetime_features: bool = True,
    drop_low_variance: bool = True,
    auto_drop_id_columns: bool = True,
    drop_duplicate_rows: bool = True,
    time_column: str | None = None,
    group_column: str | None = None,
    summary: dict[str, Any] | None = None,
    add_missing_indicators: bool = False,
    add_interactions: bool = False,
    cv_repr: str | None = None,
    selection_metric: str = "rmse",
    selection_note: str = "",
    winner_spec: dict | None = None,
    interval: dict | None = None,
) -> str:
    summary_str = json.dumps(summary or {}, indent=2, default=str)
    names_for_imports = list(model_names)
    if winner_spec and winner_spec.get("members"):
        names_for_imports += [m["name"] for m in winner_spec["members"]]
    winner_block = _winner_block(winner_spec)
    interval = interval or {}
    models_block = _model_construction_lines(
        model_names, use_xgb="XGBoost" in model_names, use_lgbm="LightGBM" in model_names
    )
    extra_imports = _imports_block(names_for_imports)
    multi_file = test_csv_path is not None
    has_val = val_csv_path is not None
    log_target = log_transform_target

    cv_import_extra = "KFold, RepeatedKFold, GroupKFold, TimeSeriesSplit, GroupShuffleSplit"
    if cv_repr:
        cv_constructor = cv_repr
    elif cv_strategy == "time":
        cv_constructor = f"TimeSeriesSplit(n_splits={cv_folds})"
    elif cv_strategy == "group":
        cv_constructor = f"GroupKFold(n_splits={cv_folds})"
    else:
        cv_constructor = f"KFold(n_splits={cv_folds}, shuffle=True, random_state=RANDOM_STATE)"

    script = f'''"""
End-to-end regression pipeline (auto-generated by Regression Crew).

Target: {target!r}
Models: {model_names}

Pipeline summary from the original UI run:
{summary_str}
"""
import warnings
warnings.filterwarnings("ignore")

import re
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate, {cv_import_extra}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import (ElasticNet, HuberRegressor, Lasso, LinearRegression,
                                  PoissonRegressor, Ridge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (make_scorer, mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)
{extra_imports}

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TRAIN_CSV    = {csv_path!r}
VAL_CSV      = {val_csv_path!r}
TEST_CSV     = {test_csv_path!r}
TARGET       = {target!r}
TEST_SIZE    = {test_size}
RANDOM_STATE = {random_state}
N_FOLDS      = {cv_folds}
LOG_TARGET   = {log_target}
AUTO_DATETIME_FEATURES = {auto_datetime_features}
DROP_LOW_VARIANCE      = {drop_low_variance}
AUTO_DROP_ID_COLUMNS   = {auto_drop_id_columns}
DROP_DUPLICATE_ROWS    = {drop_duplicate_rows}
TIME_COLUMN  = {time_column!r}
GROUP_COLUMN = {group_column!r}
ADD_MISSING_INDICATORS = {add_missing_indicators}
ADD_INTERACTIONS = {add_interactions}
SELECTION_METRIC = {selection_metric!r}       # "rmse" or "mae"
HIGH_CARDINALITY_THRESHOLD = 20
# 90% conformal prediction interval half-width measured by the app
# (in {interval.get("space", "original")!r} space); 0 means "not computed".
INTERVAL_HALFWIDTH = {interval.get("halfwidth", 0.0)!r}
# Whole-token identifier names. Matched against TOKENS, never as substrings:
# a substring test flags (and deletes) fixed_acidity, humidity, width, solidity.
ID_COLUMN_NAME_TOKENS = {{"id", "ids", "index", "idx", "rowid", "row", "rownum",
                         "rownumber", "key", "pk", "uuid", "guid", "serial", "seq"}}

# ---------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)  if VAL_CSV  else None
df_test  = pd.read_csv(TEST_CSV) if TEST_CSV else None
print(f"Train: {{df_train.shape}}")
if df_val  is not None: print(f"Val:   {{df_val.shape}}")
if df_test is not None: print(f"Test:  {{df_test.shape}}")

# replace +/-inf with NaN (SimpleImputer does NOT treat inf as missing, so an
# infinity survives imputation and turns a whole scaled column into NaN)
def sanitize_inf(frame):
    num = frame.select_dtypes(include=[np.number]).columns
    if len(num): frame = frame.copy(); frame[num] = frame[num].replace([np.inf, -np.inf], np.nan)
    return frame

df_train = sanitize_inf(df_train)
if df_val  is not None: df_val  = sanitize_inf(df_val)
if df_test is not None: df_test = sanitize_inf(df_test)

# drop rows where target is missing
df_train = df_train.dropna(subset=[TARGET])
if df_val  is not None: df_val  = df_val.dropna(subset=[TARGET])
if df_test is not None: df_test = df_test.dropna(subset=[TARGET])
assert pd.api.types.is_numeric_dtype(df_train[TARGET]), "Target must be numeric."

# drop exact duplicate rows BEFORE splitting - identical rows on both sides of
# the split mean the model has literally already seen the answer
if DROP_DUPLICATE_ROWS and df_test is None:
    _before = len(df_train)
    df_train = df_train.drop_duplicates().reset_index(drop=True)
    if _before != len(df_train):
        print(f"Dropped {{_before - len(df_train)}} exact duplicate rows")

# ---------------------------------------------------------------------
# 1.5 DROP ID-LIKE COLUMNS (e.g. pandas' "Unnamed: 0" — leaks when sorted)
# ---------------------------------------------------------------------
def all_unique(s):
    non_null = int(s.notna().sum())
    return non_null > 0 and int(s.nunique(dropna=True)) == non_null

def name_suggests_id(col):
    lowered = str(col).lower().strip()
    if lowered.startswith("unnamed:"): return True
    tokens = {{t for t in re.split(r"[^a-z0-9]+", lowered) if t}}
    return bool(tokens & ID_COLUMN_NAME_TOKENS)

def is_id_like(s):
    if not pd.api.types.is_numeric_dtype(s): return False
    if not all_unique(s): return False
    arr = s.dropna().to_numpy()
    if not np.all(arr == arr.astype(np.int64)): return False
    return s.is_monotonic_increasing or s.is_monotonic_decreasing

def looks_like_identifier(col, s):
    # A name hint alone is never enough to delete a column - the values must
    # actually behave like identifiers (all distinct).
    return is_id_like(s) or (name_suggests_id(col) and all_unique(s))

if AUTO_DROP_ID_COLUMNS:
    id_cols = []
    for col in df_train.columns:
        if col == TARGET or col == TIME_COLUMN or col == GROUP_COLUMN: continue
        if looks_like_identifier(col, df_train[col]):
            id_cols.append(col)
    if id_cols:
        df_train = df_train.drop(columns=id_cols)
        if df_val  is not None: df_val  = df_val.drop( columns=[c for c in id_cols if c in df_val.columns])
        if df_test is not None: df_test = df_test.drop(columns=[c for c in id_cols if c in df_test.columns])
        print(f"Dropped ID/index columns: {{id_cols}}")

# ---------------------------------------------------------------------
# 2. AUTO DATETIME FEATURE EXTRACTION
# ---------------------------------------------------------------------
def extract_datetime(frame, cols, hour_cols=None):
    # `hour_cols`, when given, forces hour-feature inclusion for exactly those
    # columns instead of deciding per-frame — deciding independently per split
    # can add "<col>_hour" to train's schema but not test's (or vice versa),
    # which then crashes ColumnTransformer.transform with a missing-column error.
    out = frame.copy()
    used_hour_cols = set()
    for col in cols:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        out[f"{{col}}_year"]    = parsed.dt.year
        out[f"{{col}}_month"]   = parsed.dt.month
        out[f"{{col}}_day"]     = parsed.dt.day
        out[f"{{col}}_weekday"] = parsed.dt.weekday
        include_hour = (col in hour_cols) if hour_cols is not None \
            else (parsed.dt.hour.fillna(0).sum() > 0)
        if include_hour:
            out[f"{{col}}_hour"] = parsed.dt.hour
            used_hour_cols.add(col)
        out = out.drop(columns=[col])
    return out, used_hour_cols

datetime_cols = []
if AUTO_DATETIME_FEATURES:
    for col in df_train.columns:
        # the time column must stay intact: the split sorts on it (like the app)
        if col in (TARGET, TIME_COLUMN): continue
        s = df_train[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().sum() / max(s.notna().sum(), 1) > 0.85:
                datetime_cols.append(col)
    if datetime_cols:
        df_train, datetime_hour_cols = extract_datetime(df_train, datetime_cols)
        if df_val  is not None: df_val,  _ = extract_datetime(df_val,  datetime_cols, hour_cols=datetime_hour_cols)
        if df_test is not None: df_test, _ = extract_datetime(df_test, datetime_cols, hour_cols=datetime_hour_cols)
        print(f"Extracted datetime features from: {{datetime_cols}}")

# ---------------------------------------------------------------------
# 3. DROP LOW-VARIANCE COLUMNS
# ---------------------------------------------------------------------
if DROP_LOW_VARIANCE:
    def low_variance(s):
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1: return True
        return s.value_counts(dropna=True).iloc[0] / len(s) >= 0.999
    drop = [c for c in df_train.columns
            if c not in (TARGET, TIME_COLUMN, GROUP_COLUMN) and low_variance(df_train[c])]
    if drop:
        df_train = df_train.drop(columns=drop)
        if df_val  is not None: df_val  = df_val.drop( columns=[c for c in drop if c in df_val.columns])
        if df_test is not None: df_test = df_test.drop(columns=[c for c in drop if c in df_test.columns])
        print(f"Dropped low-variance columns: {{drop}}")

# ---------------------------------------------------------------------
# 4. SPLIT
# ---------------------------------------------------------------------
if df_test is not None:
    train_df, test_df = df_train, df_test
    val_df = df_val
elif TIME_COLUMN and TIME_COLUMN in df_train.columns:
    sorted_df = df_train.sort_values(TIME_COLUMN).reset_index(drop=True)
    cut = int(len(sorted_df) * (1 - TEST_SIZE))
    train_df, test_df = sorted_df.iloc[:cut], sorted_df.iloc[cut:]
    val_df = None
    print(f"Time-aware split on '{{TIME_COLUMN}}'")
elif GROUP_COLUMN and GROUP_COLUMN in df_train.columns:
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(df_train, groups=df_train[GROUP_COLUMN].to_numpy()))
    train_df, test_df = df_train.iloc[tr_idx], df_train.iloc[te_idx]
    val_df = None
    print(f"Group-aware split on '{{GROUP_COLUMN}}'")
else:
    train_df, test_df = train_test_split(df_train, test_size=TEST_SIZE,
                                         random_state=RANDOM_STATE)
    val_df = None

groups_train = None
if GROUP_COLUMN and GROUP_COLUMN in train_df.columns:
    groups_train = train_df[GROUP_COLUMN].values
    train_df = train_df.drop(columns=[GROUP_COLUMN])
    if test_df is not None and GROUP_COLUMN in test_df.columns:
        test_df = test_df.drop(columns=[GROUP_COLUMN])
    if val_df is not None and GROUP_COLUMN in val_df.columns:
        val_df = val_df.drop(columns=[GROUP_COLUMN])

# turn the split's time column into features: elapsed days (the trend) plus the
# calendar parts that repeat at least twice inside the training period
if TIME_COLUMN and TIME_COLUMN in train_df.columns and \
        not pd.api.types.is_numeric_dtype(train_df[TIME_COLUMN]):
    _p = pd.to_datetime(train_df[TIME_COLUMN], errors="coerce")
    _t0 = _p.min(); _span = (_p.max() - _t0).total_seconds() / 86400.0
    _parts = {{"weekday": _span >= 14, "day": _span >= 60, "month": _span >= 730,
              "hour": bool(_p.dt.hour.fillna(0).sum() > 0) and _span >= 2}}
    _frames = []
    for _fr in [train_df, test_df] + ([val_df] if val_df is not None else []):
        _fr = _fr.copy(); _q = pd.to_datetime(_fr[TIME_COLUMN], errors="coerce")
        _fr[f"{{TIME_COLUMN}}_elapsed_days"] = (_q - _t0).dt.total_seconds() / 86400.0
        for _part, _keep in _parts.items():
            if _keep: _fr[f"{{TIME_COLUMN}}_{{_part}}"] = getattr(_q.dt, _part)
        _frames.append(_fr.drop(columns=[TIME_COLUMN]))
    train_df, test_df = _frames[0], _frames[1]
    if val_df is not None: val_df = _frames[2]

y_train = train_df[TARGET].values.astype(float)
y_test  = test_df [TARGET].values.astype(float)
y_val   = val_df  [TARGET].values.astype(float) if val_df is not None else None
X_train = train_df.drop(columns=[TARGET])
X_test  = test_df .drop(columns=[TARGET])
X_val   = val_df  .drop(columns=[TARGET]) if val_df is not None else None

# ---------------------------------------------------------------------
# 5. ENCODE / IMPUTE / SCALE
# ---------------------------------------------------------------------
numeric_cols, low_card_cat, high_card_cat = [], [], []
for col in X_train.columns:
    if pd.api.types.is_numeric_dtype(X_train[col]):
        numeric_cols.append(col)
    elif X_train[col].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD:
        high_card_cat.append(col)
    else:
        low_card_cat.append(col)

# Frequency encoding lives INSIDE the pipeline so the fitted lookups are
# pickled with the preprocessor. Kept as a loose loop, the mapping was never
# saved and best_model.joblib could not transform new raw data on its own.
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.lookups_ = {{c: X[c].astype(object).value_counts(normalize=True)
                         for c in X.columns}}
        return self
    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = pd.DataFrame(index=X.index)
        for c in self.feature_names_in_:
            out[c] = X[c].astype(object).map(self.lookups_[c]).astype("float64").fillna(0.0)
        return out.to_numpy(dtype="float64")
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)

class InfinityToNaN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        arr = np.asarray(X, dtype="float64")
        return np.where(np.isfinite(arr), arr, np.nan)

numeric_steps = [
    ("finite", InfinityToNaN()),
    ("impute", SimpleImputer(strategy="median", add_indicator=ADD_MISSING_INDICATORS)),
    ("scale",  StandardScaler()),
]
if ADD_INTERACTIONS and 2 <= len(numeric_cols) <= 15:
    numeric_steps.append(("interact", PolynomialFeatures(degree=2, interaction_only=True,
                                                         include_bias=False)))
numeric_pipe = Pipeline(numeric_steps)
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
freq_pipe = Pipeline([
    ("freq",  FrequencyEncoder()),
    ("scale", StandardScaler()),
])
transformers = []
if numeric_cols:  transformers.append(("num",  numeric_pipe, numeric_cols))
if low_card_cat:  transformers.append(("cat",  cat_pipe,     low_card_cat))
if high_card_cat: transformers.append(("freq", freq_pipe,    high_card_cat))
preprocessor = ColumnTransformer(transformers, remainder="drop")

X_train = preprocessor.fit_transform(X_train)
X_test  = preprocessor.transform(X_test)
if X_val is not None: X_val = preprocessor.transform(X_val)
print(f"After preprocessing: {{X_train.shape[1]}} features, "
      f"{{X_train.shape[0]}} train / {{X_test.shape[0]}} test rows")

# ---------------------------------------------------------------------
# 6. OPTIONAL TARGET LOG-TRANSFORM (y -> log1p(y))
# ---------------------------------------------------------------------
y_train_orig, y_test_orig = y_train.copy(), y_test.copy()
if LOG_TARGET and (y_train >= 0).all() and (y_test >= 0).all():
    y_train = np.log1p(y_train)
    y_test  = np.log1p(y_test)
    if y_val is not None: y_val = np.log1p(y_val)
    print("Applied log1p transform to target.")

# ---------------------------------------------------------------------
# 7. TRAIN ALL MODELS (cross-validated on the training rows only)
# ---------------------------------------------------------------------
def scaled(model):
    """Standardise the target while fitting (needed by SVR / penalised linear models)."""
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())

models = {{
{models_block}
}}
# PoissonRegressor needs a non-negative target on its original scale
if "PoissonRegressor" in models and (LOG_TARGET or (y_train < 0).any()):
    models.pop("PoissonRegressor")

inv = np.expm1 if LOG_TARGET else (lambda a: a)
scoring = {{
    "r2":   make_scorer(lambda t, p: r2_score(inv(t), inv(p))),
    "rmse": make_scorer(lambda t, p: -np.sqrt(mean_squared_error(inv(t), inv(p)))),
    "mae":  make_scorer(lambda t, p: -mean_absolute_error(inv(t), inv(p))),
}}

def safe_mape(y_true, y_pred):
    mask = y_true != 0
    if not mask.any(): return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compute_metrics(y_true, y_pred):
    return {{
        "MAE":      mean_absolute_error(y_true, y_pred),
        "RMSE":     np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":       r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
        "MAPE_pct": safe_mape(y_true, y_pred),
    }}

cv = {cv_constructor}
cv_kwargs = {{"groups": groups_train}} if GROUP_COLUMN and groups_train is not None else {{}}

def evaluate(name, est):
    t0 = time.time()
    est.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred_test = inv(est.predict(X_test))
    metrics = compute_metrics(y_test_orig, y_pred_test)
    metrics["train_time_sec"] = elapsed
    try:
        out = cross_validate(est, X_train, y_train, cv=cv, scoring=scoring,
                             return_train_score=True, n_jobs=-1, **cv_kwargs)
        metrics["CV_R2_mean"] = float(out["test_r2"].mean())
        metrics["CV_R2_std"] = float(out["test_r2"].std())
        metrics["CV_R2_train_mean"] = float(out["train_r2"].mean())
        metrics["CV_RMSE_mean"] = float(-out["test_rmse"].mean())
        metrics["CV_MAE_mean"] = float(-out["test_mae"].mean())
        gap = metrics["CV_R2_train_mean"] - metrics["CV_R2_mean"]
        metrics["fit_check"] = "overfit?" if gap > 0.15 else "ok"
    except Exception as e:
        print(f"  CV failed for {{name}}: {{e}}")
    return {{"estimator": est, "y_pred_test": y_pred_test, "metrics": metrics}}

results = {{}}
for name, est in models.items():
    print(f"Training {{name}}...")
    try:
        results[name] = evaluate(name, est)
    except Exception as e:
        print(f"  skipped {{name}}: {{e}}")

# The model the app selected, rebuilt with its FINAL settings (after tuning /
# automatic over-/underfitting fixes). App selection: {selection_note}
{winner_block}
if WINNER is not None:
    print(f"Training the app's winner: {{WINNER_NAME}} (exact settings)...")
    results[WINNER_NAME + " [app winner]"] = evaluate(WINNER_NAME, WINNER)

# ---------------------------------------------------------------------
# 8. RANK + REPORT
# ---------------------------------------------------------------------
cv_key = f"CV_{{SELECTION_METRIC.upper()}}_mean"
metrics_df = pd.DataFrame(
    [{{"Model": n, **r["metrics"]}} for n, r in results.items()]
).sort_values(cv_key if cv_key in [c for r in results.values() for c in r["metrics"]] else "RMSE")
print(f"\\n=== Model comparison (sorted by cross-validated {{SELECTION_METRIC.upper()}}) ===")
print(metrics_df.to_string(index=False))
if WINNER is not None:
    best_name = WINNER_NAME + " [app winner]"
else:
    _ranked = metrics_df[metrics_df["Model"] != "Baseline (mean)"]
    best_name = (_ranked if len(_ranked) else metrics_df).iloc[0]["Model"]
print(f"\\n>>> Final model: {{best_name}}")

# ---------------------------------------------------------------------
# 9. PLOTS
# ---------------------------------------------------------------------
sns.set_theme(style="whitegrid")

# comparison: RMSE
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=metrics_df, x="Model", y="RMSE", ax=ax, palette="viridis")
ax.set_title("Model comparison — test RMSE (lower is better)")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("comparison_rmse.png", dpi=110); plt.show()

# diagnostics for the final model
r = results[best_name]
y_pred = r["y_pred_test"]
residuals = y_test_orig - y_pred
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].scatter(y_test_orig, y_pred, alpha=0.55, edgecolor="none")
lo, hi = min(y_test_orig.min(), y_pred.min()), max(y_test_orig.max(), y_pred.max())
axes[0].plot([lo, hi], [lo, hi], "r--")
axes[0].set_title(f"{{best_name}} — Predicted vs Actual")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
axes[1].scatter(y_pred, residuals, alpha=0.55, edgecolor="none", color="#10b981")
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_title("Residuals vs Predicted")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[2].hist(residuals, bins=40, color="#8b5cf6", alpha=0.85)
axes[2].set_title("Residual Distribution")
axes[2].set_xlabel("Residual"); axes[2].set_ylabel("Frequency")
plt.tight_layout(); plt.savefig("diagnostics_final_model.png", dpi=110); plt.show()

# ---------------------------------------------------------------------
# 10. PERSIST FINAL MODEL
# ---------------------------------------------------------------------
import joblib

class RawFeatureSteps(BaseEstimator, TransformerMixin):
    """Date-part and time-split features, so the saved model accepts RAW rows."""
    def __init__(self, datetime_cols=None, hour_cols=None, time_col=None, t0=None, parts=None):
        self.datetime_cols, self.hour_cols = datetime_cols, hour_cols
        self.time_col, self.t0, self.parts = time_col, t0, parts
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        cols = [c for c in (self.datetime_cols or []) if c in X.columns]
        if cols: X, _ = extract_datetime(X, cols, hour_cols=set(self.hour_cols or []))
        if self.time_col and self.time_col in X.columns:
            q = pd.to_datetime(X[self.time_col], errors="coerce")
            X[f"{{self.time_col}}_elapsed_days"] = (q - self.t0).dt.total_seconds() / 86400.0
            for part in self.parts or []:
                X[f"{{self.time_col}}_{{part}}"] = getattr(q.dt, part)
            X = X.drop(columns=[self.time_col])
        return X

_g = globals()
full_preprocessor = Pipeline([
    ("raw_features", RawFeatureSteps(
        datetime_cols=list(_g.get("datetime_cols", [])),
        hour_cols=sorted(_g.get("datetime_hour_cols", set())),
        time_col=TIME_COLUMN if "_t0" in _g else None, t0=_g.get("_t0"),
        parts=[k for k, v in _g.get("_parts", {{}}).items() if v])),
    ("columns", preprocessor),
])
final = results[best_name]["estimator"]
# note: the classes used above live in this script, so load the file from code
# that defines (or imports) them - e.g. run this script, or import it as a module
joblib.dump({{"preprocessor": full_preprocessor, "model": final,
             "target_transform": "log1p" if LOG_TARGET else "none",
             "target": TARGET,
             "prediction_interval": {{"halfwidth": INTERVAL_HALFWIDTH, "coverage_target": 0.9}}}},
            "best_model.joblib")
print(f"\\nSaved best_model.joblib (preprocessor + {{best_name}})")
'''
    return script


def generate_notebook(**kwargs) -> str:
    """Same content as the script, packaged as a .ipynb.

    Each section in the script looks like:
        # -----...
        # 4. SPLIT
        # -----...
        <code>
    The earlier version split on the divider line alone, which turned the
    first line of every code section into a heading and broke the notebook.
    Splitting on the whole three-line header keeps titles and code separate.
    """
    script = generate_python_script(**kwargs)
    dash = "# " + "-" * 69
    pattern = re.compile(re.escape(dash) + r"\n# (.+)\n" + re.escape(dash) + r"\n")
    pieces = pattern.split(script)          # [intro, title1, body1, title2, body2, ...]
    cells: list[dict[str, Any]] = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

    def code(text):
        text = text.strip("\n")
        if text.strip():
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": (text + "\n").splitlines(keepends=True)})

    md("# End-to-end regression pipeline\n*Auto-generated by the Regression Crew agentic project.*\n")
    code(pieces[0])
    for title, body in zip(pieces[1::2], pieces[2::2]):
        md(f"## {title.strip()}\n")
        code(body)
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)
