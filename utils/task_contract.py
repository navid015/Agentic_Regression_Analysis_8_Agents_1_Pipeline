"""Explicit regression task contract; validated before any agent sees data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionTask:
    """Columns listed in `available_features` must exist at prediction time.

    When omitted, all non-target columns are assumed available. The system
    warns that it cannot infer semantic/post-outcome leakage from names alone.
    """
    available_features: tuple[str, ...] | None = None
    target_kind: str = "auto"  # auto | continuous | count | positive | nonnegative | bounded
    objective: str = "auto"  # auto | rmse | mae
    ratio_features: tuple[tuple[str, str], ...] = ()
    bounds: tuple[float, float] | None = None


@dataclass
class TaskAssessment:
    kind: str
    warnings: list[str] = field(default_factory=list)
    candidate_models: list[str] = field(default_factory=list)


def apply_feature_contract(task: RegressionTask, frame: pd.DataFrame,
                           feature_columns: list[str]) -> pd.DataFrame:
    """Pure row-wise feature construction, safe to apply at inference time."""
    missing = set(feature_columns) - set(frame)
    if missing:
        raise ValueError(f"Prediction rows lack required features: {sorted(missing)}")
    out = frame[feature_columns].copy()
    for a, b in task.ratio_features:
        name = f"ratio__{a}__over__{b}"
        if name in out:
            raise ValueError(f"Ratio feature name collision: {name}")
        numerator = pd.to_numeric(out[a], errors="coerce").astype(float)
        denominator = pd.to_numeric(out[b], errors="coerce").astype(float)
        out[name] = numerator.div(denominator.where(denominator.abs() > 1e-12)).replace(
            [np.inf, -np.inf], np.nan)
    return out


def prepare_task_frames(task: RegressionTask, train: pd.DataFrame, test: pd.DataFrame | None,
                        target: str, *, time_column: str | None, group_column: str | None):
    if task.objective not in ("auto", "rmse", "mae"):
        raise ValueError("Supported selection objectives are auto, RMSE and MAE")
    if task.target_kind not in ("auto", "continuous", "count", "positive", "nonnegative", "bounded"):
        raise ValueError("Unsupported target_kind")
    cols = set(train.columns) - {target}
    protected = {c for c in (time_column, group_column) if c}
    selected = set(task.available_features) if task.available_features is not None else cols
    if target in selected or not selected <= cols:
        raise ValueError("available_features must name existing non-target columns")
    if task.available_features is not None and not selected:
        raise ValueError("At least one prediction-time feature is required")
    selected |= protected
    if not selected <= cols:
        raise ValueError("Time/group column is missing from training data")
    if len(task.ratio_features) > 20:
        raise ValueError("At most 20 explicit ratio features are supported")
    for a, b in task.ratio_features:
        if a not in selected or b not in selected or a == b:
            raise ValueError("Ratio inputs must be distinct, available features")
        if not pd.api.types.is_numeric_dtype(train[a]) or not pd.api.types.is_numeric_dtype(train[b]):
            raise ValueError("Ratio inputs must be numeric")
    keep = [c for c in train if c == target or c in selected]
    if test is not None and not set(keep) <= set(test):
        raise ValueError("Final test is missing prediction-time features or target")
    features = [c for c in keep if c != target]
    def transform(frame):
        out = apply_feature_contract(task, frame, features)
        out[target] = frame[target].to_numpy()
        return out
    return transform(train), transform(test) if test is not None else None


def assess_target(y: Iterable, task: RegressionTask) -> TaskAssessment:
    arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        raise ValueError("At least 30 finite training targets are required")
    integral = bool(np.all(np.isclose(arr, np.round(arr))))
    if task.target_kind == "auto":
        if np.min(arr) >= 0 and integral and (np.max(arr) <= 100 or len(np.unique(arr)) < len(arr) / 3):
            kind = "count"
        elif np.min(arr) > 0 and np.std(arr) / max(np.mean(arr), 1e-12) > 0.6:
            kind = "positive"
        elif np.min(arr) >= 0:
            kind = "nonnegative"
        else:
            kind = "continuous"
    else:
        kind = task.target_kind
    if kind == "count" and (np.min(arr) < 0 or not integral):
        raise ValueError("Count targets must contain non-negative integers")
    if kind == "positive" and np.min(arr) <= 0:
        raise ValueError("Positive targets cannot contain zero or negative values")
    if kind in ("nonnegative", "bounded") and np.min(arr) < 0 and kind == "nonnegative":
        raise ValueError("Nonnegative target contains negative values")
    warnings = []
    if task.target_kind == "auto":
        warnings.append("Target type was inferred from numeric values; confirm its domain meaning (integer amounts may look like counts)")
    if task.available_features is None:
        warnings.append("Prediction-time feature availability was not confirmed; review post-outcome columns for leakage")
    if kind == "bounded":
        if task.bounds is None or task.bounds[0] >= task.bounds[1]:
            raise ValueError("Bounded target requires valid bounds=(lower, upper)")
        if np.min(arr) < task.bounds[0] or np.max(arr) > task.bounds[1]:
            raise ValueError("Target falls outside declared bounds")
        warnings.append("Bounded outcomes use general regressors; predictions are not constrained to the bounds")
    if kind == "count" and (arr == 0).mean() > .5:
        warnings.append("More than half of count targets are zero; no hurdle/zero-inflated model is implemented")
    models = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "Huber",
              "DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting",
              "HistGradientBoosting", "KNN", "SVR", "XGBoost", "LightGBM"]
    if kind in ("count", "nonnegative"):
        models += ["PoissonRegressor"]
    if kind == "positive":
        models += ["GammaRegressor", "TweedieRegressor"]
    return TaskAssessment(kind=kind, warnings=warnings, candidate_models=models)


def shift_report(train: pd.DataFrame, test: pd.DataFrame, target: str, *, max_columns=100) -> dict:
    """Label-free alerts; no final-test target is inspected."""
    report = {}
    for col in list(train.columns.drop(target, errors="ignore"))[:max_columns]:
        if col not in test:
            continue
        left, right = train[col], test[col]
        if pd.api.types.is_numeric_dtype(left):
            a, b = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
            if a.notna().sum() < 10:
                continue
            lo, hi = a.quantile([.01, .99])
            outside = float(((b < lo) | (b > hi)).mean())
            if outside > .2:
                report[col] = {"kind": "outside_training_1_99_percentiles", "fraction": round(outside, 3)}
        else:
            known = set(left.dropna().astype(str))
            unseen = float((~right.dropna().astype(str).isin(known)).mean()) if right.notna().any() else 0
            if unseen > .1:
                report[col] = {"kind": "unseen_categories", "fraction": round(unseen, 3)}
    return report


def subgroup_error_report(y_true, y_pred, groups=None, *, time_ordered=False) -> dict:
    """Final-test diagnostic only; never feed this report to an experiment agent."""
    truth, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(truth) != len(pred):
        raise ValueError("Prediction length does not match final-test target")
    out = {}
    def summarize(mask):
        return {"rows": int(mask.sum()), "mae": round(float(np.mean(np.abs(truth[mask] - pred[mask]))), 5),
                "rmse": round(float(np.sqrt(np.mean((truth[mask] - pred[mask]) ** 2))), 5)}
    if time_ordered and len(truth) >= 20:
        middle = len(truth) // 2
        first = np.arange(len(truth)) < middle
        out["earlier_test_period"] = summarize(first)
        out["later_test_period"] = summarize(~first)
    elif groups is not None:
        g = pd.Series(groups).reset_index(drop=True)
        if len(g) != len(truth):
            raise ValueError("Group labels do not align with final-test rows")
        for key, count in g.value_counts(dropna=True).head(10).items():
            if count >= 5:
                out[str(key)] = summarize((g == key).to_numpy())
    return out
