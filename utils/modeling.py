"""
Model training & evaluation for regression.

Highlights:
- Optional XGBoost / LightGBM (if installed)
- Optional hyperparameter tuning (small GridSearchCV grids)
- CV strategy: KFold (default), GroupKFold (if groups given), TimeSeriesSplit (time mode)
- Configurable n_folds
- Inverse-transforms target predictions when log1p was applied so that
  reported metrics live in the ORIGINAL units, not log units.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from contextlib import contextmanager


@contextmanager
def _quiet_fit():
    """Suppress the noisy convergence/deprecation warnings AROUND A FIT ONLY.

    This used to be a module-level warnings.filterwarnings("ignore"), which
    silenced warnings for the entire process - including genuine signals such
    as Lasso failing to converge, which is exactly what you want to hear about.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Try to import optional gradient-boosting libs. They're commonly the best on
# tabular data, but they're heavy installs — skip silently if missing.
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


@dataclass
class ModelResult:
    name: str
    estimator: Any
    y_pred_train: np.ndarray  # in original units (back-transformed if log was applied)
    y_pred_test:  np.ndarray
    metrics: dict[str, float]
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    train_time_sec: float = 0.0
    feature_importances: np.ndarray | None = None
    best_params: dict | None = None  # filled when tuning is on
    val_metrics: dict[str, float] = field(default_factory=dict)
    permutation_importances: np.ndarray | None = None
    permutation_importances_std: np.ndarray | None = None


# ---- model zoo -------------------------------------------------------------


#: Always-trained reference point. Predicts the training mean for every row, so
#: its R2 is ~0 by construction. Having it in the table makes every other score
#: interpretable ("is this model actually better than doing nothing?") and makes
#: a negative R2 obvious rather than mysterious. Excluded from winner selection.
BASELINE_MODEL_NAME = "Baseline (mean)"


def get_default_model_zoo(random_state: int = 42) -> dict[str, Any]:
    zoo: dict[str, Any] = {
        BASELINE_MODEL_NAME: DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0, random_state=random_state),
        "Lasso":            Lasso(alpha=0.01, random_state=random_state, max_iter=10000),
        "ElasticNet":       ElasticNet(alpha=0.01, l1_ratio=0.5,
                                       random_state=random_state, max_iter=10000),
        "DecisionTree":     DecisionTreeRegressor(random_state=random_state, max_depth=10),
        "RandomForest":     RandomForestRegressor(n_estimators=200,
                                                  random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200,
                                                      random_state=random_state),
        "KNN":              KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "SVR":              SVR(kernel="rbf", C=1.0, gamma="scale"),
    }
    if _HAS_XGB:
        zoo["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=random_state, n_jobs=-1, verbosity=0,
        )
    if _HAS_LGBM:
        zoo["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=random_state, n_jobs=-1, verbosity=-1,
        )
    return zoo


def available_model_names() -> list[str]:
    return list(get_default_model_zoo().keys())


# Compact tuning grids — small on purpose to keep total runtime sane.
TUNING_GRIDS: dict[str, dict[str, list]] = {
    "Ridge":            {"alpha": [0.1, 1.0, 10.0]},
    "Lasso":            {"alpha": [0.001, 0.01, 0.1]},
    "ElasticNet":       {"alpha": [0.01, 0.1], "l1_ratio": [0.3, 0.5, 0.7]},
    "DecisionTree":     {"max_depth": [5, 10, 20, None]},
    "RandomForest":     {"n_estimators": [100, 200], "max_depth": [10, 20, None]},
    "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "XGBoost":          {"n_estimators": [200, 400],
                         "learning_rate": [0.05, 0.1], "max_depth": [4, 6]},
    "LightGBM":         {"n_estimators": [200, 400],
                         "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
    "KNN":              {"n_neighbors": [3, 5, 10, 15]},
    "SVR":              {"C": [0.1, 1.0, 10.0]},
}


# ---- metrics ----------------------------------------------------------------


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "RMSE":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MSE":      float(mean_squared_error(y_true, y_pred)),
        "R2":       float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE_pct": _safe_mape(y_true, y_pred),
    }


def _extract_feature_importance(estimator: Any, n_features: int) -> np.ndarray | None:
    if hasattr(estimator, "feature_importances_"):
        try:
            return np.asarray(estimator.feature_importances_)
        except Exception:
            return None
    if hasattr(estimator, "coef_"):
        try:
            coef = np.asarray(estimator.coef_)
            return np.abs(coef.ravel()) if coef.size == n_features else None
        except Exception:
            return None
    return None


# ---- CV helpers -------------------------------------------------------------


def _make_cv(strategy: str, n_folds: int, groups: np.ndarray | None,
             random_state: int = 42):
    """Pick KFold / GroupKFold / TimeSeriesSplit.

    `random_state` is threaded through rather than hardcoded. Previously this
    was always 42, so a user who changed the seed in the UI got a reproducible
    train/test split but non-matching, unchangeable CV folds.
    """
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_folds), None
    if strategy == "group" and groups is not None:
        n_groups = len(np.unique(groups))
        # GroupKFold cannot make more folds than there are distinct groups.
        return GroupKFold(n_splits=min(n_folds, max(2, n_groups))), groups
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_state), None


def _original_unit_scorers(target_transform: str) -> dict[str, Any]:
    """Build CV scorers that report in the ORIGINAL target units.

    With log1p on, the model trains on log values, so a plain "r2" scorer
    returns a log-scale R2 while the test metrics are back-transformed to
    original units. Those two numbers then sat side by side in the UI and were
    not comparable. These scorers invert the transform inside the scorer so
    both live on the same scale.
    """
    if target_transform != "log1p":
        return {"r2": "r2", "rmse": "neg_root_mean_squared_error"}

    def _r2(y_true, y_pred):
        return r2_score(np.expm1(y_true), np.expm1(y_pred))

    def _neg_rmse(y_true, y_pred):
        return -float(np.sqrt(mean_squared_error(np.expm1(y_true),
                                                 np.expm1(y_pred))))

    return {"r2": make_scorer(_r2), "rmse": make_scorer(_neg_rmse)}


# ---- main training driver --------------------------------------------------


def train_and_evaluate(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    *,
    models: dict[str, Any] | None = None,
    cv_folds: int = 5,
    cv_strategy: Literal["kfold", "time", "group"] = "kfold",
    groups_train: np.ndarray | None = None,
    target_transform: Literal["none", "log1p"] = "none",
    tune_hyperparameters: bool = False,
    progress_callback=None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    random_state: int = 42,
) -> dict[str, ModelResult]:
    """Train each model. Metrics are reported in ORIGINAL target units."""
    if models is None:
        models = get_default_model_zoo()

    cv, group_arg = _make_cv(cv_strategy, cv_folds, groups_train, random_state)
    scorers = _original_unit_scorers(target_transform)

    # If we log-transformed the target, keep the original-unit copies for metrics
    if target_transform == "log1p":
        y_train_orig = np.expm1(y_train)
        y_test_orig  = np.expm1(y_test)
        y_val_orig   = np.expm1(y_val) if y_val is not None else None
    else:
        y_train_orig = y_train
        y_test_orig  = y_test
        y_val_orig   = y_val

    results: dict[str, ModelResult] = {}
    n_models = len(models)
    for idx, (name, est) in enumerate(models.items(), start=1):
        if progress_callback:
            progress_callback(idx, n_models, name)

        t0 = time.time()
        best_params = None
        with _quiet_fit():
            if tune_hyperparameters and name in TUNING_GRIDS:
                grid = GridSearchCV(
                    est, TUNING_GRIDS[name],
                    cv=cv, scoring=scorers["rmse"],
                    n_jobs=-1, refit=True,
                )
                try:
                    grid.fit(X_train, y_train, groups=group_arg) if group_arg is not None \
                        else grid.fit(X_train, y_train)
                    est = grid.best_estimator_
                    best_params = grid.best_params_
                except Exception:
                    est.fit(X_train, y_train)
            else:
                est.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred_train_raw = est.predict(X_train)
        y_pred_test_raw  = est.predict(X_test)
        if target_transform == "log1p":
            y_pred_train = np.expm1(y_pred_train_raw)
            y_pred_test  = np.expm1(y_pred_test_raw)
        else:
            y_pred_train = y_pred_train_raw
            y_pred_test  = y_pred_test_raw

        metrics = _compute_metrics(y_test_orig, y_pred_test)
        train_metrics = _compute_metrics(y_train_orig, y_pred_train)
        metrics["R2_train"]   = train_metrics["R2"]
        metrics["RMSE_train"] = train_metrics["RMSE"]

        # Validation metrics, when a validation set was supplied. These are what
        # model SELECTION should use, leaving the test set as an untouched,
        # unbiased estimate of the winner's performance.
        val_metrics: dict[str, float] = {}
        if X_val is not None and y_val_orig is not None and len(y_val_orig) > 0:
            with _quiet_fit():
                y_pred_val_raw = est.predict(X_val)
            y_pred_val = (np.expm1(y_pred_val_raw)
                          if target_transform == "log1p" else y_pred_val_raw)
            val_metrics = _compute_metrics(y_val_orig, y_pred_val)
            metrics["RMSE_val"] = val_metrics["RMSE"]
            metrics["R2_val"]   = val_metrics["R2"]

        # CV — done in whatever units the model was trained in (cleaner)
        try:
            with _quiet_fit():
                cv_r2 = cross_val_score(
                    est, X_train, y_train, cv=cv,
                    scoring=scorers["r2"], n_jobs=-1, groups=group_arg,
                )
                cv_neg_rmse = cross_val_score(
                    est, X_train, y_train, cv=cv,
                    scoring=scorers["rmse"], n_jobs=-1, groups=group_arg,
                )
            cv_scores = {"R2": cv_r2.tolist(), "RMSE": (-cv_neg_rmse).tolist()}
            metrics["CV_R2_mean"]   = float(np.mean(cv_r2))
            metrics["CV_R2_std"]    = float(np.std(cv_r2))
            metrics["CV_RMSE_mean"] = float(np.mean(-cv_neg_rmse))
        except Exception:
            cv_scores = {}

        importances = _extract_feature_importance(est, X_train.shape[1])

        results[name] = ModelResult(
            name=name, estimator=est,
            y_pred_train=y_pred_train, y_pred_test=y_pred_test,
            metrics=metrics, cv_scores=cv_scores,
            train_time_sec=elapsed, feature_importances=importances,
            best_params=best_params, val_metrics=val_metrics,
        )

    return results


def rank_models(results: dict[str, ModelResult]) -> list[tuple[str, float, float]]:
    """Rank every model (baseline included) by TEST RMSE, for display."""
    rows = [(n, r.metrics["RMSE"], r.metrics["R2"]) for n, r in results.items()]
    rows.sort(key=lambda row: (row[1], -row[2]))
    return rows


def selection_basis(results: dict[str, ModelResult]) -> str:
    """'validation' when every model has a validation score, else 'test'."""
    if not results:
        return "test"
    if all(r.metrics.get("RMSE_val") is not None
           and not np.isnan(r.metrics.get("RMSE_val", float("nan")))
           for r in results.values()):
        return "validation"
    return "test"


def pick_best(results: dict[str, ModelResult]) -> str:
    """Choose the winner.

    Two corrections over the original behaviour:

    1. Select on the VALIDATION set when one exists. Selecting on test turns
       the test set into a selection set, so the reported test score is
       optimistically biased - with a dozen candidates one looks good partly
       by luck. Validation selection keeps test as an honest final estimate.
    2. Exclude the mean-prediction baseline from winning. It exists as a
       reference point, not as a deployable model. (If it is the ONLY entry,
       it is returned so callers always get a name.)
    """
    if not results:
        raise ValueError("No models were trained.")

    basis = selection_basis(results)
    key = "RMSE_val" if basis == "validation" else "RMSE"
    tie = "R2_val" if basis == "validation" else "R2"

    candidates = {n: r for n, r in results.items() if n != BASELINE_MODEL_NAME}
    if not candidates:
        candidates = results

    ranked = sorted(
        candidates.items(),
        key=lambda kv: (kv[1].metrics.get(key, float("inf")),
                        -kv[1].metrics.get(tie, float("-inf"))),
    )
    return ranked[0][0]


def compute_permutation_importance(
    estimator, X, y, *, n_repeats: int = 10, random_state: int = 42,
    scoring: str = "r2",
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Permutation importance on HELD-OUT data.

    Impurity-based feature_importances_ is biased toward high-cardinality and
    continuous features, and is computed on training data, so it reflects what
    the model overfit to as much as what actually matters. Permutation
    importance measures the real drop in held-out performance when a column is
    shuffled, and works for any estimator - including KNN and SVR, which expose
    no native importances at all.
    """
    try:
        with _quiet_fit():
            result = permutation_importance(
                estimator, X, y, n_repeats=n_repeats,
                random_state=random_state, scoring=scoring, n_jobs=-1,
            )
        return (np.asarray(result.importances_mean),
                np.asarray(result.importances_std))
    except Exception:
        return None, None
