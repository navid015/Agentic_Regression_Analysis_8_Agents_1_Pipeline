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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
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

warnings.filterwarnings("ignore")

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


# ---- model zoo -------------------------------------------------------------


def get_default_model_zoo(random_state: int = 42) -> dict[str, Any]:
    zoo: dict[str, Any] = {
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


def _make_cv(strategy: str, n_folds: int, groups: np.ndarray | None):
    """Pick KFold / GroupKFold / TimeSeriesSplit."""
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_folds), None
    if strategy == "group" and groups is not None:
        return GroupKFold(n_splits=n_folds), groups
    return KFold(n_splits=n_folds, shuffle=True, random_state=42), None


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
) -> dict[str, ModelResult]:
    """Train each model. Metrics are reported in ORIGINAL target units."""
    if models is None:
        models = get_default_model_zoo()

    cv, group_arg = _make_cv(cv_strategy, cv_folds, groups_train)

    # If we log-transformed the target, keep the original-unit copies for metrics
    if target_transform == "log1p":
        y_train_orig = np.expm1(y_train)
        y_test_orig  = np.expm1(y_test)
    else:
        y_train_orig = y_train
        y_test_orig  = y_test

    results: dict[str, ModelResult] = {}
    n_models = len(models)
    for idx, (name, est) in enumerate(models.items(), start=1):
        if progress_callback:
            progress_callback(idx, n_models, name)

        t0 = time.time()
        best_params = None
        if tune_hyperparameters and name in TUNING_GRIDS:
            grid = GridSearchCV(
                est, TUNING_GRIDS[name],
                cv=cv, scoring="neg_root_mean_squared_error",
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

        # CV — done in whatever units the model was trained in (cleaner)
        try:
            cv_r2 = cross_val_score(
                est, X_train, y_train, cv=cv,
                scoring="r2", n_jobs=-1, groups=group_arg,
            )
            cv_neg_rmse = cross_val_score(
                est, X_train, y_train, cv=cv,
                scoring="neg_root_mean_squared_error", n_jobs=-1, groups=group_arg,
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
            best_params=best_params,
        )

    return results


def rank_models(results: dict[str, ModelResult]) -> list[tuple[str, float, float]]:
    rows = [(n, r.metrics["RMSE"], r.metrics["R2"]) for n, r in results.items()]
    rows.sort(key=lambda row: (row[1], -row[2]))
    return rows


def pick_best(results: dict[str, ModelResult]) -> str:
    return rank_models(results)[0][0]
