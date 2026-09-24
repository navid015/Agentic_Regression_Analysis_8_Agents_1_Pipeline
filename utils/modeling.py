"""
Model training, evaluation and selection for regression.

Design goals (v2):

* **Prevent** over/underfitting with sensible regularised defaults, early
  stopping for boosting, target scaling for scale-sensitive models, and
  randomised hyperparameter search over the knobs that actually control
  model capacity.
* **Detect** it honestly: every model gets a fit diagnosis computed from
  cross-validation on the TRAINING data only (train-fold vs held-out-fold
  score, skill vs. a mean baseline, fold-to-fold stability). The test set is
  never used to make decisions.
* **Fix** it automatically: models diagnosed as over- or underfitting are
  retried with more / less regularisation and the change is kept only if the
  cross-validated error improves.
* **Choose** robustly: the winner is picked on a validation file when one is
  supplied, otherwise on cross-validated RMSE, with the one-standard-error
  rule preferring the simplest model that is statistically as good as the best.
* **Report** in original target units even when a log transform was used,
  plus learning curves and conformal prediction intervals for the winner.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    PoissonRegressor,
    Ridge,
)
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    TimeSeriesSplit,
    cross_validate,
    learning_curve,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@contextmanager
def _quiet_fit():
    """Silence convergence / deprecation chatter around a single fit only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


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


# ---- thresholds (one place, documented) ------------------------------------

#: train-fold R2 minus held-out-fold R2 above this -> overfitting
OVERFIT_GAP = 0.15
#: less than this fractional RMSE improvement over the mean baseline -> underfitting
UNDERFIT_MIN_SKILL = 0.05
#: train-fold R2 below this, with a small gap, -> model too simple for the data
UNDERFIT_LOW_TRAIN_R2 = 0.30
#: fold-to-fold R2 standard deviation above this -> unstable
UNSTABLE_CV_STD = 0.15
#: remediation must improve CV RMSE by at least this fraction to be kept
REMEDY_MIN_IMPROVEMENT = 0.01
#: below this many training rows, K-fold CV is repeated 3x for stabler scores
SMALL_DATA_ROWS = 300
#: above this many training rows SVR is skipped (its cost grows ~n^2 to n^3)
SVR_MAX_ROWS = 15000
#: trials per model for each tuning mode
TUNING_TRIALS = {"off": 0, "fast": 10, "thorough": 40}


@dataclass
class ModelResult:
    name: str
    estimator: Any
    y_pred_train: np.ndarray  # original units
    y_pred_test: np.ndarray
    metrics: dict[str, float]
    cv_scores: dict[str, list[float]] = field(default_factory=dict)
    train_time_sec: float = 0.0
    feature_importances: np.ndarray | None = None
    best_params: dict | None = None
    val_metrics: dict[str, float] = field(default_factory=dict)
    permutation_importances: np.ndarray | None = None
    permutation_importances_std: np.ndarray | None = None
    # v2 additions
    fit_status: str = "unknown"          # good | benign | overfit | underfit | unstable | unknown
    fit_reasons: list[str] = field(default_factory=list)
    remediation: str | None = None
    cv_optimistic: bool = False          # True when CV scores come from the tuning search itself
    ensemble_members: list[str] | None = None
    learning_curve: dict | None = None
    prediction_interval: dict | None = None


# ---- model zoo -------------------------------------------------------------

BASELINE_MODEL_NAME = "Baseline (mean)"
ENSEMBLE_NAME = "Ensemble (top-3 average)"

#: Rough capacity ranking used by the one-standard-error rule (lower = simpler).
COMPLEXITY = {
    BASELINE_MODEL_NAME: 0,
    "LinearRegression": 1, "Ridge": 1, "Lasso": 1, "ElasticNet": 1,
    "Huber": 1, "PoissonRegressor": 1,
    "KNN": 2, "DecisionTree": 2, "SVR": 3,
    "RandomForest": 4, "ExtraTrees": 4,
    "GradientBoosting": 5, "HistGradientBoosting": 5, "XGBoost": 5, "LightGBM": 5,
    ENSEMBLE_NAME: 6,
}

#: Models whose loss depends on the scale of y. They are wrapped so the target
#: is standardised during fitting and predictions are un-scaled automatically.
#: Without this, SVR's epsilon=0.1 / C=1 are meaningless on a target measured
#: in hundreds of thousands and the model learns nothing (R2 ~ 0).
TARGET_SCALED = {"Ridge", "Lasso", "ElasticNet", "Huber", "SVR"}


def _scaled(est):
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def get_default_model_zoo(random_state: int = 42) -> dict[str, Any]:
    rs = random_state
    zoo: dict[str, Any] = {
        BASELINE_MODEL_NAME: DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0, random_state=rs),
        "Lasso":            Lasso(alpha=0.01, random_state=rs, max_iter=20000),
        "ElasticNet":       ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=rs, max_iter=20000),
        "Huber":            HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000),
        "PoissonRegressor": PoissonRegressor(alpha=1e-3, max_iter=2000),
        "DecisionTree":     DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=rs),
        "RandomForest":     RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                  random_state=rs, n_jobs=-1),
        "ExtraTrees":       ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                                random_state=rs, n_jobs=-1),
        # early stopping on an internal 10% split stops boosting before it memorises
        "GradientBoosting": GradientBoostingRegressor(n_estimators=600, learning_rate=0.05,
                                                      max_depth=3, subsample=0.8,
                                                      validation_fraction=0.1,
                                                      n_iter_no_change=20, random_state=rs),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                              l2_regularization=1.0,
                                                              early_stopping=True,
                                                              validation_fraction=0.1,
                                                              n_iter_no_change=20,
                                                              random_state=rs),
        "KNN":              KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
        "SVR":              SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"),
    }
    if _HAS_XGB:
        zoo["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
            random_state=rs, n_jobs=-1, verbosity=0,
        )
    if _HAS_LGBM:
        zoo["LightGBM"] = LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=rs, n_jobs=-1, verbosity=-1,
        )
    return {n: (_scaled(e) if n in TARGET_SCALED else e) for n, e in zoo.items()}


def available_model_names() -> list[str]:
    return list(get_default_model_zoo().keys())


def filter_zoo_for_data(zoo: dict[str, Any], y_train: np.ndarray, *,
                        target_transform: str = "none") -> tuple[dict[str, Any], dict[str, str]]:
    """Drop models that cannot work on this particular dataset, with reasons."""
    kept, skipped = {}, {}
    n = len(y_train)
    for name, est in zoo.items():
        if name == "PoissonRegressor":
            if target_transform != "none":
                skipped[name] = "not used with a log-transformed target (Poisson already models log-scale)"
                continue
            if np.nanmin(y_train) < 0:
                skipped[name] = "needs a non-negative target (counts, amounts, durations)"
                continue
        if name == "SVR" and n > SVR_MAX_ROWS:
            skipped[name] = f"skipped for speed: {n:,} training rows > {SVR_MAX_ROWS:,}"
            continue
        kept[name] = est
    return kept, skipped


# ---- hyperparameter search spaces -----------------------------------------
# Every space includes the knobs that control capacity (depth, leaf size,
# penalty strength, learning rate, subsampling) so the search can move a model
# away from overfitting OR underfitting, not just nudge accuracy.

def _search_space(name: str, n_train: int, n_splits: int) -> dict[str, Any] | None:
    fold_rows = max(2, int(n_train * (1 - 1 / max(n_splits, 2))))
    spaces: dict[str, dict[str, Any]] = {
        "Ridge":        {"alpha": loguniform(1e-3, 1e3)},
        "Lasso":        {"alpha": loguniform(1e-4, 10)},
        "ElasticNet":   {"alpha": loguniform(1e-4, 10), "l1_ratio": uniform(0.05, 0.9)},
        "Huber":        {"epsilon": uniform(1.1, 1.0), "alpha": loguniform(1e-6, 1)},
        "PoissonRegressor": {"alpha": loguniform(1e-6, 10)},
        "DecisionTree": {"max_depth": [3, 5, 8, 12, 20, None],
                         "min_samples_leaf": randint(1, 40),
                         "max_features": [1.0, 0.8, 0.6]},
        "RandomForest": {"n_estimators": [200, 400], "max_depth": [None, 8, 16, 24],
                         "min_samples_leaf": randint(1, 20),
                         "max_features": [1.0, 0.7, 0.5, "sqrt"]},
        "ExtraTrees":   {"n_estimators": [200, 400], "max_depth": [None, 8, 16, 24],
                         "min_samples_leaf": randint(1, 20),
                         "max_features": [1.0, 0.7, 0.5, "sqrt"]},
        "GradientBoosting": {"learning_rate": loguniform(0.01, 0.2), "max_depth": randint(2, 6),
                             "subsample": uniform(0.6, 0.4), "min_samples_leaf": randint(1, 30)},
        "HistGradientBoosting": {"learning_rate": loguniform(0.01, 0.2),
                                 "max_leaf_nodes": randint(8, 64),
                                 "min_samples_leaf": randint(5, 60),
                                 "l2_regularization": loguniform(1e-3, 10)},
        "KNN":          {"n_neighbors": randint(2, max(3, min(50, fold_rows - 1))),
                         "weights": ["uniform", "distance"]},
        "SVR":          {"C": loguniform(1e-2, 1e3), "epsilon": loguniform(1e-3, 1.0),
                         "gamma": ["scale", "auto"]},
        "XGBoost":      {"n_estimators": [300, 600, 1000], "learning_rate": loguniform(0.01, 0.3),
                         "max_depth": randint(2, 9), "subsample": uniform(0.6, 0.4),
                         "colsample_bytree": uniform(0.5, 0.5),
                         "min_child_weight": loguniform(0.5, 20), "reg_lambda": loguniform(1e-2, 10)},
        "LightGBM":     {"n_estimators": [300, 600, 1000], "learning_rate": loguniform(0.01, 0.3),
                         "num_leaves": randint(8, 128), "min_child_samples": randint(5, 60),
                         "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                         "reg_lambda": loguniform(1e-2, 10)},
    }
    space = spaces.get(name)
    if space is None:
        return None
    if name in TARGET_SCALED:
        return {f"regressor__{k}": v for k, v in space.items()}
    return space


#: Targeted fixes tried when a model is diagnosed as over- or underfitting.
REMEDIES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "DecisionTree": {"overfit": [{"max_depth": 6, "min_samples_leaf": 10},
                                 {"max_depth": 4, "min_samples_leaf": 25}],
                     "underfit": [{"max_depth": None, "min_samples_leaf": 2}]},
    "RandomForest": {"overfit": [{"min_samples_leaf": 5, "max_features": 0.6},
                                 {"min_samples_leaf": 12, "max_depth": 10, "max_features": 0.5},
                                 {"min_samples_leaf": 30, "max_depth": 6, "max_features": 0.4}],
                     "underfit": [{"min_samples_leaf": 1, "max_features": 1.0, "max_depth": None}]},
    "ExtraTrees": {"overfit": [{"min_samples_leaf": 5, "max_features": 0.6},
                                 {"min_samples_leaf": 12, "max_depth": 10, "max_features": 0.5},
                                 {"min_samples_leaf": 30, "max_depth": 6, "max_features": 0.4}],
                     "underfit": [{"min_samples_leaf": 1, "max_features": 1.0, "max_depth": None}]},
    "GradientBoosting": {"overfit": [{"max_depth": 2, "min_samples_leaf": 20, "subsample": 0.7},
                                     {"learning_rate": 0.02, "max_depth": 2, "min_samples_leaf": 40},
                                     {"max_depth": 1, "min_samples_leaf": 60}],
                         "underfit": [{"max_depth": 5, "n_estimators": 1500}]},
    "HistGradientBoosting": {"overfit": [{"max_leaf_nodes": 15, "min_samples_leaf": 40,
                                          "l2_regularization": 10.0},
                                         {"max_leaf_nodes": 7, "min_samples_leaf": 80,
                                          "l2_regularization": 30.0, "learning_rate": 0.03}],
                             "underfit": [{"max_leaf_nodes": 63, "min_samples_leaf": 5,
                                           "max_iter": 1500}]},
    "XGBoost":      {"overfit": [{"max_depth": 3, "min_child_weight": 10, "reg_lambda": 10.0,
                                  "subsample": 0.7},
                                 {"max_depth": 2, "min_child_weight": 30, "reg_lambda": 30.0,
                                  "subsample": 0.6, "learning_rate": 0.03}],
                     "underfit": [{"max_depth": 8, "n_estimators": 1200}]},
    "LightGBM":     {"overfit": [{"num_leaves": 15, "min_child_samples": 40, "reg_lambda": 10.0},
                                 {"num_leaves": 7, "min_child_samples": 80, "reg_lambda": 30.0,
                                  "learning_rate": 0.03}],
                     "underfit": [{"num_leaves": 63, "min_child_samples": 5, "n_estimators": 1200}]},
    "KNN":          {"overfit": [{"n_neighbors": 15, "weights": "uniform"},
                                 {"n_neighbors": 30, "weights": "uniform"}],
                     "underfit": [{"n_neighbors": 3}]},
    "SVR":          {"overfit": [{"C": 0.3}], "underfit": [{"C": 10.0}, {"C": 100.0}]},
    "Ridge":        {"overfit": [{"alpha": 10.0}, {"alpha": 100.0}], "underfit": [{"alpha": 0.1}]},
    "Lasso":        {"overfit": [{"alpha": 0.05}, {"alpha": 0.2}], "underfit": [{"alpha": 0.001}]},
    "ElasticNet":   {"overfit": [{"alpha": 0.05}, {"alpha": 0.2}], "underfit": [{"alpha": 0.001}]},
    "Huber":        {"overfit": [{"alpha": 0.1}], "underfit": [{"alpha": 1e-6}]},
}


def _inner(est):
    """The actual model inside a target-scaling wrapper (or the model itself)."""
    return est.regressor if isinstance(est, TransformedTargetRegressor) else est


def _with_params(est, params: dict[str, Any], *, max_neighbors: int | None = None):
    new = clone(est)
    p = dict(params)
    if max_neighbors is not None and "n_neighbors" in p:
        p["n_neighbors"] = max(1, min(p["n_neighbors"], max_neighbors))
    if isinstance(new, TransformedTargetRegressor):
        new.set_params(**{f"regressor__{k}": v for k, v in p.items()})
    else:
        new.set_params(**p)
    return new


def _strip_prefix(params: dict | None) -> dict | None:
    if not params:
        return params
    return {k.replace("regressor__", ""): v for k, v in params.items()}


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
    est = _inner(estimator)
    if isinstance(estimator, TransformedTargetRegressor) and hasattr(estimator, "regressor_"):
        est = estimator.regressor_
    if hasattr(est, "feature_importances_"):
        try:
            return np.asarray(est.feature_importances_, dtype=float)
        except Exception:
            return None
    if hasattr(est, "coef_"):
        try:
            coef = np.asarray(est.coef_, dtype=float).ravel()
            return np.abs(coef) if coef.size == n_features else None
        except Exception:
            return None
    return None


# ---- CV helpers -------------------------------------------------------------

def _make_cv(strategy: str, n_folds: int, groups: np.ndarray | None,
             random_state: int = 42, n_samples: int | None = None):
    """KFold (repeated 3x on small data) / GroupKFold / TimeSeriesSplit."""
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_folds), None
    if strategy == "group" and groups is not None:
        n_groups = len(np.unique(groups))
        return GroupKFold(n_splits=min(n_folds, max(2, n_groups))), groups
    if n_samples is not None and n_samples < SMALL_DATA_ROWS:
        # Small data: a single K-fold is noisy. Repeating it with different
        # shuffles gives steadier scores and a more reliable std.
        return RepeatedKFold(n_splits=n_folds, n_repeats=3, random_state=random_state), None
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_state), None


def _original_unit_scorers(target_transform: str) -> dict[str, Any]:
    """CV scorers in ORIGINAL target units, even when y was log1p-transformed."""
    if target_transform != "log1p":
        return {"r2": "r2", "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error"}

    def _r2(y_true, y_pred):
        return r2_score(np.expm1(y_true), np.expm1(y_pred))

    def _neg_rmse(y_true, y_pred):
        return -float(np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred))))

    def _neg_mae(y_true, y_pred):
        return -float(mean_absolute_error(np.expm1(y_true), np.expm1(y_pred)))

    return {"r2": make_scorer(_r2), "rmse": make_scorer(_neg_rmse), "mae": make_scorer(_neg_mae)}


def _cv_scores(est, X, y, cv, groups, scorers) -> dict[str, list[float]]:
    with _quiet_fit():
        out = cross_validate(
            est, X, y, cv=cv, groups=groups,
            scoring={"r2": scorers["r2"], "rmse": scorers["rmse"], "mae": scorers["mae"]},
            return_train_score=True, n_jobs=-1, error_score="raise",
        )
    return {"R2": out["test_r2"].tolist(), "RMSE": (-out["test_rmse"]).tolist(),
            "MAE": (-out["test_mae"]).tolist(),
            "R2_train": out["train_r2"].tolist(), "RMSE_train": (-out["train_rmse"]).tolist()}


def _cv_into_metrics(metrics: dict, cv: dict, y_scale: float | None = None) -> None:
    """Summarise fold scores.

    Besides the usual per-fold R2, it stores a "global" R2 per fold,
    1 - (RMSE / std of the whole training target)^2. Per-fold R2 compares each
    fold with its OWN mean, which is fine for shuffled K-fold but explodes on
    time-series folds (short windows with little variance give R2 of -30 for a
    decent model). The global version is on one fixed scale, so train-vs-
    held-out gaps and fold-to-fold stability mean the same thing for every CV
    scheme. The fit diagnosis uses it.
    """
    if not cv or not cv.get("R2"):
        return
    r2 = np.asarray(cv["R2"], float); rmse = np.asarray(cv["RMSE"], float)
    metrics["CV_R2_mean"] = float(np.nanmean(r2))
    metrics["CV_R2_std"] = float(np.nanstd(r2))
    metrics["CV_RMSE_mean"] = float(np.nanmean(rmse))
    metrics["CV_RMSE_std"] = float(np.nanstd(rmse))
    metrics["CV_n_splits"] = int(len(r2))
    if cv.get("MAE"):
        mae = np.asarray(cv["MAE"], float)
        metrics["CV_MAE_mean"] = float(np.nanmean(mae))
        metrics["CV_MAE_std"] = float(np.nanstd(mae))
    if cv.get("R2_train"):
        metrics["CV_R2_train_mean"] = float(np.nanmean(np.asarray(cv["R2_train"], float)))
    if y_scale and y_scale > 0:
        g = 1 - (rmse / y_scale) ** 2
        metrics["CV_R2g_mean"] = float(np.nanmean(g))
        metrics["CV_R2g_std"] = float(np.nanstd(g))
        if cv.get("RMSE_train"):
            gt = 1 - (np.asarray(cv["RMSE_train"], float) / y_scale) ** 2
            metrics["CV_R2g_train_mean"] = float(np.nanmean(gt))


# ---- fit diagnosis ----------------------------------------------------------

def diagnose_fit(metrics: dict[str, float], baseline_cv_rmse: float | None,
                 selection_metric: str = "rmse") -> tuple[str, list[str], float | None]:
    """Classify a model as good / overfit / underfit / unstable using CV only.

    Uses the TRAINING data's cross-validation, never the test set, so the
    test score stays an honest final estimate. With `selection_metric="mae"`
    (outlier-heavy targets) skill is measured with MAE against a median
    baseline, and the low-training-R2 rule is skipped because R2 is dominated
    by the outliers themselves.
    """
    reasons: list[str] = []
    # prefer the fold-consistent "global" R2 (see _cv_into_metrics)
    cv_r2 = metrics.get("CV_R2g_mean", metrics.get("CV_R2_mean"))
    tr_r2 = metrics.get("CV_R2g_train_mean",
                        metrics.get("CV_R2_train_mean", metrics.get("R2_train")))
    robust = selection_metric == "mae"
    cv_rmse = metrics.get("CV_MAE_mean" if robust else "CV_RMSE_mean")
    std = metrics.get("CV_R2g_std", metrics.get("CV_R2_std"))
    if cv_r2 is None or cv_rmse is None:
        return "unknown", ["no cross-validation scores available"], None

    skill = None
    if baseline_cv_rmse and baseline_cv_rmse > 0:
        skill = 1.0 - cv_rmse / baseline_cv_rmse
    gap = (tr_r2 - cv_r2) if tr_r2 is not None else 0.0

    status = "good"
    # A large train/held-out gap is checked FIRST: a model that memorises noise
    # also has little held-out skill, but the cure is less capacity, not more.
    if gap > OVERFIT_GAP:
        status = "overfit"
        reasons.append(f"train-fold R\u00b2 {tr_r2:.3f} vs held-out-fold R\u00b2 {cv_r2:.3f} "
                       f"(gap {gap:.3f} > {OVERFIT_GAP})")
        if skill is not None and skill < UNDERFIT_MIN_SKILL:
            reasons.append(f"and only {skill * 100:.1f}% better than predicting the "
                           f"{'median' if robust else 'mean'}: it is memorising noise")
    elif skill is not None and skill < UNDERFIT_MIN_SKILL:
        status = "underfit"
        reasons.append(f"only {skill * 100:.1f}% lower error than predicting the "
                       f"{'median' if robust else 'mean'} "
                       f"(needs at least {UNDERFIT_MIN_SKILL * 100:.0f}%)")
    elif not robust and tr_r2 is not None and tr_r2 < UNDERFIT_LOW_TRAIN_R2:
        status = "underfit"
        reasons.append(f"R\u00b2 is low even on data it trained on ({tr_r2:.3f}); the model is "
                       "too simple for the pattern, or the features carry little signal")
    if std is not None and std > UNSTABLE_CV_STD:
        reasons.append(f"scores swing between folds (R\u00b2 std {std:.3f} > {UNSTABLE_CV_STD})")
        if status == "good":
            status = "unstable"
    if status == "good":
        reasons.append("train and held-out scores agree and it clearly beats the baseline")
    return status, reasons, skill


# ---- one model: (tune) -> fit -> predict -> score ---------------------------

def _fit_and_score(name, est, *, X_train, y_train, X_test, y_test_orig, X_val, y_val_orig,
                   target_transform, cv, groups, scorers, tuning, n_iter, nested_cv,
                   cv_strategy, random_state, selection_metric="rmse") -> ModelResult:
    inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
    n_train = len(y_train)
    n_splits = cv.get_n_splits(X_train, y_train, groups)
    t0 = time.time()
    best_params = None
    cv_dict: dict | None = None
    cv_optimistic = False
    space = _search_space(name, n_train, n_splits) if tuning != "off" else None

    if space and n_iter > 0:
        # Search on single K-fold even when evaluation repeats it (3x cheaper).
        search_cv = KFold(cv.cvargs["n_splits"], shuffle=True, random_state=random_state) \
            if isinstance(cv, RepeatedKFold) else cv
        search = RandomizedSearchCV(
            est, space, n_iter=n_iter, cv=search_cv, refit=selection_metric,
            scoring={"r2": scorers["r2"], "rmse": scorers["rmse"], "mae": scorers["mae"]},
            return_train_score=True, n_jobs=-1, random_state=random_state,
            error_score=np.nan,
        )
        try:
            with _quiet_fit():
                search.fit(X_train, y_train, groups=groups)
            est = search.best_estimator_
            best_params = _strip_prefix(search.best_params_)
            res, i = search.cv_results_, search.best_index_
            k = sum(1 for key in res if re.fullmatch(r"split\d+_test_rmse", key))
            cv_dict = {
                "R2": [float(res[f"split{j}_test_r2"][i]) for j in range(k)],
                "RMSE": [float(-res[f"split{j}_test_rmse"][i]) for j in range(k)],
                "MAE": [float(-res[f"split{j}_test_mae"][i]) for j in range(k)],
                "R2_train": [float(res[f"split{j}_train_r2"][i]) for j in range(k)],
                "RMSE_train": [float(-res[f"split{j}_train_rmse"][i]) for j in range(k)],
            }
            cv_optimistic = True
            if nested_cv and cv_strategy != "group":
                inner = TimeSeriesSplit(3) if cv_strategy == "time" else \
                    KFold(3, shuffle=True, random_state=random_state)
                nested = RandomizedSearchCV(clone(search.estimator), space, n_iter=n_iter,
                                            cv=inner, scoring=scorers[selection_metric],
                                            refit=True,
                                            n_jobs=1, random_state=random_state,
                                            error_score=np.nan)
                cv_dict = _cv_scores(nested, X_train, y_train, cv, None, scorers)
                cv_optimistic = False
        except Exception:
            best_params, cv_dict, cv_optimistic = None, None, False
            est = clone(est)

    if best_params is None:
        with _quiet_fit():
            est.fit(X_train, y_train)
    elapsed = time.time() - t0

    with _quiet_fit():
        y_pred_train = inv(est.predict(X_train))
        y_pred_test = inv(est.predict(X_test))
    y_train_orig = inv(y_train)
    metrics = _compute_metrics(y_test_orig, y_pred_test)
    train_m = _compute_metrics(y_train_orig, y_pred_train)
    metrics["R2_train"] = train_m["R2"]
    metrics["RMSE_train"] = train_m["RMSE"]

    val_metrics: dict[str, float] = {}
    if X_val is not None and y_val_orig is not None and len(y_val_orig) > 0:
        with _quiet_fit():
            y_pred_val = inv(est.predict(X_val))
        val_metrics = _compute_metrics(y_val_orig, y_pred_val)
        metrics["RMSE_val"] = val_metrics["RMSE"]
        metrics["MAE_val"] = val_metrics["MAE"]
        metrics["R2_val"] = val_metrics["R2"]

    if cv_dict is None:
        try:
            cv_dict = _cv_scores(est, X_train, y_train, cv, groups, scorers)
        except Exception:
            cv_dict = {}
    _cv_into_metrics(metrics, cv_dict, y_scale=float(np.std(y_train_orig)))

    return ModelResult(
        name=name, estimator=est, y_pred_train=y_pred_train, y_pred_test=y_pred_test,
        metrics=metrics, cv_scores=cv_dict, train_time_sec=elapsed,
        feature_importances=_extract_feature_importance(est, X_train.shape[1]),
        best_params=best_params, val_metrics=val_metrics, cv_optimistic=cv_optimistic,
    )


def _safe_member_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


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
    tuning: str | None = None,
    n_iter: int | None = None,
    nested_cv: bool = False,
    auto_remediate: bool = True,
    build_ensemble: bool = True,
    progress_callback=None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    random_state: int = 42,
    failures: dict[str, str] | None = None,
    selection_metric: str = "rmse",
) -> dict[str, ModelResult]:
    """Train, diagnose, remediate and (optionally) ensemble. Metrics in ORIGINAL units.

    `tuning` is "off" | "fast" | "thorough". The older boolean
    `tune_hyperparameters=True` still works and means "fast".
    `selection_metric` is "rmse" (default) or "mae" (robust to outliers); it
    drives tuning, remediation and the ensemble's member choice.
    """
    selection_metric = selection_metric.lower()
    if selection_metric not in ("rmse", "mae"):
        raise ValueError("selection_metric must be 'rmse' or 'mae'")
    sel_cv_key = f"CV_{selection_metric.upper()}_mean"
    if models is None:
        models = get_default_model_zoo(random_state)
    if tuning is None:
        tuning = "fast" if tune_hyperparameters else "off"
    if tuning not in TUNING_TRIALS:
        raise ValueError(f"tuning must be one of {list(TUNING_TRIALS)}")
    trials = n_iter if n_iter is not None else TUNING_TRIALS[tuning]
    failures = failures if failures is not None else {}

    cv, group_arg = _make_cv(cv_strategy, cv_folds, groups_train, random_state,
                             n_samples=len(y_train))
    scorers = _original_unit_scorers(target_transform)
    inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
    y_test_orig = inv(y_test)
    y_val_orig = inv(y_val) if y_val is not None else None
    n_splits = cv.get_n_splits(X_train, y_train, group_arg)
    max_neighbors = max(1, int(len(y_train) * (1 - 1 / max(n_splits, 2))) - 1)

    common = dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test_orig=y_test_orig,
                  X_val=X_val, y_val_orig=y_val_orig, target_transform=target_transform,
                  cv=cv, groups=group_arg, scorers=scorers, nested_cv=nested_cv,
                  cv_strategy=cv_strategy, random_state=random_state,
                  selection_metric=selection_metric)

    # Reference error of "always predict the mean" on the same folds. Used to
    # judge underfitting even if the user unticked the baseline model.
    try:
        ref_model = DummyRegressor(strategy="median" if selection_metric == "mae" else "mean")
        base_cv = _cv_scores(ref_model, X_train, y_train, cv, group_arg, scorers)
        baseline_cv_rmse = float(np.mean(base_cv["MAE" if selection_metric == "mae" else "RMSE"]))
    except Exception:
        baseline_cv_rmse = None

    results: dict[str, ModelResult] = {}
    n_models = len(models)
    for idx, (name, est) in enumerate(models.items(), start=1):
        if progress_callback:
            progress_callback(idx, n_models, name + (" (tuning)" if tuning != "off"
                                                     and name != BASELINE_MODEL_NAME else ""))
        try:
            r = _fit_and_score(name, est, tuning=tuning, n_iter=trials, **common)
        except Exception as e:  # one broken model must not sink the run
            failures[name] = f"{type(e).__name__}: {e}"
            continue
        r.fit_status, r.fit_reasons, skill = diagnose_fit(r.metrics, baseline_cv_rmse, selection_metric)
        if skill is not None:
            r.metrics["Skill_vs_baseline_pct"] = skill * 100
        results[name] = r

    # ---- automatic remediation ----
    if auto_remediate:
        for name in list(results):
            r = results[name]
            if name == BASELINE_MODEL_NAME or r.fit_status not in ("overfit", "underfit"):
                continue
            candidates = REMEDIES.get(name, {}).get(r.fit_status, [])
            if not candidates:
                r.remediation = (f"{r.fit_status} detected; no automatic fix defined for this "
                                 "model type" + ("" if r.fit_status == "overfit" else
                                                 " \u2014 try the interaction-features option"))
                continue
            if progress_callback:
                progress_callback(n_models, n_models, f"{name} \u2014 fixing {r.fit_status}")
            best_r, best_p = r, None
            old = r.metrics.get(sel_cv_key, math.inf)
            for params in candidates:
                try:
                    cand_est = _with_params(r.estimator, params, max_neighbors=max_neighbors)
                    cand = _fit_and_score(name, cand_est, tuning="off", n_iter=0, **common)
                except Exception:
                    continue
                if cand.metrics.get(sel_cv_key, math.inf) < \
                        best_r.metrics.get(sel_cv_key, math.inf) * (1 - REMEDY_MIN_IMPROVEMENT):
                    best_r, best_p = cand, params
            if best_p is not None:
                new = best_r.metrics[sel_cv_key]
                best_r.best_params = {**(r.best_params or {}), **best_p}
                best_r.fit_status, best_r.fit_reasons, skill = diagnose_fit(
                    best_r.metrics, baseline_cv_rmse, selection_metric)
                if skill is not None:
                    best_r.metrics["Skill_vs_baseline_pct"] = skill * 100
                best_r.remediation = (f"{r.fit_status} detected \u2192 retrained with {best_p}; "
                                      f"CV {selection_metric.upper()} {old:.4g} \u2192 {new:.4g}; "
                                      f"now: {best_r.fit_status}")
                best_r.train_time_sec += r.train_time_sec
                results[name] = best_r
            else:
                r.remediation = (f"{r.fit_status} detected; tried {len(candidates)} alternative "
                                 f"setting(s), none cut CV {selection_metric.upper()} by \u2265"
                                 f"{REMEDY_MIN_IMPROVEMENT:.0%} \u2014 kept the original")
                if r.fit_status == "overfit":
                    # Regularising harder made held-out error no better, so the
                    # train/held-out gap is not costing accuracy. Common for
                    # forests and boosting, which always fit training rows closely.
                    r.fit_status = "benign"
                    r.fit_reasons.append("stronger regularisation did not lower held-out error, "
                                         "so this gap is not costing accuracy")

    # ---- ensemble of the three best distinct models ----
    if build_ensemble:
        key = f"{selection_metric.upper()}_val" if X_val is not None else sel_cv_key
        ranked = sorted(
            (n for n, r in results.items()
             if n != BASELINE_MODEL_NAME and np.isfinite(r.metrics.get(key, np.nan))),
            key=lambda n: results[n].metrics[key],
        )
        if len(ranked) >= 3:
            top = ranked[:3]
            if progress_callback:
                progress_callback(n_models, n_models, "Ensemble of " + ", ".join(top))
            try:
                ens = VotingRegressor([(_safe_member_name(n), clone(results[n].estimator))
                                       for n in top])
                r = _fit_and_score(ENSEMBLE_NAME, ens, tuning="off", n_iter=0, **common)
                r.ensemble_members = top
                r.fit_status, r.fit_reasons, skill = diagnose_fit(r.metrics, baseline_cv_rmse, selection_metric)
                if skill is not None:
                    r.metrics["Skill_vs_baseline_pct"] = skill * 100
                results[ENSEMBLE_NAME] = r
            except Exception as e:
                failures[ENSEMBLE_NAME] = f"{type(e).__name__}: {e}"

    return results


# ---- ranking & selection -----------------------------------------------------

def rank_models(results: dict[str, ModelResult]) -> list[tuple[str, float, float]]:
    """All models (baseline included) by TEST RMSE — for display only."""
    rows = [(n, r.metrics["RMSE"], r.metrics["R2"]) for n, r in results.items()]
    rows.sort(key=lambda row: (row[1], -row[2]))
    return rows


def selection_basis(results: dict[str, ModelResult]) -> str:
    """'validation' if every model has a validation score, else 'cv', else 'test'."""
    if not results:
        return "test"
    vals = [r.metrics.get("RMSE_val") for r in results.values()]
    if all(v is not None and not np.isnan(v) for v in vals):
        return "validation"
    cands = [r for n, r in results.items() if n != BASELINE_MODEL_NAME] or list(results.values())
    if all(np.isfinite(r.metrics.get("CV_RMSE_mean", np.nan)) for r in cands):
        return "cv"
    return "test"


def explain_selection(results: dict[str, ModelResult], *, one_se_rule: bool = True,
                      selection_metric: str = "rmse") -> tuple[str, str]:
    """Return (winner, plain-English note on how it was chosen)."""
    if not results:
        raise ValueError("No models were trained.")
    basis = selection_basis(results)
    M = selection_metric.upper()
    key, tie = {"validation": (f"{M}_val", "R2_val"), "cv": (f"CV_{M}_mean", "CV_R2_mean"),
                "test": (M, "R2")}[basis]
    cands = {n: r for n, r in results.items() if n != BASELINE_MODEL_NAME} or results
    ordered = sorted(cands.items(), key=lambda kv: (kv[1].metrics.get(key, math.inf),
                                                    -kv[1].metrics.get(tie, -math.inf)))
    best_name, best = ordered[0]
    label = {"validation": f"validation-set {M}", "cv": f"cross-validated {M}",
             "test": f"test {M}"}[basis]
    note = f"lowest {label}"
    if one_se_rule and basis == "cv":
        m = best.metrics
        se = m.get(f"CV_{M}_std", 0.0) / math.sqrt(max(m.get("CV_n_splits", 1), 1))
        limit = m[key] + se
        within = [(n, r) for n, r in ordered if r.metrics.get(key, math.inf) <= limit]
        simplest = min(within, key=lambda kv: (COMPLEXITY.get(kv[0], 5), kv[1].metrics[key]))
        if simplest[0] != best_name:
            note = (f"one-standard-error rule: {simplest[0]} is within one standard error of "
                    f"the best cross-validated {M} ({best_name}: {m[key]:.4g} \u00b1 {se:.2g}) "
                    "and is a simpler model, so it is preferred")
            return simplest[0], note
    return best_name, note


def pick_best(results: dict[str, ModelResult], *, one_se_rule: bool = True,
              selection_metric: str = "rmse") -> str:
    """Winner: validation file if supplied, else CV error (+1-SE rule), never the baseline."""
    return explain_selection(results, one_se_rule=one_se_rule,
                             selection_metric=selection_metric)[0]


def target_outlier_share(y: np.ndarray) -> float:
    """Share of target values beyond 3 IQRs from the quartiles (extreme outliers)."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(y) < 20:
        return 0.0
    q1, q3 = np.percentile(y, [25, 75]); iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    return float(np.mean((y < q1 - 3 * iqr) | (y > q3 + 3 * iqr)))


def resolve_selection_metric(choice: str, y_train: np.ndarray) -> tuple[str, str]:
    """'auto' -> 'mae' when >2% of targets are extreme outliers, else 'rmse'."""
    c = (choice or "auto").lower()
    if c in ("rmse", "mae"):
        return c, f"{c.upper()} (chosen by you)"
    share = target_outlier_share(y_train)
    if share > 0.02:
        return "mae", (f"MAE, because {share:.1%} of training targets are extreme outliers; "
                       "RMSE would let those few rows decide the winner")
    return "rmse", "RMSE (no heavy outliers in the target)"


# ---- winner-only extras --------------------------------------------------------

def compute_permutation_importance(estimator, X, y, *, n_repeats: int = 10,
                                   random_state: int = 42, scoring: str = "r2"):
    """Permutation importance on HELD-OUT data (works for any model)."""
    try:
        with _quiet_fit():
            result = permutation_importance(estimator, X, y, n_repeats=n_repeats,
                                            random_state=random_state, scoring=scoring,
                                            n_jobs=-1)
        return np.asarray(result.importances_mean), np.asarray(result.importances_std)
    except Exception:
        return None, None


def compute_learning_curve(estimator, X, y, *, cv_strategy="kfold", cv_folds=5, groups=None,
                           target_transform="none", random_state=42, n_points=6) -> dict | None:
    """Train vs held-out R2 as the training set grows, plus a plain-English reading."""
    cv, g = _make_cv(cv_strategy, cv_folds, groups, random_state)
    scorer = _original_unit_scorers(target_transform)["r2"]
    try:
        with _quiet_fit():
            sizes, tr, va = learning_curve(clone(estimator), X, y, cv=cv, groups=g,
                                           train_sizes=np.linspace(0.2, 1.0, n_points),
                                           scoring=scorer, n_jobs=-1, error_score=np.nan)
    except Exception:
        return None
    tr_m, va_m = np.nanmean(tr, axis=1), np.nanmean(va, axis=1)
    ok = np.isfinite(tr_m) & np.isfinite(va_m)
    if ok.sum() < 2:
        return None
    t_end, v_end = tr_m[ok][-1], va_m[ok][-1]
    gap, slope = t_end - v_end, va_m[ok][-1] - va_m[ok][-2]
    if v_end < UNDERFIT_LOW_TRAIN_R2 and gap < 0.1:
        reading = ("Both curves are low and close together: the model is underfitting. More rows "
                   "won't help much; a more flexible model or better features will.")
    elif gap > 0.1 and slope > 0.005:
        reading = ("There is a gap and the held-out score is still rising: collecting more data "
                   "is likely to improve this model.")
    elif gap > 0.1:
        reading = ("A persistent gap between training and held-out scores: some overfitting. "
                   "Stronger regularisation (or a simpler model) should help.")
    else:
        reading = "The curves meet at a good level: the model is well balanced for this data."
    return {"train_sizes": sizes.tolist(), "train_mean": tr_m.tolist(),
            "train_std": np.nanstd(tr, axis=1).tolist(), "val_mean": va_m.tolist(),
            "val_std": np.nanstd(va, axis=1).tolist(), "reading": reading}


def compute_prediction_interval(estimator, X, y, *, cv_strategy="kfold", cv_folds=5,
                                groups=None, target_transform="none", random_state=42,
                                coverage=0.90, X_test=None, y_test=None) -> dict | None:
    """Split-conformal prediction interval from out-of-fold residuals.

    Residuals are measured in the model's own space (log space if log1p was
    used), so intervals become multiplicative in original units, which is
    what you want for skewed targets. Coverage is then checked on the test set.
    """
    cv, g = _make_cv(cv_strategy, cv_folds, groups, random_state)
    res: list[float] = []
    try:
        for tr_idx, te_idx in cv.split(X, y, g):
            m = clone(estimator)
            with _quiet_fit():
                m.fit(X[tr_idx], y[tr_idx])
                res.extend(np.abs(y[te_idx] - m.predict(X[te_idx])).tolist())
    except Exception:
        return None
    if len(res) < 10:
        return None
    n = len(res)
    q = float(np.quantile(res, min(1.0, math.ceil((n + 1) * coverage) / n)))
    out = {"coverage_target": coverage, "halfwidth": q,
           "space": "log1p" if target_transform == "log1p" else "original"}
    if X_test is not None and y_test is not None:
        with _quiet_fit():
            p = estimator.predict(X_test)
        lo, hi = p - q, p + q
        out["test_coverage"] = float(np.mean((y_test >= lo) & (y_test <= hi)))
        inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
        out["mean_width_original_units"] = float(np.mean(inv(hi) - inv(lo)))
    return out


def predict_with_interval(bundle: dict, X_raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience for users of best_model.joblib: (prediction, lower, upper)."""
    X = bundle["preprocessor"].transform(X_raw)
    p = bundle["model"].predict(X)
    q = (bundle.get("prediction_interval") or {}).get("halfwidth", 0.0)
    inv = np.expm1 if bundle.get("target_transform") == "log1p" else (lambda a: a)
    return inv(p), inv(p - q), inv(p + q)
