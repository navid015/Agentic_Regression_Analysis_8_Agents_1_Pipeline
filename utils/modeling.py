"""
Model training, evaluation and selection for regression (v3).

What changed and why
--------------------
* **One set of evaluation folds for every model.** Folds are built once and
  shared, so model-vs-model differences are *paired* and fair. (Before,
  small-data tuned models were scored on 5 single-KFold splits while untuned
  ones got 15 repeated splits.)
* **Preprocessing is fitted inside every fold** when a preprocessor template
  is passed (the app always does): imputation, scaling and target encoding
  never see the held-out rows.
* **Tuning searches on different folds from the ones used for comparison**
  (for shuffled K-fold), so a tuned model's reported CV score is no longer the
  maximum of its own search.
* **Selection uses a corrected paired test** (Nadeau & Bengio's corrected
  resampled t-test) for the one-standard-error rule, for accepting automatic
  fixes, and for a new **"no reliable signal" check**: if the best model is not
  significantly better than predicting the mean (Bonferroni-corrected for the
  number of models tried), the baseline wins and the app says so.
* **A model whose CV fails is excluded and reported** - selection never falls
  back silently to the test set.
* **Relative fit diagnosis.** A flexible "capacity probe" (gradient boosting)
  is run on the same folds. A model is called *underfit* only if something
  demonstrably does better; if nothing beats the mean, the data are
  *low-signal* and no capacity is added.
* **Early stopping for XGBoost / LightGBM** (and for HistGradientBoosting in
  time-ordered data, using the most recent rows as the validation slice).
* **Diverse ensemble**: the best model of each of the top three model
  families, scored from stored fold predictions (no refits).
* **Baseline in original units** under log1p (was a geometric mean, which
  inflated every model's "skill").
* **Winner extras**: permutation importance per ORIGINAL column in original
  units; locally-adaptive conformal intervals reusing the winner's
  out-of-fold residuals; Duan smearing correction for log-target models
  when the selection metric is RMSE.
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
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, uniform
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,  # noqa: F401  (re-exported for old pickles / callers)
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
    learning_curve,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@contextmanager
def _quiet_fit():
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
#: less than this fractional error improvement over the mean baseline -> no skill
UNDERFIT_MIN_SKILL = 0.05
#: train-fold R2 below this (with a small gap) -> candidate for underfitting
UNDERFIT_LOW_TRAIN_R2 = 0.30
#: "underfit" needs another model (or the capacity probe) this much better
UNDERFIT_BETTER_BY = 0.10
#: fold-to-fold R2 standard deviation above this -> unstable
UNSTABLE_CV_STD = 0.15
#: a remedy must cut CV error by at least this fraction AND one corrected SE
REMEDY_MIN_IMPROVEMENT = 0.01
#: "no reliable signal" = best CV skill <= NO_SIGNAL_MIN_SKILL, or skill below
#: UNDERFIT_MIN_SKILL AND not significant (one-sided, corrected paired t-test)
NO_SIGNAL_MIN_SKILL = 0.01
NO_SIGNAL_ALPHA = 0.05
SMALL_DATA_ROWS = 300
SVR_MAX_ROWS = 15000
TUNING_TRIALS = {"off": 0, "fast": 10, "thorough": 40}
#: boosting rounds ceiling when early stopping picks the actual number
EARLY_STOP_MAX_ROUNDS = 2000
EARLY_STOP_PATIENCE = 50


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
    fit_status: str = "unknown"  # good|benign|overfit|underfit|unstable|low_signal|cv_failed|reference
    fit_reasons: list[str] = field(default_factory=list)
    remediation: str | None = None
    cv_optimistic: bool = False
    ensemble_members: list[str] | None = None
    learning_curve: dict | None = None
    prediction_interval: dict | None = None
    # ---- v3 ----
    family: str = "other"
    spec_estimator: Any = None            # unfitted, fully configured (for refits / clones)
    fold_test_pred: list | None = None    # model-space predictions per shared fold
    fold_train_pred: list | None = None
    y_pred_train_model: np.ndarray | None = None
    y_pred_val_model: np.ndarray | None = None
    y_pred_test_model: np.ndarray | None = None
    importance_feature_names: list[str] | None = None
    permutation_feature_names: list[str] | None = None
    smearing_factor: float | None = None


# ---- names, families, complexity -------------------------------------------

BASELINE_MODEL_NAME = "Baseline (mean)"
ENSEMBLE_NAME = "Ensemble (top-3 families)"
LEGACY_ENSEMBLE_NAME = "Ensemble (top-3 average)"

COMPLEXITY = {
    BASELINE_MODEL_NAME: 0,
    "LinearRegression": 1, "Ridge": 1, "Lasso": 1, "ElasticNet": 1,
    "Huber": 1, "PoissonRegressor": 1,
    "KNN": 2, "DecisionTree": 2, "SVR": 3,
    "RandomForest": 4, "ExtraTrees": 4,
    "GradientBoosting": 5, "HistGradientBoosting": 5, "XGBoost": 5, "LightGBM": 5,
    ENSEMBLE_NAME: 6, LEGACY_ENSEMBLE_NAME: 6,
}

FAMILY = {
    BASELINE_MODEL_NAME: "baseline",
    "LinearRegression": "linear", "Ridge": "linear", "Lasso": "linear", "ElasticNet": "linear",
    "Huber": "linear", "PoissonRegressor": "linear",
    "DecisionTree": "tree", "RandomForest": "bagging", "ExtraTrees": "bagging",
    "GradientBoosting": "boosting", "HistGradientBoosting": "boosting",
    "XGBoost": "boosting", "LightGBM": "boosting",
    "KNN": "neighbors", "SVR": "kernel", ENSEMBLE_NAME: "ensemble",
}
LINEAR_FAMILY_MODELS = {n for n, f in FAMILY.items() if f == "linear"}

#: target is standardised while fitting (loss depends on the scale of y)
TARGET_SCALED = {"Ridge", "Lasso", "ElasticNet", "Huber", "SVR"}


# ---- custom estimators ---------------------------------------------------------

def _n_rows(X) -> int:
    return X.shape[0] if hasattr(X, "shape") else len(X)


def _take(X, idx):
    return X.iloc[idx] if isinstance(X, (pd.DataFrame, pd.Series)) else X[idx]


class OriginalScaleMean(BaseEstimator, RegressorMixin):
    """Predict the mean (or median) of the target in ORIGINAL units.

    With a log1p target, sklearn's DummyRegressor predicts the mean of
    log(y) - a geometric mean, systematically below the real mean - which made
    every model look more skilful than it was.
    """

    def __init__(self, strategy: str = "mean", target_transform: str = "none"):
        self.strategy = strategy
        self.target_transform = target_transform

    def fit(self, X, y):
        y = np.asarray(y, float)
        orig = np.expm1(y) if self.target_transform == "log1p" else y
        c = float(np.median(orig) if self.strategy == "median" else np.mean(orig))
        self.constant_ = float(np.log1p(c)) if self.target_transform == "log1p" else c
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else 1
        return self

    def predict(self, X):
        return np.full(_n_rows(X), self.constant_, dtype=float)


class EarlyStoppingRegressor(BaseEstimator, RegressorMixin):
    """Fit a booster with early stopping on an internal validation slice.

    sklearn's CV machinery cannot pass `eval_set`, so XGBoost / LightGBM used
    to run a fixed 600 rounds. This wrapper holds out `validation_fraction` of
    the rows it is given - the MOST RECENT rows when `time_ordered` (rows are
    in time order in time-aware mode), a random slice otherwise - and stops
    when the validation loss has not improved for `patience` rounds.
    """

    def __init__(self, estimator=None, validation_fraction: float = 0.1,
                 patience: int = EARLY_STOP_PATIENCE, time_ordered: bool = False,
                 random_state: int = 0):
        self.estimator = estimator
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.time_ordered = time_ordered
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, float)
        n = len(y)
        n_val = int(round(n * self.validation_fraction))
        est = clone(self.estimator)
        kind = type(est).__name__
        self.best_iteration_ = None
        if n < 50 or n_val < 10:
            with _quiet_fit():
                est.fit(X, y)
            self.estimator_ = est
            return self
        if self.time_ordered:
            tr, va = np.arange(n - n_val), np.arange(n - n_val, n)
        else:
            perm = np.random.default_rng(self.random_state).permutation(n)
            tr, va = np.sort(perm[n_val:]), np.sort(perm[:n_val])
        Xtr, Xva, ytr, yva = _take(X, tr), _take(X, va), y[tr], y[va]
        with _quiet_fit():
            if kind == "XGBRegressor":
                est.set_params(early_stopping_rounds=self.patience)
                est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                self.best_iteration_ = getattr(est, "best_iteration", None)
            elif kind == "LGBMRegressor":
                import lightgbm
                est.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                        callbacks=[lightgbm.early_stopping(self.patience, verbose=False)])
                self.best_iteration_ = getattr(est, "best_iteration_", None)
            elif kind == "HistGradientBoostingRegressor":
                est.set_params(early_stopping=True, n_iter_no_change=min(self.patience, 20))
                est.fit(Xtr, ytr, X_val=Xva, y_val=yva)
                self.best_iteration_ = getattr(est, "n_iter_", None)
            else:
                est.fit(X, y)
        self.estimator_ = est
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


class AveragingEnsemble(BaseEstimator, RegressorMixin):
    """Equal-weight average of full model pipelines (in model space)."""

    def __init__(self, estimators=None):
        self.estimators = estimators

    def fit(self, X, y):
        self.estimators_ = []
        for name, est in self.estimators:
            m = clone(est)
            with _quiet_fit():
                m.fit(X, y)
            self.estimators_.append((name, m))
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for _, m in self.estimators_], axis=0)

    @classmethod
    def from_fitted(cls, fitted: list[tuple[str, Any]], unfitted: list[tuple[str, Any]] | None = None):
        obj = cls(estimators=unfitted or fitted)
        obj.estimators_ = list(fitted)
        return obj


# ---- model zoo -----------------------------------------------------------------------

def _scaled(est):
    return TransformedTargetRegressor(regressor=est, transformer=StandardScaler())


def get_default_model_zoo(random_state: int = 42, *, time_ordered: bool = False,
                          target_transform: str = "none") -> dict[str, Any]:
    """The candidate models. Parallel-capable models use n_jobs=1 here because
    CV folds and the tuning search are parallelised one level up (nesting
    n_jobs=-1 inside n_jobs=-1 oversubscribes the CPU)."""
    rs = random_state
    zoo: dict[str, Any] = {
        BASELINE_MODEL_NAME: OriginalScaleMean(strategy="mean", target_transform=target_transform),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=rs),
        "Lasso": Lasso(alpha=0.01, random_state=rs, max_iter=20000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=rs, max_iter=20000),
        "Huber": HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000),
        "PoissonRegressor": PoissonRegressor(alpha=1e-3, max_iter=2000),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=rs),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                              random_state=rs, n_jobs=1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                          random_state=rs, n_jobs=1),
        "KNN": KNeighborsRegressor(n_neighbors=7, n_jobs=1),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"),
    }
    if time_ordered:
        # sklearn's internal early-stopping split is random: in time-ordered
        # data it validates on the past using the future. Use a fixed budget
        # for GB and a most-recent-rows validation slice for HGB instead.
        zoo["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=rs)
        zoo["HistGradientBoosting"] = EarlyStoppingRegressor(
            HistGradientBoostingRegressor(max_iter=EARLY_STOP_MAX_ROUNDS, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=rs),
            time_ordered=True, random_state=rs)
    else:
        zoo["GradientBoosting"] = GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=3, subsample=0.8,
            validation_fraction=0.1, n_iter_no_change=20, random_state=rs)
        zoo["HistGradientBoosting"] = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, random_state=rs)
    if _HAS_XGB:
        zoo["XGBoost"] = EarlyStoppingRegressor(
            XGBRegressor(n_estimators=EARLY_STOP_MAX_ROUNDS, learning_rate=0.05, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                         random_state=rs, n_jobs=1, verbosity=0),
            time_ordered=time_ordered, random_state=rs)
    if _HAS_LGBM:
        zoo["LightGBM"] = EarlyStoppingRegressor(
            LGBMRegressor(n_estimators=EARLY_STOP_MAX_ROUNDS, learning_rate=0.05, num_leaves=31,
                          min_child_samples=10, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=1.0, random_state=rs, n_jobs=1,
                          verbosity=-1),
            time_ordered=time_ordered, random_state=rs)
    order = [BASELINE_MODEL_NAME, "LinearRegression", "Ridge", "Lasso", "ElasticNet", "Huber",
             "PoissonRegressor", "DecisionTree", "RandomForest", "ExtraTrees", "GradientBoosting",
             "HistGradientBoosting", "KNN", "SVR", "XGBoost", "LightGBM"]
    return {n: (_scaled(zoo[n]) if n in TARGET_SCALED else zoo[n]) for n in order if n in zoo}


def available_model_names() -> list[str]:
    return list(get_default_model_zoo().keys())


def filter_zoo_for_data(zoo: dict[str, Any], y_train: np.ndarray, *,
                        target_transform: str = "none") -> tuple[dict[str, Any], dict[str, str]]:
    """Drop models that cannot work on this dataset, with reasons."""
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


# ---- parameter plumbing ----------------------------------------------------------------

def _inner(est):
    """Innermost learner (through Pipeline / TransformedTargetRegressor / early stopping)."""
    while True:
        if isinstance(est, Pipeline):
            est = est.steps[-1][1]
        elif isinstance(est, TransformedTargetRegressor):
            est = getattr(est, "regressor_", None) or est.regressor
        elif isinstance(est, EarlyStoppingRegressor):
            est = getattr(est, "estimator_", None) or est.estimator
        else:
            return est


def _inner_unfitted(est):
    while True:
        if isinstance(est, Pipeline):
            est = est.steps[-1][1]
        elif isinstance(est, TransformedTargetRegressor):
            est = est.regressor
        elif isinstance(est, EarlyStoppingRegressor):
            est = est.estimator
        else:
            return est


def _param_prefix(est) -> str:
    prefix = ""
    while True:
        if isinstance(est, Pipeline):
            prefix += f"{est.steps[-1][0]}__"
            est = est.steps[-1][1]
        elif isinstance(est, TransformedTargetRegressor):
            prefix += "regressor__"
            est = est.regressor
        elif isinstance(est, EarlyStoppingRegressor):
            prefix += "estimator__"
            est = est.estimator
        else:
            return prefix


def _with_params(est, params: dict[str, Any], *, max_neighbors: int | None = None):
    new = clone(est)
    p = dict(params)
    if max_neighbors is not None and "n_neighbors" in p:
        p["n_neighbors"] = max(1, min(int(p["n_neighbors"]), max_neighbors))
    prefix = _param_prefix(new)
    new.set_params(**{f"{prefix}{k}": v for k, v in p.items()})
    return new


_PREFIX_RE = re.compile(r"^(?:(?:model|regressor|estimator)__)+")


def _strip_prefix(params: dict | None) -> dict | None:
    if not params:
        return params
    return {_PREFIX_RE.sub("", k): v for k, v in params.items()}


def _set_n_jobs(est, n_jobs: int):
    """Set every nested n_jobs parameter (members of an ensemble included)."""
    if isinstance(est, AveragingEnsemble):
        for _, m in (est.estimators or []):
            _set_n_jobs(m, n_jobs)
        return est
    try:
        keys = [k for k in est.get_params(deep=True) if k == "n_jobs" or k.endswith("__n_jobs")]
        if keys:
            est.set_params(**{k: n_jobs for k in keys})
    except Exception:
        pass
    return est


def _plain_params(est) -> dict[str, Any]:
    """Literal-safe hyperparameters of the innermost learner."""
    inner = _inner_unfitted(est)
    out = {}
    for k, v in inner.get_params(deep=False).items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
    return out


def _wrap(name: str, est, preprocessor):
    """Full model = Pipeline(preprocessing, learner); ensembles wrap their members."""
    if preprocessor is None or isinstance(est, (AveragingEnsemble, Pipeline)):
        return est
    if isinstance(preprocessor, dict):
        prep = preprocessor.get("linear") if name in LINEAR_FAMILY_MODELS and "linear" in preprocessor \
            else preprocessor["default"]
    else:
        prep = preprocessor
    return Pipeline([("prep", clone(prep)), ("model", est)])


def rebuild_estimator(spec: dict, preprocessor=None, *, random_state: int = 42,
                      time_ordered: bool = False, target_transform: str = "none"):
    """Recreate a winner from its spec {'name', 'params'} or {'name', 'members'}."""
    zoo = get_default_model_zoo(random_state, time_ordered=time_ordered,
                                target_transform=target_transform)
    if spec.get("members"):
        members = [(m["name"], _wrap(m["name"], rebuild_estimator(m, None, random_state=random_state,
                                                                  time_ordered=time_ordered,
                                                                  target_transform=target_transform),
                                     preprocessor))
                   for m in spec["members"]]
        return AveragingEnsemble([(_safe_member_name(n), e) for n, e in members])
    est = zoo[spec["name"]]
    if spec.get("params"):
        est = _with_params(est, spec["params"])
    return est


def _safe_member_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# ---- search spaces and remedies ------------------------------------------------------------

def _search_space(name: str, min_train_rows: int, n_splits: int = 5) -> dict[str, Any] | None:
    max_k = max(3, min(50, int(min_train_rows) - 1))
    spaces: dict[str, dict[str, Any]] = {
        "Ridge": {"alpha": loguniform(1e-3, 1e3)},
        "Lasso": {"alpha": loguniform(1e-4, 10)},
        "ElasticNet": {"alpha": loguniform(1e-4, 10), "l1_ratio": uniform(0.05, 0.9)},
        "Huber": {"epsilon": uniform(1.1, 1.0), "alpha": loguniform(1e-6, 1)},
        "PoissonRegressor": {"alpha": loguniform(1e-6, 10)},
        "DecisionTree": {"max_depth": [3, 5, 8, 12, 20, None], "min_samples_leaf": randint(1, 40),
                         "max_features": [1.0, 0.8, 0.6]},
        "RandomForest": {"n_estimators": [200, 400], "max_depth": [None, 8, 16, 24],
                         "min_samples_leaf": randint(1, 20), "max_features": [1.0, 0.7, 0.5, "sqrt"]},
        "ExtraTrees": {"n_estimators": [200, 400], "max_depth": [None, 8, 16, 24],
                       "min_samples_leaf": randint(1, 20), "max_features": [1.0, 0.7, 0.5, "sqrt"]},
        "GradientBoosting": {"learning_rate": loguniform(0.01, 0.2), "max_depth": randint(2, 6),
                             "subsample": uniform(0.6, 0.4), "min_samples_leaf": randint(1, 30)},
        "HistGradientBoosting": {"learning_rate": loguniform(0.01, 0.2), "max_leaf_nodes": randint(8, 64),
                                 "min_samples_leaf": randint(5, 60), "l2_regularization": loguniform(1e-3, 10)},
        "KNN": {"n_neighbors": randint(2, max_k), "weights": ["uniform", "distance"]},
        "SVR": {"C": loguniform(1e-2, 1e3), "epsilon": loguniform(1e-3, 1.0), "gamma": ["scale", "auto"]},
        # rounds are chosen by early stopping, so n_estimators is not searched
        "XGBoost": {"learning_rate": loguniform(0.01, 0.3), "max_depth": randint(2, 9),
                    "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                    "min_child_weight": loguniform(0.5, 20), "reg_lambda": loguniform(1e-2, 10)},
        "LightGBM": {"learning_rate": loguniform(0.01, 0.3), "num_leaves": randint(8, 128),
                     "min_child_samples": randint(5, 60), "subsample": uniform(0.6, 0.4),
                     "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(1e-2, 10)},
    }
    return spaces.get(name)


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
    "HistGradientBoosting": {"overfit": [{"max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 10.0},
                                         {"max_leaf_nodes": 7, "min_samples_leaf": 80, "l2_regularization": 30.0,
                                          "learning_rate": 0.03}],
                             "underfit": [{"max_leaf_nodes": 63, "min_samples_leaf": 5}]},
    "XGBoost": {"overfit": [{"max_depth": 3, "min_child_weight": 10, "reg_lambda": 10.0, "subsample": 0.7},
                            {"max_depth": 2, "min_child_weight": 30, "reg_lambda": 30.0, "subsample": 0.6,
                             "learning_rate": 0.03}],
                "underfit": [{"max_depth": 8, "min_child_weight": 1}]},
    "LightGBM": {"overfit": [{"num_leaves": 15, "min_child_samples": 40, "reg_lambda": 10.0},
                             {"num_leaves": 7, "min_child_samples": 80, "reg_lambda": 30.0, "learning_rate": 0.03}],
                 "underfit": [{"num_leaves": 63, "min_child_samples": 5}]},
    "KNN": {"overfit": [{"n_neighbors": 15, "weights": "uniform"}, {"n_neighbors": 30, "weights": "uniform"}],
            "underfit": [{"n_neighbors": 3}]},
    "SVR": {"overfit": [{"C": 0.3}], "underfit": [{"C": 10.0}, {"C": 100.0}]},
    "Ridge": {"overfit": [{"alpha": 10.0}, {"alpha": 100.0}], "underfit": [{"alpha": 0.1}]},
    "Lasso": {"overfit": [{"alpha": 0.05}, {"alpha": 0.2}], "underfit": [{"alpha": 0.001}]},
    "ElasticNet": {"overfit": [{"alpha": 0.05}, {"alpha": 0.2}], "underfit": [{"alpha": 0.001}]},
    "Huber": {"overfit": [{"alpha": 0.1}], "underfit": [{"alpha": 1e-6}]},
}


# ---- metrics -----------------------------------------------------------------------------

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE_pct": _safe_mape(y_true, y_pred),
    }


def _extract_feature_importance(estimator: Any, n_features: int | None = None) -> np.ndarray | None:
    est = _inner(estimator)
    if hasattr(est, "feature_importances_"):
        try:
            return np.asarray(est.feature_importances_, dtype=float)
        except Exception:
            return None
    if hasattr(est, "coef_"):
        try:
            coef = np.abs(np.asarray(est.coef_, dtype=float).ravel())
            return coef if (n_features is None or coef.size == n_features) else None
        except Exception:
            return None
    return None


def _transformed_feature_names(estimator) -> list[str] | None:
    """Output names of the fitted preprocessing step of a model pipeline."""
    if not isinstance(estimator, Pipeline) or "prep" not in estimator.named_steps:
        return None
    try:
        names = []
        for n in estimator.named_steps["prep"].get_feature_names_out():
            n = str(n)
            for p in ("num__", "cat__", "freq__", "te__"):
                if n.startswith(p):
                    n = n[len(p):]
                    break
            names.append(n)
        return names
    except Exception:
        return None


# ---- folds -----------------------------------------------------------------------------

def _make_cv(strategy: str, n_folds: int, groups: np.ndarray | None,
             random_state: int = 42, n_samples: int | None = None):
    """KFold (repeated 3x on small data) / GroupKFold / TimeSeriesSplit."""
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_folds), None
    if strategy == "group" and groups is not None:
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("Group-aware CV needs at least 2 distinct groups in the training rows.")
        return GroupKFold(n_splits=min(n_folds, n_groups)), groups
    if n_samples is not None and n_samples < SMALL_DATA_ROWS:
        return RepeatedKFold(n_splits=n_folds, n_repeats=3, random_state=random_state), None
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_state), None


def make_folds(strategy: str, n_folds: int, y: np.ndarray, groups: np.ndarray | None = None,
               random_state: int = 42) -> dict[str, Any]:
    """Build the evaluation folds ONCE; every model is scored on exactly these."""
    if strategy == "group" and groups is None:
        strategy = "kfold"
    cv, g = _make_cv(strategy, n_folds, groups, random_state, n_samples=len(y))
    splits = [(np.asarray(tr), np.asarray(te))
              for tr, te in cv.split(np.zeros((len(y), 1)), y, g)]
    return {"splits": splits, "groups": g, "strategy": strategy, "repr": repr(cv),
            "n_repeats": 3 if isinstance(cv, RepeatedKFold) else 1,
            "ratio": float(np.mean([len(te) / max(len(tr), 1) for tr, te in splits]))}


def _search_cv(strategy: str, n_folds: int, groups, random_state: int, n: int):
    """Folds for hyperparameter SEARCH - reshuffled relative to the evaluation
    folds (where possible) so a tuned model is not scored on the very splits
    it was optimised against."""
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_folds)
    if strategy == "group" and groups is not None:
        return GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_state + 1)


def _original_unit_scorers(target_transform: str) -> dict[str, Any]:
    if target_transform != "log1p":
        return {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

    def _r2(t, p):
        return r2_score(np.expm1(t), np.expm1(p))

    def _neg_rmse(t, p):
        return -float(np.sqrt(mean_squared_error(np.expm1(t), np.expm1(p))))

    def _neg_mae(t, p):
        return -float(mean_absolute_error(np.expm1(t), np.expm1(p)))
    return {"r2": make_scorer(_r2), "rmse": make_scorer(_neg_rmse), "mae": make_scorer(_neg_mae)}


# ---- fold engine ---------------------------------------------------------------------------

def _fit_fold(est, X, y, tr, te):
    m = _set_n_jobs(clone(est), 1)
    with _quiet_fit():
        m.fit(_take(X, tr), y[tr])
        return (np.asarray(m.predict(_take(X, tr)), float),
                np.asarray(m.predict(_take(X, te)), float))


def _safe_fold(est, X, y, tr, te):
    try:
        return _fit_fold(est, X, y, tr, te), None
    except Exception as e:  # recorded, never silently turned into "use the test set"
        return None, f"{type(e).__name__}: {e}"


def _run_folds(est, X, y, splits, n_jobs: int = -1):
    if n_jobs == 1 or len(splits) <= 1:
        outs = [_safe_fold(est, X, y, tr, te) for tr, te in splits]
    else:
        outs = Parallel(n_jobs=n_jobs)(delayed(_safe_fold)(est, X, y, tr, te) for tr, te in splits)
    fold_tr = [o[0][0] if o[0] is not None else None for o in outs]
    fold_te = [o[0][1] if o[0] is not None else None for o in outs]
    errors = [o[1] for o in outs if o[1]]
    return fold_tr, fold_te, errors


def _cv_from_fold_preds(y, splits, fold_tr, fold_te, target_transform, errors=None) -> dict:
    inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
    keys = ("R2", "RMSE", "MAE", "R2_train", "RMSE_train")
    out: dict[str, Any] = {k: [] for k in keys}
    for (tr, te), ptr, pte in zip(splits, fold_tr, fold_te):
        if pte is None or ptr is None:
            for k in keys:
                out[k].append(float("nan"))
            continue
        yt, yp = inv(y[te]), inv(pte)
        ytr, yptr = inv(y[tr]), inv(ptr)
        out["R2"].append(float(r2_score(yt, yp)) if len(te) > 1 else float("nan"))
        out["RMSE"].append(float(np.sqrt(mean_squared_error(yt, yp))))
        out["MAE"].append(float(mean_absolute_error(yt, yp)))
        out["R2_train"].append(float(r2_score(ytr, yptr)) if len(tr) > 1 else float("nan"))
        out["RMSE_train"].append(float(np.sqrt(mean_squared_error(ytr, yptr))))
    out["failed_folds"] = int(sum(p is None for p in fold_te))
    out["errors"] = list(errors or [])
    return out


def _cv_into_metrics(metrics: dict, cv: dict, y_scale: float | None = None,
                     ratio: float | None = None) -> None:
    """Summarise fold scores. Also stores a fold-consistent "global" R2,
    1 - (RMSE / std of the whole training target)^2, which (unlike per-fold R2)
    stays meaningful on short time-series folds."""
    if not cv or not cv.get("R2"):
        return
    r2 = np.asarray(cv["R2"], float)
    rmse = np.asarray(cv["RMSE"], float)
    failed = int(cv.get("failed_folds", 0))
    metrics["CV_failed_folds"] = failed
    metrics["CV_n_splits"] = int(len(r2))
    if ratio is not None:
        metrics["CV_test_train_ratio"] = float(ratio)
    if failed == len(r2):
        return
    metrics["CV_R2_mean"] = float(np.nanmean(r2))
    metrics["CV_R2_std"] = float(np.nanstd(r2))
    metrics["CV_RMSE_mean"] = float(np.nanmean(rmse))
    metrics["CV_RMSE_std"] = float(np.nanstd(rmse))
    if cv.get("MAE"):
        mae = np.asarray(cv["MAE"], float)
        metrics["CV_MAE_mean"] = float(np.nanmean(mae))
        metrics["CV_MAE_std"] = float(np.nanstd(mae))
    if cv.get("R2_train"):
        metrics["CV_R2_train_mean"] = float(np.nanmean(np.asarray(cv["R2_train"], float)))
    if y_scale and y_scale > 0:
        g = 1 - (rmse / y_scale) ** 2
        cv["R2g"] = g.tolist()
        metrics["CV_R2g_mean"] = float(np.nanmean(g))
        metrics["CV_R2g_std"] = float(np.nanstd(g))
        if cv.get("RMSE_train"):
            gt = 1 - (np.asarray(cv["RMSE_train"], float) / y_scale) ** 2
            metrics["CV_R2g_train_mean"] = float(np.nanmean(gt))


def _cv_complete(r) -> bool:
    m = getattr(r, "metrics", {}) or {}
    return bool(np.isfinite(m.get("CV_RMSE_mean", np.nan)) and m.get("CV_failed_folds", 0) == 0)


def _fold_errors(r, M: str) -> np.ndarray | None:
    cv = getattr(r, "cv_scores", None) or {}
    v = cv.get(M)
    return np.asarray(v, float) if v else None


def corrected_paired_difference(a: np.ndarray, b: np.ndarray, ratio: float) -> tuple[float, float, int]:
    """mean(a - b) over shared folds and its Nadeau-Bengio corrected SE.

    Fold scores are not independent (training sets overlap), so the naive
    std/sqrt(K) is far too small; the correction multiplies the variance by
    (1/K + n_test/n_train).
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    k = len(d)
    if k == 0:
        return float("nan"), float("inf"), 0
    if k < 2:
        return float(d.mean()), float("inf"), k
    var = float(np.var(d, ddof=1))
    return float(d.mean()), math.sqrt(max((1.0 / k + ratio) * var, 0.0)), k


# ---- fit diagnosis ---------------------------------------------------------------------------

def diagnose_fit(metrics: dict[str, float], baseline_cv_rmse: float | None,
                 selection_metric: str = "rmse", best_cv_error: float | None = None
                 ) -> tuple[str, list[str], float | None]:
    """good / overfit / underfit / unstable / low_signal, from training CV only.

    `best_cv_error` (the best model's or the capacity probe's CV error) makes
    the verdict relative: a model is "underfit" only if something demonstrably
    does better. If nothing beats the mean, the verdict is "low_signal" - more
    capacity would only fit noise.
    """
    reasons: list[str] = []
    cv_r2 = metrics.get("CV_R2g_mean", metrics.get("CV_R2_mean"))
    tr_r2 = metrics.get("CV_R2g_train_mean", metrics.get("CV_R2_train_mean", metrics.get("R2_train")))
    robust = selection_metric == "mae"
    cv_err = metrics.get("CV_MAE_mean" if robust else "CV_RMSE_mean")
    std = metrics.get("CV_R2g_std", metrics.get("CV_R2_std"))
    if cv_r2 is None or cv_err is None:
        return "unknown", ["no cross-validation scores available"], None

    skill = None
    if baseline_cv_rmse and baseline_cv_rmse > 0:
        skill = 1.0 - cv_err / baseline_cv_rmse
    best_skill = (1.0 - best_cv_error / baseline_cv_rmse) \
        if best_cv_error is not None and baseline_cv_rmse else None
    better_exists = best_cv_error is None or best_cv_error <= cv_err * (1 - UNDERFIT_BETTER_BY)
    gap = (tr_r2 - cv_r2) if tr_r2 is not None else 0.0
    ref = "median" if robust else "mean"

    status = "good"
    if gap > OVERFIT_GAP:
        status = "overfit"
        reasons.append(f"train-fold R\u00b2 {tr_r2:.3f} vs held-out-fold R\u00b2 {cv_r2:.3f} "
                       f"(gap {gap:.3f} > {OVERFIT_GAP})")
        if skill is not None and skill < UNDERFIT_MIN_SKILL:
            reasons.append(f"and only {skill * 100:.1f}% better than predicting the {ref}: "
                           "it is memorising noise")
    elif skill is not None and skill < UNDERFIT_MIN_SKILL:
        if better_exists and not (best_skill is not None and best_skill < UNDERFIT_MIN_SKILL):
            status = "underfit"
            reasons.append(f"only {skill * 100:.1f}% lower error than predicting the {ref}, while "
                           "another model does clearly better: this one is too simple")
        else:
            status = "low_signal"
            reasons.append(f"only {skill * 100:.1f}% lower error than predicting the {ref}, and no "
                           "model (including a flexible reference model) does much better: the "
                           "features carry little signal, so adding capacity would fit noise")
    elif not robust and tr_r2 is not None and tr_r2 < UNDERFIT_LOW_TRAIN_R2:
        if better_exists:
            status = "underfit"
            reasons.append(f"R\u00b2 is low even on data it trained on ({tr_r2:.3f}) and a more "
                           "flexible model does clearly better")
        else:
            reasons.append(f"R\u00b2 is low even on training folds ({tr_r2:.3f}), but no model does "
                           f"\u2265{UNDERFIT_BETTER_BY:.0%} better: the target is noisy, not this model")
    if std is not None and std > UNSTABLE_CV_STD:
        reasons.append(f"scores swing between folds (R\u00b2 std {std:.3f} > {UNSTABLE_CV_STD})")
        if status == "good":
            status = "unstable"
    if status == "good" and not reasons:
        reasons.append("train and held-out scores agree and it clearly beats the baseline")
    return status, reasons, skill


# ---- one model: (tune) -> fit -> CV on the shared folds ----------------------------------------

def _nested_folds(base_full, pspace, X, y, splits, groups, strategy, n_iter, scorer,
                  random_state, n_jobs):
    ftr, fte = [], []
    for tr, te in splits:
        g_tr = groups[tr] if groups is not None else None
        if strategy == "time":
            inner = TimeSeriesSplit(3)
        elif strategy == "group" and g_tr is not None:
            inner = GroupKFold(min(3, len(np.unique(g_tr))))
        else:
            inner = KFold(3, shuffle=True, random_state=random_state)
        s = RandomizedSearchCV(clone(base_full), pspace, n_iter=n_iter, cv=inner, scoring=scorer,
                               refit=True, n_jobs=n_jobs, random_state=random_state,
                               error_score=np.nan)
        with _quiet_fit():
            s.fit(_take(X, tr), y[tr], **({"groups": g_tr} if g_tr is not None else {}))
        ftr.append(np.asarray(s.predict(_take(X, tr)), float))
        fte.append(np.asarray(s.predict(_take(X, te)), float))
    return ftr, fte


def _fit_and_score(name, est, *, X_train, y_train, X_test, y_test_orig, X_val, y_val_orig,
                   target_transform, folds, search_cv, scorers, tuning, n_iter, nested_cv,
                   random_state, selection_metric, preprocessor, n_jobs, min_train_rows,
                   **_ignored) -> ModelResult:
    inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
    t0 = time.time()
    full = _wrap(name, est, preprocessor)
    inner = _inner_unfitted(full)
    if isinstance(inner, KNeighborsRegressor) and inner.n_neighbors > max(1, min_train_rows - 1):
        # the smallest training fold (early TimeSeriesSplit folds!) must hold k neighbours
        full = _with_params(full, {"n_neighbors": max(1, min_train_rows - 1)})
    groups = folds["groups"]
    best_params, cv_optimistic = None, False
    base_full = full
    space = _search_space(name, min_train_rows) if tuning != "off" and n_iter > 0 else None
    pspace = None
    if space:
        pspace = {_param_prefix(full) + k: v for k, v in space.items()}
        search = RandomizedSearchCV(full, pspace, n_iter=n_iter, cv=search_cv, scoring=scorers,
                                    refit=False, n_jobs=n_jobs, random_state=random_state,
                                    error_score=np.nan)
        try:
            with _quiet_fit():
                search.fit(X_train, y_train, **({"groups": groups} if groups is not None else {}))
            scores = np.asarray(search.cv_results_[f"mean_test_{selection_metric}"], float)
            if np.isfinite(scores).any():
                params = search.cv_results_["params"][int(np.nanargmax(scores))]
                full = clone(full).set_params(**params)
                best_params, cv_optimistic = _strip_prefix(params), True
        except Exception:
            full = base_full

    fitted = _set_n_jobs(clone(full), -1)
    with _quiet_fit():
        fitted.fit(X_train, y_train)          # a failure here is reported by the caller
        p_train = np.asarray(fitted.predict(X_train), float)
        p_test = np.asarray(fitted.predict(X_test), float)
        p_val = np.asarray(fitted.predict(X_val), float) if X_val is not None else None
    elapsed = time.time() - t0

    fold_tr = fold_te = None
    errors: list[str] = []
    if cv_optimistic and nested_cv and pspace:
        try:
            fold_tr, fold_te = _nested_folds(base_full, pspace, X_train, y_train, folds["splits"],
                                             groups, folds["strategy"], n_iter,
                                             scorers[selection_metric], random_state, n_jobs)
            cv_optimistic = False
        except Exception as e:
            errors.append(f"nested CV failed ({type(e).__name__}); used plain CV")
            fold_tr = fold_te = None
    if fold_te is None:
        fold_tr, fold_te, errs = _run_folds(full, X_train, y_train, folds["splits"], n_jobs)
        errors += errs
    cv = _cv_from_fold_preds(y_train, folds["splits"], fold_tr, fold_te, target_transform, errors)

    y_train_orig = inv(y_train)
    metrics = _compute_metrics(y_test_orig, inv(p_test))
    tm = _compute_metrics(y_train_orig, inv(p_train))
    metrics["R2_train"], metrics["RMSE_train"] = tm["R2"], tm["RMSE"]
    val_metrics: dict[str, float] = {}
    if p_val is not None and y_val_orig is not None and len(y_val_orig) > 0:
        val_metrics = _compute_metrics(y_val_orig, inv(p_val))
        metrics["RMSE_val"], metrics["MAE_val"], metrics["R2_val"] = \
            val_metrics["RMSE"], val_metrics["MAE"], val_metrics["R2"]
    _cv_into_metrics(metrics, cv, y_scale=float(np.std(y_train_orig)), ratio=folds["ratio"])
    return ModelResult(
        name=name, estimator=fitted, y_pred_train=inv(p_train), y_pred_test=inv(p_test),
        metrics=metrics, cv_scores=cv, train_time_sec=elapsed,
        feature_importances=_extract_feature_importance(fitted),
        best_params=best_params, val_metrics=val_metrics, cv_optimistic=cv_optimistic,
        family=FAMILY.get(name, "other"), spec_estimator=full,
        fold_test_pred=fold_te, fold_train_pred=fold_tr,
        y_pred_train_model=p_train, y_pred_val_model=p_val, y_pred_test_model=p_test,
        importance_feature_names=_transformed_feature_names(fitted),
    )


def _ensemble_result(members: list[str], results: dict[str, ModelResult], *, y_train,
                     y_test_orig, y_val_orig, folds, target_transform) -> ModelResult:
    inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
    k = len(folds["splits"])

    def _avg(attr, i=None):
        vals = [getattr(results[m], attr) if i is None else getattr(results[m], attr)[i] for m in members]
        return None if any(v is None for v in vals) else np.mean(vals, axis=0)
    fold_tr = [_avg("fold_train_pred", i) for i in range(k)]
    fold_te = [_avg("fold_test_pred", i) for i in range(k)]
    p_train, p_test, p_val = _avg("y_pred_train_model"), _avg("y_pred_test_model"), _avg("y_pred_val_model")
    cv = _cv_from_fold_preds(y_train, folds["splits"], fold_tr, fold_te, target_transform)
    metrics = _compute_metrics(y_test_orig, inv(p_test))
    tm = _compute_metrics(inv(y_train), inv(p_train))
    metrics["R2_train"], metrics["RMSE_train"] = tm["R2"], tm["RMSE"]
    val_metrics: dict[str, float] = {}
    if p_val is not None and y_val_orig is not None:
        val_metrics = _compute_metrics(y_val_orig, inv(p_val))
        metrics["RMSE_val"], metrics["MAE_val"], metrics["R2_val"] = \
            val_metrics["RMSE"], val_metrics["MAE"], val_metrics["R2"]
    _cv_into_metrics(metrics, cv, y_scale=float(np.std(inv(y_train))), ratio=folds["ratio"])
    names = [_safe_member_name(m) for m in members]
    fitted = AveragingEnsemble.from_fitted(
        [(n, results[m].estimator) for n, m in zip(names, members)],
        unfitted=[(n, results[m].spec_estimator) for n, m in zip(names, members)])
    return ModelResult(
        name=ENSEMBLE_NAME, estimator=fitted, y_pred_train=inv(p_train), y_pred_test=inv(p_test),
        metrics=metrics, cv_scores=cv, val_metrics=val_metrics,
        train_time_sec=float(sum(results[m].train_time_sec for m in members)),
        ensemble_members=list(members), family="ensemble",
        cv_optimistic=any(results[m].cv_optimistic for m in members),
        spec_estimator=AveragingEnsemble([(n, results[m].spec_estimator) for n, m in zip(names, members)]),
        fold_test_pred=fold_te, fold_train_pred=fold_tr, y_pred_train_model=p_train,
        y_pred_val_model=p_val, y_pred_test_model=p_test)


def _pick_ensemble_members(results, key: str, k: int = 3) -> list[str]:
    """Best model of each of the top-k model FAMILIES (three boosters averaged
    together add little); topped up with the next-best models if fewer
    families are available."""
    ranked = sorted((n for n, r in results.items()
                     if n not in (BASELINE_MODEL_NAME, ENSEMBLE_NAME) and _cv_complete(r)
                     and r.fold_test_pred is not None and np.isfinite(r.metrics.get(key, np.nan))),
                    key=lambda n: results[n].metrics[key])
    members, fams = [], set()
    for n in ranked:
        if results[n].family not in fams:
            members.append(n)
            fams.add(results[n].family)
        if len(members) == k:
            return members
    for n in ranked:
        if len(members) == k:
            break
        if n not in members:
            members.append(n)
    return members


# ---- main training driver ----------------------------------------------------------------------

def _capacity_probe(time_ordered: bool, random_state: int):
    hgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, random_state=random_state,
                                        early_stopping=not time_ordered, validation_fraction=0.1,
                                        n_iter_no_change=20)
    return EarlyStoppingRegressor(hgb, time_ordered=True, random_state=random_state) \
        if time_ordered else hgb


def train_and_evaluate(
    X_train, X_test, y_train, y_test, *,
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
    X_val=None,
    y_val: np.ndarray | None = None,
    random_state: int = 42,
    failures: dict[str, str] | None = None,
    selection_metric: str = "rmse",
    preprocessor=None,
    capacity_probe: bool = True,
    n_jobs: int = -1,
    folds: dict | None = None,
    diagnostics: dict | None = None,
) -> dict[str, ModelResult]:
    """Train, diagnose, remediate and ensemble. Metrics in ORIGINAL units.

    Pass raw engineered frames plus `preprocessor` (a ColumnTransformer
    template, or {"default": ..., "linear": ...}) to fit preprocessing inside
    every fold; with plain arrays and no preprocessor it behaves as before.
    """
    selection_metric = selection_metric.lower()
    if selection_metric not in ("rmse", "mae"):
        raise ValueError("selection_metric must be 'rmse' or 'mae'")
    M = selection_metric.upper()
    sel_cv_key = f"CV_{M}_mean"
    y_train = np.asarray(y_train, float)
    y_test = np.asarray(y_test, float)
    time_ordered = cv_strategy == "time"
    if models is None:
        models = get_default_model_zoo(random_state, time_ordered=time_ordered,
                                       target_transform=target_transform)
    for est in models.values():
        if isinstance(est, OriginalScaleMean):
            est.set_params(target_transform=target_transform)
    if tuning is None:
        tuning = "fast" if tune_hyperparameters else "off"
    if tuning not in TUNING_TRIALS:
        raise ValueError(f"tuning must be one of {list(TUNING_TRIALS)}")
    trials = n_iter if n_iter is not None else TUNING_TRIALS[tuning]
    failures = failures if failures is not None else {}
    diagnostics = diagnostics if diagnostics is not None else {}

    if cv_strategy == "group" and groups_train is None:
        cv_strategy = "kfold"
    folds = folds or make_folds(cv_strategy, cv_folds, y_train, groups_train, random_state)
    splits = folds["splits"]
    min_train_rows = int(min(len(tr) for tr, _ in splits))
    scorers = _original_unit_scorers(target_transform)
    inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
    common = dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test_orig=inv(y_test),
                  X_val=X_val, y_val_orig=inv(np.asarray(y_val, float)) if y_val is not None else None,
                  target_transform=target_transform, folds=folds,
                  search_cv=_search_cv(folds["strategy"], cv_folds, folds["groups"], random_state, len(y_train)),
                  scorers=scorers, nested_cv=nested_cv, random_state=random_state,
                  selection_metric=selection_metric, preprocessor=preprocessor, n_jobs=n_jobs,
                  min_train_rows=min_train_rows)

    # reference error of "predict the mean/median" on the SAME folds, original units
    ref = OriginalScaleMean(strategy="median" if selection_metric == "mae" else "mean",
                            target_transform=target_transform)
    rtr, rte, _ = _run_folds(ref, X_train, y_train, splits, 1)
    ref_cv = _cv_from_fold_preds(y_train, splits, rtr, rte, target_transform)
    baseline_err = float(np.nanmean(ref_cv[M]))

    probe_err = None
    if capacity_probe:
        if progress_callback:
            progress_callback(0, len(models), "reference model (capacity probe)")
        ptr, pte, perr = _run_folds(_wrap("probe", _capacity_probe(time_ordered, random_state),
                                          preprocessor), X_train, y_train, splits, n_jobs)
        pcv = _cv_from_fold_preds(y_train, splits, ptr, pte, target_transform)
        if pcv["failed_folds"] == 0:
            probe_err = float(np.nanmean(pcv[M]))
            diagnostics["probe_cv_error"] = probe_err
    diagnostics.update({"baseline_cv_error": baseline_err, "folds": folds,
                        "selection_metric": selection_metric, "min_train_fold_rows": min_train_rows})

    results: dict[str, ModelResult] = {}
    n_models = len(models)
    for idx, (name, est) in enumerate(models.items(), start=1):
        if progress_callback:
            progress_callback(idx, n_models, name + (" (tuning)" if tuning != "off"
                                                     and name != BASELINE_MODEL_NAME else ""))
        try:
            results[name] = _fit_and_score(name, est, tuning=tuning, n_iter=trials, **common)
        except Exception as e:
            failures[name] = f"{type(e).__name__}: {e}"

    def _diagnose_all(res_names):
        errs = [r.metrics[sel_cv_key] for n, r in results.items()
                if n != BASELINE_MODEL_NAME and _cv_complete(r) and sel_cv_key in r.metrics]
        best = min(errs + ([probe_err] if probe_err is not None else []), default=None)
        for n in res_names:
            r = results[n]
            if n == BASELINE_MODEL_NAME:
                r.fit_status, r.fit_reasons = "reference", ["the reference every model must beat"]
                continue
            if not _cv_complete(r):
                msg = (r.cv_scores or {}).get("errors") or ["unknown error"]
                r.fit_status = "cv_failed"
                r.fit_reasons = [f"cross-validation failed on {r.metrics.get('CV_failed_folds', '?')} "
                                 f"fold(s) ({msg[0][:120]}); excluded from selection"]
                continue
            r.fit_status, r.fit_reasons, skill = diagnose_fit(r.metrics, baseline_err,
                                                              selection_metric, best_cv_error=best)
            if skill is not None:
                r.metrics["Skill_vs_baseline_pct"] = skill * 100
        return best

    _diagnose_all(list(results))

    # ---- automatic remediation, accepted only on a real paired improvement ----
    if auto_remediate:
        for name in list(results):
            r = results[name]
            if name == BASELINE_MODEL_NAME or r.fit_status not in ("overfit", "underfit"):
                continue
            candidates = REMEDIES.get(name, {}).get(r.fit_status, [])
            if not candidates:
                r.remediation = (f"{r.fit_status} detected; no automatic fix defined for this model type"
                                 + ("" if r.fit_status == "overfit" else " \u2014 try interaction features"))
                continue
            if progress_callback:
                progress_callback(n_models, n_models, f"{name} \u2014 fixing {r.fit_status}")
            best_r, best_p = r, None
            for params in candidates:
                try:
                    cand = _fit_and_score(name, _with_params(r.spec_estimator, params,
                                                             max_neighbors=min_train_rows - 1),
                                          tuning="off", n_iter=0, **common)
                except Exception:
                    continue
                if not _cv_complete(cand):
                    continue
                gain, se, _ = corrected_paired_difference(_fold_errors(best_r, M), _fold_errors(cand, M),
                                                          folds["ratio"])
                if gain > max(REMEDY_MIN_IMPROVEMENT * best_r.metrics[sel_cv_key], se):
                    best_r, best_p = cand, params
            old = r.metrics.get(sel_cv_key, math.inf)
            if best_p is not None:
                best_r.best_params = {**(r.best_params or {}), **best_p}
                best_r.cv_optimistic = r.cv_optimistic
                best_r.train_time_sec += r.train_time_sec
                results[name] = best_r
                _diagnose_all([name])
                best_r.remediation = (f"{r.fit_status} detected \u2192 retrained with {best_p}; "
                                      f"CV {M} {old:.4g} \u2192 {best_r.metrics[sel_cv_key]:.4g} "
                                      f"(beyond the corrected fold-to-fold noise); now: {best_r.fit_status}")
            else:
                r.remediation = (f"{r.fit_status} detected; tried {len(candidates)} alternative "
                                 f"setting(s), none cut CV {M} by more than fold-to-fold noise "
                                 "\u2014 kept the original")
                if r.fit_status == "overfit":
                    r.fit_status = "benign"
                    r.fit_reasons.append("stronger regularisation did not lower held-out error, "
                                         "so this gap is not costing accuracy")

    # ---- ensemble of the best model from each of three families ----
    if build_ensemble:
        key = f"{M}_val" if X_val is not None else sel_cv_key
        members = _pick_ensemble_members(results, key)
        if len(members) >= 3:
            if progress_callback:
                progress_callback(n_models, n_models, "Ensemble of " + ", ".join(members))
            try:
                ens = _ensemble_result(members, results, y_train=y_train, y_test_orig=inv(y_test),
                                       y_val_orig=common["y_val_orig"], folds=folds,
                                       target_transform=target_transform)
                results[ENSEMBLE_NAME] = ens
                _diagnose_all([ENSEMBLE_NAME])
            except Exception as e:
                failures[ENSEMBLE_NAME] = f"{type(e).__name__}: {e}"
    return results


# ---- ranking & selection ----------------------------------------------------------------------

def rank_models(results: dict[str, ModelResult]) -> list[tuple[str, float, float]]:
    """All models by TEST RMSE - for display only, never for selection."""
    rows = [(n, r.metrics["RMSE"], r.metrics["R2"]) for n, r in results.items()]
    rows.sort(key=lambda row: (row[1], -row[2]))
    return rows


def selection_basis(results: dict[str, ModelResult]) -> str:
    """'validation' if every model has a validation score; 'cv' if ANY candidate
    has complete CV (candidates without it are excluded, not a reason to fall
    back); 'test' only when no candidate could be cross-validated at all."""
    if not results:
        return "test"
    vals = [r.metrics.get("RMSE_val") for r in results.values()]
    if all(v is not None and np.isfinite(v) for v in vals):
        return "validation"
    cands = [r for n, r in results.items() if n != BASELINE_MODEL_NAME] or list(results.values())
    if any(_cv_complete(r) for r in cands):
        return "cv"
    return "test"


def explain_selection(results: dict[str, ModelResult], *, one_se_rule: bool = True,
                      selection_metric: str = "rmse", no_signal_check: bool = True) -> tuple[str, str]:
    """Return (winner, plain-English note on how it was chosen)."""
    if not results:
        raise ValueError("No models were trained.")
    basis = selection_basis(results)
    M = selection_metric.upper()
    key, tie = {"validation": (f"{M}_val", "R2_val"), "cv": (f"CV_{M}_mean", "CV_R2_mean"),
                "test": (M, "R2")}[basis]
    if basis == "cv":
        cands = {n: r for n, r in results.items() if n != BASELINE_MODEL_NAME and _cv_complete(r)}
    else:
        cands = {n: r for n, r in results.items() if n != BASELINE_MODEL_NAME}
    if not cands:
        if BASELINE_MODEL_NAME in results:
            return BASELINE_MODEL_NAME, "no other model could be evaluated, so the baseline is reported"
        cands = dict(results)
    ordered = sorted(cands.items(), key=lambda kv: (kv[1].metrics.get(key, math.inf),
                                                    -kv[1].metrics.get(tie, -math.inf)))
    best_name, best = ordered[0]
    label = {"validation": f"validation-set {M}", "cv": f"cross-validated {M}",
             "test": f"test {M} (no model could be cross-validated - treat the test score as optimistic)"}[basis]
    note, winner = f"lowest {label}", best_name
    ratio = best.metrics.get("CV_test_train_ratio")

    if one_se_rule and basis == "cv":
        be = _fold_errors(best, M)
        within = []
        for n, r in ordered:
            e = _fold_errors(r, M)
            if be is not None and e is not None and ratio is not None and len(e) == len(be):
                md, se, _ = corrected_paired_difference(e, be, ratio)
            else:  # legacy / summary-only results
                md = r.metrics.get(key, math.inf) - best.metrics[key]
                se = best.metrics.get(f"CV_{M}_std", 0.0) / math.sqrt(max(best.metrics.get("CV_n_splits", 1), 1))
            if md <= se:
                within.append((n, r))
        simplest = min(within, key=lambda kv: (COMPLEXITY.get(kv[0], 5), kv[1].metrics.get(key, math.inf)))
        if simplest[0] != best_name:
            winner = simplest[0]
            note = (f"one-standard-error rule: {winner} is within one corrected standard error of "
                    f"the best cross-validated {M} ({best_name}: {best.metrics[key]:.4g}) and is a "
                    "simpler model, so it is preferred")

    base = results.get(BASELINE_MODEL_NAME)
    if no_signal_check and base is not None and winner != BASELINE_MODEL_NAME:
        if basis == "cv" and _cv_complete(base):
            eb, ew = _fold_errors(base, M), _fold_errors(best, M)
            base_err = base.metrics.get(key, math.inf)
            skill = 1.0 - best.metrics.get(key, math.inf) / base_err if base_err > 0 else 0.0
            significant = False
            if eb is not None and ew is not None and ratio is not None and len(eb) == len(ew):
                gain, se, k = corrected_paired_difference(eb, ew, ratio)
                tcrit = float(student_t.ppf(1 - NO_SIGNAL_ALPHA, df=max(k - 1, 1)))
                significant = bool(se > 0 and np.isfinite(se) and gain > tcrit * se)
            if skill <= NO_SIGNAL_MIN_SKILL or (skill < UNDERFIT_MIN_SKILL and not significant):
                return BASELINE_MODEL_NAME, (
                    f"no reliable signal: the best model ({best_name}) is only {skill:.1%} better than "
                    f"predicting the mean in cross-validation"
                    + ("" if skill <= NO_SIGNAL_MIN_SKILL else ", and that gain is within fold-to-fold noise")
                    + ". The baseline is reported as the honest answer; more rows or more informative "
                    "features are needed")
        elif basis == "validation":
            if best.metrics.get(key, math.inf) >= base.metrics.get(key, math.inf):
                return BASELINE_MODEL_NAME, ("no reliable signal: no model beats predicting the mean "
                                             "on the validation file")
    return winner, note


def pick_best(results: dict[str, ModelResult], *, one_se_rule: bool = True,
              selection_metric: str = "rmse", no_signal_check: bool = True) -> str:
    return explain_selection(results, one_se_rule=one_se_rule, selection_metric=selection_metric,
                             no_signal_check=no_signal_check)[0]


def target_outlier_share(y: np.ndarray) -> float:
    """Share of target values beyond 3 IQRs from the quartiles (extreme outliers)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 20:
        return 0.0
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
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


# ---- log-target decision, verified by cross-validation ------------------------------------------

def compare_target_transforms(X_raw, y_orig: np.ndarray, preprocessor, folds: dict, *,
                              selection_metric: str = "rmse", count_like: bool = False,
                              random_state: int = 42, time_ordered: bool = False,
                              n_jobs: int = -1) -> tuple[str, dict[str, float]]:
    """Pick raw vs log1p target by CV error in ORIGINAL units, using a linear
    and a boosting probe on the shared folds. Replaces a fragile skew/integer
    heuristic (integer prices used to be mistaken for counts)."""
    M = selection_metric.upper()
    y_orig = np.asarray(y_orig, float)
    errors: dict[str, float] = {}
    for mode in ("none", "log1p"):
        y = np.log1p(y_orig) if mode == "log1p" else y_orig
        probes = {"Ridge": _scaled(Ridge(alpha=1.0)), "HGB": _capacity_probe(time_ordered, random_state)}
        if mode == "none" and count_like:
            probes["HGB-poisson"] = HistGradientBoostingRegressor(loss="poisson", max_iter=400,
                                                                  learning_rate=0.1,
                                                                  random_state=random_state)
        best = math.inf
        for pname, est in probes.items():
            full = _wrap("Ridge" if pname == "Ridge" else pname, est, preprocessor)
            tr, te, _ = _run_folds(full, X_raw, y, folds["splits"], n_jobs)
            cv = _cv_from_fold_preds(y, folds["splits"], tr, te, mode)
            if cv["failed_folds"] == 0:
                e = float(np.nanmean(cv[M]))
                errors[f"{mode}:{pname}"] = e
                best = min(best, e)
        errors[mode] = best
    chosen = "log1p" if errors["log1p"] < errors["none"] else "none"
    return chosen, errors


# ---- winner-only extras --------------------------------------------------------------------------

def compute_permutation_importance(estimator, X, y, *, n_repeats: int = 10, random_state: int = 42,
                                   scoring: str | None = None, target_transform: str = "none",
                                   selection_metric: str = "rmse"):
    """Held-out permutation importance, measured in ORIGINAL target units.

    With a model pipeline and a raw frame, each ORIGINAL column is permuted as
    a whole (a one-hot-encoded category counts once, not once per dummy).
    """
    scorer = scoring or _original_unit_scorers(target_transform)[selection_metric]
    try:
        with _quiet_fit():
            result = permutation_importance(estimator, X, y, n_repeats=n_repeats,
                                            random_state=random_state, scoring=scorer, n_jobs=1)
        return np.asarray(result.importances_mean), np.asarray(result.importances_std)
    except Exception:
        return None, None


def compute_learning_curve(estimator, X, y, *, cv_strategy="kfold", cv_folds=5, groups=None,
                           target_transform="none", random_state=42, n_points=6,
                           folds: dict | None = None) -> dict | None:
    """Train vs held-out R2 as the training set grows, plus a plain-English reading."""
    if folds is not None:
        per_rep = max(1, len(folds["splits"]) // max(folds.get("n_repeats", 1), 1))
        cv, g = folds["splits"][:per_rep], None
    else:
        cv, g = _make_cv(cv_strategy, cv_folds, groups, random_state)
    scorer = _original_unit_scorers(target_transform)["r2"]
    try:
        with _quiet_fit():
            sizes, tr, va = learning_curve(_set_n_jobs(clone(estimator), 1), X, y, cv=cv, groups=g,
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


def _oof_from_folds(n: int, splits, fold_test_pred) -> tuple[np.ndarray, np.ndarray]:
    s, c = np.zeros(n), np.zeros(n)
    for (tr, te), p in zip(splits, fold_test_pred):
        if p is None:
            continue
        s[te] += p
        c[te] += 1
    mask = c > 0
    oof = np.full(n, np.nan)
    oof[mask] = s[mask] / c[mask]
    return oof, mask


def smearing_factor(y_model: np.ndarray, fold_test_pred, folds: dict) -> float | None:
    """Duan's smearing estimate E[exp(residual)] from OUT-OF-FOLD log residuals.

    expm1(prediction on the log scale) estimates the conditional MEDIAN, which
    sits below the mean for skewed targets, so an RMSE-selected model is
    biased low. Multiplying exp(pred) by this factor corrects the bias.
    """
    if fold_test_pred is None or folds is None:
        return None
    oof, mask = _oof_from_folds(len(y_model), folds["splits"], fold_test_pred)
    if mask.sum() < 20:
        return None
    r = np.asarray(y_model, float)[mask] - oof[mask]
    s = float(np.mean(np.exp(np.clip(r, -20, 20))))
    return s if np.isfinite(s) and 0.5 < s < 5.0 else None


def apply_smearing(pred_model: np.ndarray, factor: float | None) -> np.ndarray:
    """Model-space (log1p) prediction -> bias-corrected original units."""
    if not factor:
        return np.expm1(pred_model)
    return np.exp(pred_model) * factor - 1.0


def _interval_halfwidth(pi: dict, X, n: int) -> np.ndarray:
    if pi.get("sigma_model") is not None:
        s = np.maximum(np.asarray(pi["sigma_model"].predict(X), float), pi.get("sigma_floor") or 0.0)
        return pi["q"] * s
    return np.full(n, float(pi.get("halfwidth", 0.0)))


def compute_prediction_interval(estimator, X, y, *, cv_strategy="kfold", cv_folds=5, groups=None,
                                target_transform="none", random_state=42, coverage=0.90,
                                X_test=None, y_test=None, fold_test_pred=None,
                                folds: dict | None = None, adaptive: bool = True,
                                sigma_preprocessor=None) -> dict | None:
    """Conformal prediction interval from OUT-OF-FOLD residuals.

    Reuses the winner's stored fold predictions when available (no refits).
    With `adaptive=True` (and >= 100 residuals) the interval is locally
    adaptive: a small model predicts each row's typical absolute residual and
    the conformal quantile is taken of residual / predicted scale, so easy
    rows get narrow intervals and hard rows wide ones. The scale model is
    cross-fitted on the calibration rows so the calibration stays honest.
    Residuals live in the model's space (log space under log1p, so intervals
    are multiplicative in original units).
    """
    y = np.asarray(y, float)
    n_all = len(y)
    if fold_test_pred is not None and folds is not None:
        oof, mask = _oof_from_folds(n_all, folds["splits"], fold_test_pred)
    else:
        cv, g = _make_cv(cv_strategy, cv_folds, groups, random_state, n_samples=None)
        splits = list(cv.split(np.zeros((n_all, 1)), y, g))
        _, fte, errs = _run_folds(estimator, X, y, splits, -1)
        if any(p is None for p in fte):
            return None
        oof, mask = _oof_from_folds(n_all, splits, fte)
    idx = np.where(mask)[0]
    if len(idx) < 10:
        return None
    abs_r = np.abs(y[idx] - oof[idx])
    n = len(abs_r)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    out: dict[str, Any] = {"coverage_target": coverage,
                           "space": "log1p" if target_transform == "log1p" else "original",
                           "method": "constant", "sigma_model": None, "sigma_floor": None}
    if adaptive and n >= 100:
        try:
            sig = _wrap("sigma", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05,
                                                               max_leaf_nodes=15, min_samples_leaf=20,
                                                               random_state=random_state),
                        sigma_preprocessor)
            Xm = _take(X, idx)
            s_cal = np.zeros(n)
            for tr, te in KFold(5, shuffle=True, random_state=random_state).split(abs_r):
                with _quiet_fit():
                    m = clone(sig).fit(_take(Xm, tr), abs_r[tr])
                s_cal[te] = m.predict(_take(Xm, te))
            floor = max(0.1 * float(np.median(abs_r)), 1e-12)
            s_cal = np.maximum(s_cal, floor)
            q = float(np.quantile(abs_r / s_cal, level))
            with _quiet_fit():
                sigma_model = clone(sig).fit(Xm, abs_r)
            out.update(method="adaptive", q=q, sigma_model=sigma_model, sigma_floor=floor,
                       halfwidth=float(q * np.median(s_cal)))
        except Exception:
            out["method"] = "constant"
    if out["method"] == "constant":
        q = float(np.quantile(abs_r, level))
        out.update(q=q, halfwidth=q)
    if X_test is not None and y_test is not None:
        with _quiet_fit():
            p = np.asarray(estimator.predict(X_test), float)
        hw = _interval_halfwidth(out, X_test, len(p))
        lo, hi = p - hw, p + hw
        y_test = np.asarray(y_test, float)
        out["test_coverage"] = float(np.mean((y_test >= lo) & (y_test <= hi)))
        inv = np.expm1 if target_transform == "log1p" else (lambda a: a)
        out["mean_width_original_units"] = float(np.mean(inv(hi) - inv(lo)))
        if out["method"] == "adaptive":
            qc = float(np.quantile(abs_r, level))
            out["constant_interval_test_coverage"] = float(np.mean(np.abs(y_test - p) <= qc))
    return out


def predict_with_interval(bundle: dict, X_raw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For users of best_model.joblib: (prediction, lower, upper) in original units."""
    X = bundle["preprocessor"].transform(X_raw)
    p = np.asarray(bundle["model"].predict(X), float)
    pi = bundle.get("prediction_interval") or {}
    hw = _interval_halfwidth(pi, X, len(p)) if pi else np.zeros(len(p))
    if bundle.get("target_transform") == "log1p":
        return apply_smearing(p, bundle.get("smearing_factor")), np.expm1(p - hw), np.expm1(p + hw)
    return p, p - hw, p + hw
