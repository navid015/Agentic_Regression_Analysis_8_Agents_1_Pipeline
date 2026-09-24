"""Tests for the over/underfitting safeguards added in v2.

Each test builds a small synthetic dataset where the right answer is known,
then checks that the pipeline prevents, detects or fixes the problem.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from utils.code_generator import generate_notebook, generate_python_script
from utils.modeling import (
    BASELINE_MODEL_NAME,
    ENSEMBLE_NAME,
    compute_learning_curve,
    compute_prediction_interval,
    diagnose_fit,
    explain_selection,
    filter_zoo_for_data,
    get_default_model_zoo,
    resolve_selection_metric,
    target_outlier_share,
    train_and_evaluate,
)
from utils.preprocessing import preprocess


def _zoo(*names):
    return {k: v for k, v in get_default_model_zoo().items() if k in names}


def _fit(df, target, names, **kw):
    pre_kw = {k: kw.pop(k) for k in list(kw) if k in ("log_transform_target", "add_interactions",
                                                      "split_strategy", "time_column")}
    pre = preprocess(df, target, **pre_kw)
    res = train_and_evaluate(pre.X_train, pre.X_test, pre.y_train, pre.y_test,
                             models=_zoo(*names), cv_folds=3, **kw)
    return pre, res


# ---- prevention ---------------------------------------------------------------

def test_target_scaling_rescues_svr_on_large_targets():
    """SVR with C=1 on a target in the hundreds of thousands used to score R2 ~ 0."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    df["y"] = 300_000 + 80_000 * df.a + 30_000 * df.b + rng.normal(0, 10_000, 400)
    _, res = _fit(df, "y", [BASELINE_MODEL_NAME, "SVR"], build_ensemble=False)
    assert res["SVR"].metrics["R2"] > 0.8


def test_auto_log_applies_only_to_skewed_targets():
    rng = np.random.default_rng(1)
    x = rng.normal(size=600)
    skewed = pd.DataFrame({"x": x, "y": np.exp(2 + x + rng.normal(0, .3, 600))})
    normal = pd.DataFrame({"x": x, "y": 5 + x + rng.normal(0, .3, 600)})
    assert preprocess(skewed, "y", log_transform_target="auto").target_transform == "log1p"
    assert preprocess(normal, "y", log_transform_target="auto").target_transform == "none"


def test_auto_log_leaves_count_targets_for_poisson():
    rng = np.random.default_rng(2)
    x = rng.normal(size=800)
    df = pd.DataFrame({"x": x, "y": rng.poisson(np.exp(0.3 + 0.9 * x))})
    pre = preprocess(df, "y", log_transform_target="auto")
    assert pre.target_transform == "none" and pre.summary["target_count_like"]
    zoo, skipped = filter_zoo_for_data(get_default_model_zoo(), pre.y_train)
    assert "PoissonRegressor" in zoo


def test_poisson_is_skipped_for_negative_targets():
    zoo, skipped = filter_zoo_for_data(get_default_model_zoo(), np.array([-1.0, 2.0, 3.0]))
    assert "PoissonRegressor" not in zoo and "PoissonRegressor" in skipped


def test_interaction_features_fix_an_underfitting_linear_model():
    rng = np.random.default_rng(3)
    a, b = rng.uniform(-2, 2, 700), rng.uniform(-2, 2, 700)
    df = pd.DataFrame({"a": a, "b": b, "y": a * b + rng.normal(0, .1, 700)})
    _, plain = _fit(df, "y", [BASELINE_MODEL_NAME, "LinearRegression"], build_ensemble=False)
    _, inter = _fit(df, "y", [BASELINE_MODEL_NAME, "LinearRegression"], build_ensemble=False,
                    add_interactions=True)
    assert plain["LinearRegression"].fit_status == "underfit"
    assert inter["LinearRegression"].metrics["R2"] > 0.9


def test_time_split_keeps_a_trend_feature():
    rng = np.random.default_rng(4)
    n = 400
    df = pd.DataFrame({"day": pd.date_range("2021-01-01", periods=n).astype(str),
                       "x": rng.normal(size=n)})
    df["y"] = np.arange(n) * 0.5 + df.x + rng.normal(0, .5, n)
    pre = preprocess(df, "y", split_strategy="time", time_column="day")
    assert "day_elapsed_days" in pre.summary["time_features_from_split_column"]
    res = train_and_evaluate(pre.X_train, pre.X_test, pre.y_train, pre.y_test,
                             models=_zoo(BASELINE_MODEL_NAME, "Ridge"), cv_strategy="time",
                             cv_folds=3, build_ensemble=False)
    assert res["Ridge"].metrics["R2"] > 0.8   # extrapolates the trend into the future


# ---- detection ----------------------------------------------------------------

def test_diagnosis_flags_overfitting():
    m = {"CV_R2_mean": 0.55, "CV_R2_train_mean": 0.98, "CV_RMSE_mean": 1.0, "CV_R2_std": 0.02}
    assert diagnose_fit(m, baseline_cv_rmse=2.0)[0] == "overfit"


def test_diagnosis_flags_underfitting_vs_baseline():
    m = {"CV_R2_mean": 0.01, "CV_R2_train_mean": 0.02, "CV_RMSE_mean": 1.98, "CV_R2_std": 0.01}
    status, reasons, skill = diagnose_fit(m, baseline_cv_rmse=2.0)
    assert status == "underfit" and skill < 0.05


def test_diagnosis_calls_a_balanced_model_good():
    m = {"CV_R2_mean": 0.90, "CV_R2_train_mean": 0.93, "CV_RMSE_mean": 0.6, "CV_R2_std": 0.02}
    assert diagnose_fit(m, baseline_cv_rmse=2.0)[0] == "good"


def test_diagnosis_uses_training_cv_not_the_test_set():
    """Changing test metrics must not change the diagnosis (test stays untouched)."""
    m = {"CV_R2_mean": 0.9, "CV_R2_train_mean": 0.92, "CV_RMSE_mean": 0.6, "CV_R2_std": 0.02,
         "R2": -5.0, "RMSE": 99.0}
    assert diagnose_fit(m, baseline_cv_rmse=2.0)[0] == "good"


def test_outlier_heavy_targets_switch_selection_to_mae():
    rng = np.random.default_rng(5)
    y = rng.normal(size=1000); y[:50] += 60
    assert target_outlier_share(y) > 0.02
    assert resolve_selection_metric("auto", y)[0] == "mae"
    assert resolve_selection_metric("auto", rng.normal(size=1000))[0] == "rmse"


# ---- fixing -------------------------------------------------------------------

def test_overfitting_tree_is_remediated():
    rng = np.random.default_rng(6)
    n = 300
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    df["y"] = df.a + rng.normal(0, 1.0, n)            # mostly noise -> trees overfit
    zoo = _zoo(BASELINE_MODEL_NAME, "DecisionTree")
    zoo["DecisionTree"].set_params(max_depth=None, min_samples_leaf=1)
    pre = preprocess(df, "y")
    res = train_and_evaluate(pre.X_train, pre.X_test, pre.y_train, pre.y_test, models=zoo,
                             cv_folds=3, build_ensemble=False)
    tree = res["DecisionTree"]
    assert tree.remediation and "overfit" in tree.remediation
    assert tree.best_params and tree.best_params.get("min_samples_leaf", 1) >= 10


def test_tuning_returns_settings_and_marks_cv_as_optimistic():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"a": rng.normal(size=300)})
    df["y"] = 2 * df.a + rng.normal(0, .3, 300)
    _, res = _fit(df, "y", [BASELINE_MODEL_NAME, "Ridge"], tuning="fast", n_iter=4,
                  build_ensemble=False)
    assert res["Ridge"].best_params and "alpha" in res["Ridge"].best_params
    assert res["Ridge"].cv_optimistic


def test_nested_cv_gives_non_optimistic_scores():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"a": rng.normal(size=300)})
    df["y"] = 2 * df.a + rng.normal(0, .3, 300)
    _, res = _fit(df, "y", [BASELINE_MODEL_NAME, "Ridge"], tuning="fast", n_iter=3,
                  nested_cv=True, build_ensemble=False)
    assert not res["Ridge"].cv_optimistic


def test_ensemble_is_built_from_the_top_three():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    df["y"] = df.a - df.b + rng.normal(0, .3, 300)
    _, res = _fit(df, "y", [BASELINE_MODEL_NAME, "LinearRegression", "Ridge", "KNN",
                            "DecisionTree"])
    assert ENSEMBLE_NAME in res and len(res[ENSEMBLE_NAME].ensemble_members) == 3
    assert BASELINE_MODEL_NAME not in res[ENSEMBLE_NAME].ensemble_members


def test_one_standard_error_rule_prefers_the_simpler_model():
    class R:  # minimal stand-in for ModelResult
        def __init__(self, rmse, std):
            self.metrics = {"CV_RMSE_mean": rmse, "CV_RMSE_std": std, "CV_n_splits": 5,
                            "CV_R2_mean": 0.9, "RMSE": rmse, "R2": 0.9}
    results = {"XGBoost": R(1.00, 0.10), "Ridge": R(1.03, 0.10), BASELINE_MODEL_NAME: R(3, .1)}
    assert explain_selection(results, one_se_rule=True)[0] == "Ridge"
    assert explain_selection(results, one_se_rule=False)[0] == "XGBoost"


def test_a_broken_model_does_not_sink_the_run():
    class Broken:
        def get_params(self, deep=True): return {}
        def set_params(self, **p): return self
        def fit(self, X, y): raise RuntimeError("boom")
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"a": rng.normal(size=100)}); df["y"] = df.a
    pre = preprocess(df, "y")
    fails = {}
    res = train_and_evaluate(pre.X_train, pre.X_test, pre.y_train, pre.y_test,
                             models={"LinearRegression": get_default_model_zoo()["LinearRegression"],
                                     "Broken": Broken()}, cv_folds=3, failures=fails)
    assert "LinearRegression" in res and "Broken" in fails


# ---- winner extras ------------------------------------------------------------

def test_prediction_interval_reaches_its_target_coverage():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"a": rng.normal(size=1500)})
    df["y"] = 3 * df.a + rng.normal(0, 1, 1500)
    pre = preprocess(df, "y")
    est = get_default_model_zoo()["LinearRegression"].fit(pre.X_train, pre.y_train)
    pi = compute_prediction_interval(est, pre.X_train, pre.y_train, coverage=0.9,
                                     X_test=pre.X_test, y_test=pre.y_test)
    assert 0.85 <= pi["test_coverage"] <= 0.95


def test_learning_curve_reads_underfitting():
    rng = np.random.default_rng(12)
    a = rng.uniform(-3, 3, 600)
    df = pd.DataFrame({"a": a, "y": np.sin(3 * a) + rng.normal(0, .05, 600)})
    pre = preprocess(df, "y")
    lc = compute_learning_curve(get_default_model_zoo()["LinearRegression"],
                                pre.X_train, pre.y_train)
    assert "underfitting" in lc["reading"]


# ---- generated code -----------------------------------------------------------

def test_generated_notebook_code_is_valid_python_and_keeps_every_line():
    nb = json.loads(generate_notebook(target="y", model_names=["LinearRegression"]))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    compile(code, "notebook", "exec")
    assert "TRAIN_CSV    =" in code and "df_train = pd.read_csv(TRAIN_CSV)" in code
    headings = [c["source"][0] for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert any("SPLIT" in h for h in headings)


def test_generated_script_rebuilds_the_exact_winner():
    script = generate_python_script(
        target="y", model_names=["LinearRegression", "SVR"],
        winner_spec={"name": "SVR", "params": {"C": 12.5, "epsilon": 0.02}})
    compile(script, "script", "exec")
    assert "WINNER = scaled(SVR(" in script and "'C': 12.5" in script


def test_saved_preprocessor_accepts_raw_rows_with_dates():
    """best_model.joblib must clean raw CSV rows, including date and time-split columns."""
    import io

    import joblib
    rng = np.random.default_rng(13)
    n = 300
    df = pd.DataFrame({"when": pd.date_range("2020-01-01", periods=n).astype(str),
                       "signup": pd.date_range("2019-06-01", periods=n, freq="3D").astype(str),
                       "x": rng.normal(size=n)})
    df["y"] = np.arange(n) * 0.2 + df.x
    pre = preprocess(df, "y", split_strategy="time", time_column="when")
    buf = io.BytesIO(); joblib.dump(pre.raw_preprocessor, buf); buf.seek(0)
    out = joblib.load(buf).transform(df.drop(columns=["y"]).tail(4))
    assert out.shape == (4, pre.X_train.shape[1]) and np.isfinite(out).all()
