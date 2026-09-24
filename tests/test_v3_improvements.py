"""Regression tests for the v3 fixes.

Each test reproduces a failure found in the review of v2 and checks it stays
fixed: deleted real features, lost categorical signal, silent test-set
selection, time/group CV mix-ups, selection optimism on noise, biased log
predictions, blind leakage audit, non-reproducible exported code, shared
state between users.
"""
from __future__ import annotations

import io
import os
import runpy
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from crew import advisor, tools
from crew.orchestrator import reconcile_strategies, run_full_pipeline
from utils.diagnostics import leakage_suspects, residual_diagnostics
from utils.io import read_table
from utils.modeling import (
    BASELINE_MODEL_NAME,
    ENSEMBLE_NAME,
    EarlyStoppingRegressor,
    apply_smearing,
    compute_prediction_interval,
    explain_selection,
    get_default_model_zoo,
    rebuild_estimator,
    selection_basis,
    smearing_factor,
    train_and_evaluate,
)
from utils.preprocessing import _looks_like_identifier, preprocess

FAST = dict(tuning="off", compute_learning_curves=False, compute_intervals=False,
            compute_permutation_importances=False)


def _zoo(*names, **kw):
    return {k: v for k, v in get_default_model_zoo(**kw).items() if k in names}


def _fit_pipeline(pre, names, **kw):
    """Leak-free path: raw frames + preprocessing fitted inside each fold."""
    return train_and_evaluate(pre.X_train_raw, pre.X_test_raw, pre.y_train, pre.y_test,
                              models=_zoo(*names), preprocessor=pre.preprocessor_templates,
                              groups_train=pre.groups_train, target_transform=pre.target_transform,
                              **kw)


# ---- identifiers and column types -----------------------------------------------

def test_sorted_real_integer_features_are_kept():
    rng = np.random.default_rng(0)
    years = pd.DataFrame({"year": np.arange(1950, 2024), "y": rng.normal(size=74)})
    sqft = pd.DataFrame({"sqft": np.sort(rng.choice(np.arange(500, 6000), 400, replace=False)),
                         "y": rng.normal(size=400)})
    for df, col in ((years, "year"), (sqft, "sqft")):
        pre = preprocess(df, "y")
        assert col in pre.summary["numeric_cols"], f"{col} was deleted as an 'ID'"
        assert col in pre.summary["id_suspect_columns_kept"]      # reported, not dropped


def test_row_counters_are_still_dropped_even_when_shuffled():
    rng = np.random.default_rng(1)
    assert _looks_like_identifier("x", pd.Series(rng.permutation(300)))
    assert _looks_like_identifier("x", pd.Series(np.arange(1, 301)))
    assert _looks_like_identifier("customerId", pd.Series(rng.permutation(300) + 5000))  # camelCase


def test_formatted_numbers_are_parsed_and_codes_stay_categories():
    df = pd.DataFrame({
        "income": ["$1,200", "$3,400", "(50)", "12%", None] * 40,
        "zip": ["02139", "10001", "94105", "60601", "02139"] * 40,     # leading zeros -> code
        "MSSubClass": [20, 60, 20, 50, 70] * 40,                       # integer code
        "y": np.arange(200.0),
    })
    pre = preprocess(df, "y")
    s = pre.summary
    assert s["numeric_string_columns_parsed"] == ["income"]
    assert pre.X_train_raw["income"].dtype.kind == "f"
    assert "MSSubClass" in s["categorical_overrides"]
    assert "zip" not in s["numeric_cols"] and "MSSubClass" not in s["numeric_cols"]


def test_target_encoding_keeps_high_cardinality_signal():
    rng = np.random.default_rng(2)
    n, k = 1500, 60
    levels = np.array([f"Z{i:03d}" for i in range(k)])
    z = rng.choice(levels, n)
    effect = dict(zip(levels, rng.normal(0, 1, k)))
    df = pd.DataFrame({"zipcode": z, "y": 2 * np.vectorize(effect.get)(z) + rng.normal(0, .5, n)})
    r2 = {}
    for enc in ("target", "frequency"):
        pre = preprocess(df, "y", high_cardinality_encoding=enc)
        res = _fit_pipeline(pre, ["Ridge"], cv_folds=3, build_ensemble=False, capacity_probe=False)
        r2[enc] = res["Ridge"].metrics["R2"]
    assert r2["target"] > 0.85 and r2["target"] > r2["frequency"] + 0.4


# ---- splits and CV consistency ----------------------------------------------------

def test_time_split_with_group_column_uses_time_series_cv():
    warns = []
    assert reconcile_strategies("time", "group", "date", "store", warns) == ("time", "time")
    assert warns and "forward in time" in warns[0]
    assert reconcile_strategies("random", "time", None, None, []) == ("random", "kfold")
    assert reconcile_strategies("random", "group", None, None, []) == ("random", "kfold")


def test_user_supplied_time_files_are_sorted_and_checked():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2021-01-01", periods=300).astype(str)
    df = pd.DataFrame({"date": dates, "x": rng.normal(size=300)})
    df["y"] = np.arange(300) * 0.1 + df.x
    train = df.iloc[:240].sample(frac=1, random_state=0)       # shuffled on purpose
    pre = preprocess(train, "y", df_test=df.iloc[240:], split_strategy="time", time_column="date")
    assert pre.X_train_raw["date_elapsed_days"].is_monotonic_increasing
    assert pre.summary["time_order_ok"] is True


def test_rows_shared_between_train_and_test_files_are_reported():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=200)})
    df["y"] = df.x * 2
    pre = preprocess(df, "y", df_test=pd.concat([df.iloc[:15], df.iloc[:5] + 100]))
    assert pre.summary["train_test_duplicate_rows"] == 15


# ---- selection honesty -----------------------------------------------------------------

class _FailsOnSmallData(BaseEstimator, RegressorMixin):
    """Fits on the full training set but raises inside every CV fold."""
    def __init__(self, min_rows=10_000):
        self.min_rows = min_rows

    def fit(self, X, y):
        if len(y) < self.min_rows:
            raise RuntimeError("too few rows")
        self.m_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


def test_a_cv_failure_never_switches_selection_to_the_test_set():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"a": rng.normal(size=400)})
    df["y"] = 2 * df.a + rng.normal(0, .3, 400)
    pre = preprocess(df, "y")
    models = _zoo(BASELINE_MODEL_NAME, "Ridge")
    models["Fragile"] = _FailsOnSmallData(min_rows=len(pre.y_train))
    res = train_and_evaluate(pre.X_train, pre.X_test, pre.y_train, pre.y_test, models=models,
                             cv_folds=3, build_ensemble=False)
    assert res["Fragile"].fit_status == "cv_failed"
    assert selection_basis(res) == "cv"
    assert explain_selection(res)[0] == "Ridge"


def test_knn_works_in_small_time_series_folds():
    rng = np.random.default_rng(6)
    n = 45
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n, freq="MS").astype(str),
                       "promo": rng.normal(size=n)})
    df["sales"] = 100 + 2 * np.arange(n) + 5 * df.promo + rng.normal(0, 3, n)
    out = run_full_pipeline(df_train=df, target="sales", split_strategy="time", time_column="date",
                            selected_models=[BASELINE_MODEL_NAME, "Ridge", "KNN"], build_ensemble=False, **FAST)
    assert out.results["KNN"].fit_status != "cv_failed"
    assert out.selection_basis == "cv"
    assert out.best_model == "Ridge"          # strong trend: must NOT be called "no signal"


def test_pure_noise_is_reported_as_no_signal():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(250, 10)), columns=[f"x{i}" for i in range(10)])
    df["y"] = rng.normal(size=250)
    pre = preprocess(df, "y")
    res = _fit_pipeline(pre, [BASELINE_MODEL_NAME, "LinearRegression", "Ridge", "KNN",
                              "HistGradientBoosting"], cv_folds=5)
    winner, note = explain_selection(res)
    assert winner == BASELINE_MODEL_NAME and "no reliable signal" in note
    flexible = [r.fit_status for n, r in res.items() if n not in (BASELINE_MODEL_NAME, ENSEMBLE_NAME)]
    assert "underfit" not in flexible          # no capacity is pushed onto noise


def test_all_models_share_the_same_folds():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"a": rng.normal(size=200)})
    df["y"] = df.a + rng.normal(0, .5, 200)
    pre = preprocess(df, "y")
    res = _fit_pipeline(pre, [BASELINE_MODEL_NAME, "Ridge", "DecisionTree"], cv_folds=5,
                        tuning="fast", n_iter=3, build_ensemble=False)
    n_splits = {r.metrics["CV_n_splits"] for r in res.values()}
    assert n_splits == {15}                    # small data: 3x5 repeated folds for EVERY model


# ---- target transform ---------------------------------------------------------------------

def test_integer_prices_are_not_mistaken_for_counts():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(1000, 3))
    df = pd.DataFrame(X, columns=list("abc"))
    df["price"] = np.round(np.exp(12 + X @ [0.4, 0.3, -0.2] + rng.normal(0, .3, 1000)))
    out = run_full_pipeline(df_train=df, target="price",
                            selected_models=[BASELINE_MODEL_NAME, "Ridge", "HistGradientBoosting"],
                            build_ensemble=False, **FAST)
    assert out.target_transform_decision["method"] == "cv"
    assert out.preprocessing.target_transform == "log1p"


def test_smearing_removes_the_log_retransformation_bias():
    rng = np.random.default_rng(10)
    y_log = rng.normal(5, 0.6, 4000)
    pred = np.full_like(y_log, 5.0)                    # a perfect median predictor
    folds = {"splits": [(np.arange(0), np.arange(4000))]}
    s = smearing_factor(y_log, [pred], folds)
    naive, corrected = np.expm1(pred).mean(), apply_smearing(pred, s).mean()
    truth = np.expm1(y_log).mean()
    assert s > 1.1 and abs(corrected - truth) < 0.25 * abs(naive - truth)


# ---- intervals, early stopping, ensembles ------------------------------------------------------

def test_adaptive_interval_is_honest_where_errors_are_large():
    rng = np.random.default_rng(11)
    n = 3000
    x = rng.uniform(0, 1, n)
    y = x + rng.normal(0, 0.05 + 1.0 * x, n)         # noise grows with x
    df = pd.DataFrame({"x": x, "y": y})
    pre = preprocess(df, "y")
    est = get_default_model_zoo()["LinearRegression"].fit(pre.X_train, pre.y_train)
    hard = pre.X_test[:, 0] > np.quantile(pre.X_test[:, 0], 0.8)
    cover = {}
    for adaptive in (False, True):
        pi = compute_prediction_interval(est, pre.X_train, pre.y_train, coverage=0.9, adaptive=adaptive)
        p = est.predict(pre.X_test)
        from utils.modeling import _interval_halfwidth
        hw = _interval_halfwidth(pi, pre.X_test, len(p))
        inside = np.abs(pre.y_test - p) <= hw
        cover[adaptive] = (inside.mean(), inside[hard].mean())
    assert 0.85 <= cover[True][0] <= 0.95
    assert cover[True][1] > cover[False][1] + 0.05      # much better on the noisy rows


def test_boosters_use_early_stopping():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(12)
    X = rng.normal(size=(600, 5))
    y = X[:, 0] + rng.normal(0, 1, 600)
    m = get_default_model_zoo()["XGBoost"]
    assert isinstance(m, EarlyStoppingRegressor)
    m.fit(X, y)
    assert m.best_iteration_ is not None and m.best_iteration_ < 1000


def test_ensemble_uses_distinct_families_and_rebuilds_exactly():
    rng = np.random.default_rng(13)
    df = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    df["y"] = df.a + np.sin(3 * df.b) + rng.normal(0, .3, 500)
    pre = preprocess(df, "y")
    res = _fit_pipeline(pre, [BASELINE_MODEL_NAME, "Ridge", "Lasso", "RandomForest", "ExtraTrees", "KNN"],
                        cv_folds=3)
    ens = res[ENSEMBLE_NAME]
    fams = {res[m].family for m in ens.ensemble_members}
    assert len(fams) == 3
    from crew.orchestrator import _winner_spec
    rebuilt = rebuild_estimator(_winner_spec(ENSEMBLE_NAME, res), pre.preprocessor_templates)
    rebuilt.fit(pre.X_train_raw, pre.y_train)
    np.testing.assert_allclose(rebuilt.predict(pre.X_test_raw), ens.y_pred_test_model, rtol=1e-8)


# ---- audits and agent tools -------------------------------------------------------------------

def test_leakage_audit_catches_a_log_of_the_target():
    rng = np.random.default_rng(14)
    n = 800
    df = pd.DataFrame({"sqft": rng.uniform(500, 4000, n)})
    y = np.exp(11 + df.sqft / 3000 + rng.normal(0, .6, n))
    df["price_log"] = np.log(y)                      # Pearson r with y is only ~0.9
    df["noise"] = rng.normal(size=n)
    sus = leakage_suspects(df, y, "price")
    assert [s["column"] for s in sus if s["strong"]] == ["price_log"]


def test_residual_diagnostics_detect_bias_and_fanning():
    rng = np.random.default_rng(15)
    p = rng.uniform(1, 10, 2000)
    y = p + 1.0 + rng.normal(0, 0.3 * p)
    d = residual_diagnostics(y, p)
    text = " ".join(d["findings"])
    assert "biased" in text and "heteroscedasticity" in text


def test_reports_do_not_claim_negative_values_or_hide_skew():
    rng = np.random.default_rng(16)
    n = 600
    X = rng.normal(size=(n, 2))
    df = pd.DataFrame(X, columns=["a", "b"])
    df["y"] = np.exp(3 + X[:, 0] + rng.normal(0, 1.2, n))      # heavy skew, all positive
    out = run_full_pipeline(df_train=df, target="y", selected_models=[BASELINE_MODEL_NAME, "Ridge"],
                            build_ensemble=False, log_transform_target=False, **FAST)
    assert "negative" not in tools.preprocess_text(out).lower()
    assert "skew" in tools.quality_review_text(out).lower()


def test_llm_proposals_are_whitelisted_and_drops_need_confirmation():
    class Out:
        run_options = {"tuning": "off"}
        run_inputs = {"df_train": pd.DataFrame({"a": [1], "leak": [2], "y": [3]})}
        options_used = {"target": "y"}
    reply = ('{"actions": [{"action": "tuning", "value": "fast"}, {"action": "exec", "value": "rm"},'
             ' {"action": "drop_columns", "value": ["leak", "missing"]}]}')
    props = advisor.llm_proposals(Out(), lambda prompt: reply, "")
    assert [(p.action, p.value, p.auto_ok) for p in props] == [
        ("tuning", "fast", True), ("drop_columns", ["leak"], False)]


def test_refinement_loop_keeps_a_change_only_when_cv_improves():
    rng = np.random.default_rng(17)
    a, b = rng.uniform(-2, 2, 500), rng.uniform(-2, 2, 500)
    df = pd.DataFrame({"a": a, "b": b, "y": a * b + rng.normal(0, .2, 500)})
    out = run_full_pipeline(df_train=df, target="y", selected_models=[BASELINE_MODEL_NAME, "Ridge"],
                            build_ensemble=False, auto_refine=True, **FAST)
    kept = [e for e in out.refinement_log if e["accepted"]]
    assert any(("add_interactions", True) in e["actions"] for e in kept)
    assert out.results[out.best_model].metrics["R2"] > 0.8


# ---- artifacts ---------------------------------------------------------------------------

def test_generated_script_reproduces_the_app_winner(tmp_path):
    rng = np.random.default_rng(18)
    n = 500
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    df["y"] = np.exp(2 + df.a + 0.5 * df.b + rng.normal(0, .3, n))
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    out = run_full_pipeline(df_train=df, target="y", csv_filename=str(csv), tuning="fast",
                            selected_models=[BASELINE_MODEL_NAME, "Ridge", "HistGradientBoosting"],
                            build_ensemble=False, compute_learning_curves=False,
                            compute_permutation_importances=False, output_dir=str(tmp_path / "run"))
    os.environ["REGRESSION_CREW_DIR"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with redirect_stdout(io.StringIO()):
            g = runpy.run_path(out.script_path, run_name="__main__")
    finally:
        os.chdir(cwd)
    app = out.results[out.best_model].metrics
    for k in ("RMSE", "MAE", "R2"):
        assert g["WINNER_TEST_METRICS"][k] == pytest.approx(app[k], rel=1e-9)


def test_each_run_writes_to_its_own_directory():
    rng = np.random.default_rng(19)
    df = pd.DataFrame({"a": rng.normal(size=120)})
    df["y"] = df.a + rng.normal(0, .1, 120)
    kw = dict(selected_models=[BASELINE_MODEL_NAME, "Ridge"], build_ensemble=False, **FAST)
    a = run_full_pipeline(df_train=df, target="y", **kw)
    b = run_full_pipeline(df_train=df, target="y", **kw)
    assert a.output_dir != b.output_dir and a.model_bundle_path != b.model_bundle_path


def test_csv_reader_handles_semicolons_and_latin1(tmp_path):
    p = tmp_path / "eu.csv"
    p.write_bytes("ville;prix\nZürich;1,5\nGenève;2,5\n".encode("latin-1"))
    df = read_table(str(p))
    assert list(df.columns) == ["ville", "prix"] and df.shape == (2, 2)
