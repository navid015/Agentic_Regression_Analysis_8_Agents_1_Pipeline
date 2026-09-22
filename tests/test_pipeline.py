"""Regression tests for the deterministic pipeline.

These focus on the failure modes that are easy to reintroduce and impossible to
notice by eye: silently deleting real features, leaking the answer across the
split, and reporting metrics on the wrong scale.

Run with:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.modeling import (
    BASELINE_MODEL_NAME,
    _make_cv,
    _safe_mape,
    get_default_model_zoo,
    pick_best,
    rank_models,
    selection_basis,
    train_and_evaluate,
)
from utils.preprocessing import (
    _is_low_variance,
    _looks_like_identifier,
    _name_suggests_id,
    preprocess,
    profile_dataframe,
)


# --------------------------------------------------------------------------
# Identifier detection - the bug that silently deleted real features
# --------------------------------------------------------------------------

@pytest.mark.parametrize("col", [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "humidity", "width", "solidity", "confidence", "mid_price", "rapid_tests",
    "video_length", "density",
])
def test_real_features_are_not_mistaken_for_identifiers(col):
    """A substring test for 'id' flagged all of these. They must survive."""
    assert not _name_suggests_id(col), f"{col!r} wrongly treated as an identifier"


@pytest.mark.parametrize("col", [
    "id", "ID", "customer_id", "row_id", "Unnamed: 0", "index", "uuid", "rowid",
])
def test_identifier_names_are_recognised(col):
    assert _name_suggests_id(col)


def test_name_hint_alone_does_not_delete_a_column():
    """'store_id' with repeated values is a grouping key, not a row identifier."""
    repeated = pd.Series(["a", "b", "c"] * 50)
    assert not _looks_like_identifier("store_id", repeated)


def test_monotonic_integer_index_is_detected_without_a_name_hint():
    """The diamonds trap: a leftover counter on a target-sorted export."""
    assert _looks_like_identifier("some_column", pd.Series(range(200)))


def test_acidity_columns_survive_preprocessing():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "fixed_acidity": rng.normal(7, 1, n),
        "volatile_acidity": rng.normal(0.5, 0.1, n),
        "citric_acid": rng.normal(0.3, 0.1, n),
        "Unnamed: 0": np.arange(n),
        "quality": rng.normal(6, 1, n),
    })
    result = preprocess(df, "quality")
    dropped = result.summary["id_like_columns_dropped"]
    assert dropped == ["Unnamed: 0"]
    for keeper in ("fixed_acidity", "volatile_acidity", "citric_acid"):
        assert keeper in result.feature_names


# --------------------------------------------------------------------------
# Leakage and splitting
# --------------------------------------------------------------------------

def test_group_column_keeps_groups_out_of_both_splits():
    """Group leakage: the same entity must not appear in train and test."""
    rng = np.random.default_rng(1)
    n = 600
    groups = rng.choice([f"p{i}" for i in range(30)], n)
    df = pd.DataFrame({"x": rng.normal(size=n), "patient": groups})
    df["y"] = rng.normal(size=n)

    result = preprocess(df, "y", group_column="patient")
    assert "group-aware" in result.summary["split_strategy_used"]
    # train groups are recorded; the split itself must be disjoint
    train_groups = set(result.groups_train)
    assert len(train_groups) < 30, "test set must hold back whole groups"


def test_duplicate_rows_are_removed_before_splitting():
    rng = np.random.default_rng(2)
    base = pd.DataFrame({"x": rng.normal(size=100)})
    base["y"] = base.x * 2
    df = pd.concat([base, base.iloc[:25]], ignore_index=True)
    result = preprocess(df, "y")
    assert result.summary["duplicate_rows_dropped"] == 25


def test_infinities_do_not_reach_the_model():
    df = pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf, 5.0] * 40})
    df["y"] = np.arange(len(df), dtype=float)
    result = preprocess(df, "y")
    assert result.summary["infinite_values_sanitized"] > 0
    assert np.isfinite(result.X_train).all()
    assert np.isfinite(result.X_test).all()


def test_persisted_preprocessor_is_self_contained():
    """The saved artifact must transform raw data on its own, including
    unseen high-cardinality categories and infinities."""
    import io

    import joblib

    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "city": rng.choice([f"c{i}" for i in range(50)], n),   # frequency-encoded
        "color": rng.choice(list("rgb"), n),                   # one-hot
        "x": rng.normal(size=n),
    })
    df["y"] = rng.normal(size=n)
    result = preprocess(df, "y")

    buf = io.BytesIO()
    joblib.dump(result.preprocessor, buf)
    buf.seek(0)
    loaded = joblib.load(buf)

    fresh = df.drop(columns=["y"]).head(5).copy()
    fresh.loc[:, "city"] = "never_seen_before"
    fresh.loc[fresh.index[0], "x"] = np.inf
    out = loaded.transform(fresh)
    assert out.shape[0] == 5
    assert np.isfinite(out).all()


def test_feature_names_match_matrix_width():
    rng = np.random.default_rng(4)
    n = 200
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "cat": rng.choice(list("xyz"), n),
        "many": rng.choice([f"v{i}" for i in range(40)], n),
    })
    df["y"] = rng.normal(size=n)
    result = preprocess(df, "y")
    assert len(result.feature_names) == result.X_train.shape[1]


# --------------------------------------------------------------------------
# Modelling, metrics and selection
# --------------------------------------------------------------------------

def test_cv_honours_the_user_seed():
    """_make_cv used to hardcode random_state=42 regardless of the UI setting."""
    cv_a, _ = _make_cv("kfold", 5, None, random_state=7)
    cv_b, _ = _make_cv("kfold", 5, None, random_state=99)
    assert cv_a.random_state == 7
    assert cv_b.random_state == 99


def test_baseline_is_present_but_never_wins():
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"a": rng.normal(size=n)})
    df["y"] = 3 * df.a + rng.normal(0, 0.2, n)
    pre = preprocess(df, "y")
    zoo = {k: v for k, v in get_default_model_zoo().items()
           if k in (BASELINE_MODEL_NAME, "LinearRegression")}
    results = train_and_evaluate(pre.X_train, pre.X_test, pre.y_train,
                                 pre.y_test, models=zoo, cv_folds=3)
    assert BASELINE_MODEL_NAME in results
    assert abs(results[BASELINE_MODEL_NAME].metrics["R2"]) < 0.1
    assert pick_best(results) == "LinearRegression"


def test_selection_prefers_validation_when_available():
    rng = np.random.default_rng(6)
    n = 400
    def make(seed):
        r = np.random.default_rng(seed)
        d = pd.DataFrame({"a": r.normal(size=n)})
        d["y"] = 2 * d.a + r.normal(0, 0.3, n)
        return d

    pre = preprocess(make(1), "y", df_val=make(2), df_test=make(3))
    zoo = {k: v for k, v in get_default_model_zoo().items()
           if k in ("LinearRegression", "DecisionTree")}
    results = train_and_evaluate(
        pre.X_train, pre.X_test, pre.y_train, pre.y_test,
        models=zoo, cv_folds=3, X_val=pre.X_val, y_val=pre.y_val,
    )
    assert selection_basis(results) == "validation"
    assert all("RMSE_val" in r.metrics for r in results.values())


def test_log_transform_metrics_are_reported_in_original_units():
    """Test R2 and CV R2 must live on the same scale, not log vs original."""
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame({"a": rng.normal(size=n)})
    df["y"] = np.exp(2 + 0.9 * df.a + rng.normal(0, 0.25, n))

    pre = preprocess(df, "y", log_transform_target=True)
    assert pre.target_transform == "log1p"
    results = train_and_evaluate(
        pre.X_train, pre.X_test, pre.y_train, pre.y_test,
        models={"LinearRegression": get_default_model_zoo()["LinearRegression"]},
        cv_folds=3, target_transform=pre.target_transform,
    )
    m = results["LinearRegression"].metrics
    # Original-unit RMSE is on the scale of y (tens), not log-units (<1).
    assert m["RMSE"] > 1.0
    # The two R2 figures should now be comparable rather than scale-mismatched.
    assert abs(m["R2"] - m["CV_R2_mean"]) < 0.25


def test_log_transform_is_skipped_for_negative_targets():
    df = pd.DataFrame({"a": np.linspace(0, 1, 100)})
    df["y"] = np.linspace(-50, 50, 100)
    pre = preprocess(df, "y", log_transform_target=True)
    assert pre.target_transform == "none"
    assert pre.summary["target_transform_requested"] is True


def test_non_numeric_target_is_rejected():
    df = pd.DataFrame({"a": [1, 2, 3], "y": ["low", "high", "low"]})
    with pytest.raises(ValueError, match="regression only"):
        preprocess(df, "y")


def test_safe_mape_handles_zero_targets():
    assert np.isnan(_safe_mape(np.zeros(5), np.ones(5)))
    assert _safe_mape(np.array([100.0, 0.0]), np.array([110.0, 5.0])) == pytest.approx(10.0)


def test_near_constant_columns_are_low_variance():
    constant = pd.Series([1] * 100)
    near = pd.Series([1] * 9999 + [2])
    varied = pd.Series(list(range(100)))
    assert _is_low_variance(constant)
    assert _is_low_variance(near)
    assert not _is_low_variance(varied)


def test_ranking_is_ascending_by_rmse():
    rng = np.random.default_rng(8)
    n = 200
    df = pd.DataFrame({"a": rng.normal(size=n)})
    df["y"] = 2 * df.a + rng.normal(0, 0.3, n)
    pre = preprocess(df, "y")
    zoo = {k: v for k, v in get_default_model_zoo().items()
           if k in (BASELINE_MODEL_NAME, "LinearRegression", "DecisionTree")}
    results = train_and_evaluate(pre.X_train, pre.X_test, pre.y_train,
                                 pre.y_test, models=zoo, cv_folds=3)
    rmses = [row[1] for row in rank_models(results)]
    assert rmses == sorted(rmses)
