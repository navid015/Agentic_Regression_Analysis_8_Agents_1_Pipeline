"""
CrewAI tools — wrap deterministic utilities, return concrete numbers
so the agents can write specific commentary instead of vague summaries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from crewai.tools import tool

from utils.modeling import get_default_model_zoo, rank_models, train_and_evaluate
from utils.preprocessing import preprocess, profile_dataframe

# Module-level state populated by the orchestrator before crew.kickoff().
STATE: dict[str, Any] = {
    "df_train": None, "df_val": None, "df_test": None,
    "target": None, "test_size": 0.2, "random_state": 42,
    "selected_models": None, "cv_folds": 5, "cv_strategy": "kfold",
    "split_strategy": "random", "time_column": None, "group_column": None,
    "log_transform_target": False, "auto_datetime_features": True,
    "drop_low_variance": True, "tune_hyperparameters": False,
    "profile": None, "preprocessing": None, "results": None,
}


def reset_state() -> None:
    for k in STATE:
        STATE[k] = None
    STATE.update({"test_size": 0.2, "random_state": 42, "cv_folds": 5,
                  "cv_strategy": "kfold", "split_strategy": "random",
                  "log_transform_target": False, "auto_datetime_features": True,
                  "drop_low_variance": True, "tune_hyperparameters": False})


@tool("Profile dataset")
def profile_dataset_tool(_: str = "") -> str:
    """Profile the training set: shape, dtypes, missingness, target stats,
    datetime candidates, low-variance columns, high-cardinality columns.
    """
    df = STATE["df_train"]
    target = STATE["target"]
    if df is None or target is None:
        return "ERROR: dataset and target must be loaded before profiling."
    p = profile_dataframe(df, target)
    STATE["profile"] = p

    lines = [
        f"Shape: {p['n_rows']} rows × {p['n_cols']} cols",
        f"Target: '{p['target']}'",
        f"  mean={p['target_stats']['mean']:.3f}, std={p['target_stats']['std']:.3f}, "
        f"min={p['target_stats']['min']:.3f}, max={p['target_stats']['max']:.3f}, "
        f"skew={p['target_stats']['skew']:+.2f}",
        f"Total missing values: {p['missing_total']}",
    ]

    # type breakdown
    n_num = sum(1 for c in p["columns"] if c["kind"] == "numeric")
    n_cat = sum(1 for c in p["columns"] if c["kind"] == "categorical")
    n_dt  = sum(1 for c in p["columns"] if c["kind"] == "datetime")
    lines.append(f"Feature types: {n_num} numeric, {n_cat} categorical, {n_dt} datetime")

    # specific findings
    if p["datetime_candidates"]:
        lines.append(f"Datetime columns detected: {p['datetime_candidates']}")
    if p["low_variance_columns"]:
        lines.append(f"Low-variance columns (will be dropped): {p['low_variance_columns']}")
    if p["high_cardinality_columns"]:
        lines.append(f"High-cardinality categoricals (will be frequency-encoded): "
                     f"{p['high_cardinality_columns']}")

    # missingness — list top offenders
    high_missing = sorted(
        [(c["name"], c["missing_pct"]) for c in p["columns"] if c["missing_pct"] > 5],
        key=lambda x: -x[1],
    )[:5]
    if high_missing:
        lines.append("Columns with >5% missing: " +
                     ", ".join(f"{n}({pct:.1f}%)" for n, pct in high_missing))

    if abs(p["target_stats"]["skew"]) > 1.5:
        lines.append(f"Target is heavily skewed (skew={p['target_stats']['skew']:+.2f}) — "
                     "consider log-transform.")
    return "\n".join(lines)


@tool("Preprocess dataset")
def preprocess_dataset_tool(_: str = "") -> str:
    """Run preprocessing using the options stored in STATE.

    Stores the result for downstream tools.
    """
    if STATE["df_train"] is None or STATE["target"] is None:
        return "ERROR: dataset and target must be loaded first."
    try:
        result = preprocess(
            STATE["df_train"], STATE["target"],
            df_val=STATE["df_val"], df_test=STATE["df_test"],
            test_size=STATE["test_size"], random_state=STATE["random_state"],
            split_strategy=STATE["split_strategy"], time_column=STATE["time_column"],
            group_column=STATE["group_column"],
            log_transform_target=STATE["log_transform_target"],
            auto_datetime_features=STATE["auto_datetime_features"],
            drop_low_variance=STATE["drop_low_variance"],
        )
    except ValueError as e:
        return f"ERROR: {e}"
    STATE["preprocessing"] = result
    s = result.summary
    lines = [
        f"Train rows: {s['n_train']}, Test rows: {s['n_test']}, "
        f"Val rows: {s['n_val']}",
        f"Features after encoding: {s['n_features_after']}",
        f"  numeric: {len(s['numeric_cols'])}",
        f"  one-hot encoded: {len(s['low_cardinality_categorical'])}",
        f"  frequency-encoded (high-card): "
        f"{len(s['high_cardinality_categorical_freq_encoded'])}",
        f"Split strategy: {s['split_strategy_used']}",
        f"Target transform: {s['target_transform']}",
    ]
    if s.get("datetime_cols_extracted"):
        lines.append(f"Datetime features extracted from: {s['datetime_cols_extracted']}")
    if s.get("low_variance_dropped"):
        lines.append(f"Dropped low-variance columns: {s['low_variance_dropped']}")
    return "\n".join(lines)


@tool("Train and evaluate models")
def train_models_tool(_: str = "") -> str:
    """Train every regressor; compute MAE/RMSE/R²/MAPE plus CV scores."""
    pre = STATE["preprocessing"]
    if pre is None:
        return "ERROR: must run preprocessing before training."

    zoo = get_default_model_zoo(random_state=STATE["random_state"])
    if STATE.get("selected_models"):
        zoo = {k: v for k, v in zoo.items() if k in STATE["selected_models"]}

    results = train_and_evaluate(
        pre.X_train, pre.X_test, pre.y_train, pre.y_test,
        models=zoo,
        cv_folds=STATE["cv_folds"],
        cv_strategy=STATE["cv_strategy"],
        groups_train=pre.groups_train,
        target_transform=pre.target_transform,
        tune_hyperparameters=STATE["tune_hyperparameters"],
    )
    STATE["results"] = results
    ranked = rank_models(results)
    lines = ["Trained models (best to worst by RMSE):"]
    for name, rmse, r2 in ranked:
        m = results[name].metrics
        lines.append(
            f"  {name}: RMSE={rmse:.4f}, MAE={m['MAE']:.4f}, R²={r2:.4f}"
        )
    lines.append(f"\nWinner: {ranked[0][0]}")
    return "\n".join(lines)


@tool("Get best model summary")
def best_model_tool(_: str = "") -> str:
    """Detailed metrics for the winning model."""
    results = STATE["results"]
    if not results:
        return "ERROR: no models trained yet."
    ranked = rank_models(results)
    best = ranked[0][0]
    m = results[best].metrics
    lines = [
        f"Best model: {best}",
        f"  RMSE: {m['RMSE']:.4f} (train: {m.get('RMSE_train', float('nan')):.4f})",
        f"  MAE:  {m['MAE']:.4f}",
        f"  R²:   {m['R2']:.4f} (train: {m.get('R2_train', float('nan')):.4f})",
        f"  MedianAE: {m['MedianAE']:.4f}",
    ]
    if "CV_R2_mean" in m:
        lines.append(f"  CV R²: {m['CV_R2_mean']:.4f} ± {m['CV_R2_std']:.4f}")
    if "R2_train" in m and m["R2_train"] - m["R2"] > 0.15:
        lines.append("  ⚠ POSSIBLE OVERFITTING: train R² much higher than test R².")
    return "\n".join(lines)


@tool("Quality review of pipeline")
def quality_review_tool(_: str = "") -> str:
    """Audit for leakage signals, overfitting, degenerate targets, dimension blow-ups.

    Real target leakage almost always shows up as a SINGLE feature with
    near-perfect correlation with the target. R²≈1.0 on both train and test
    can also occur on highly predictable datasets (e.g. diamonds, where
    carat, x, y, z together almost determine price), so we only flag leakage
    confidently when both signals are present.
    """
    issues: list[str] = []
    p = STATE["profile"]; pre = STATE["preprocessing"]; results = STATE["results"]
    df_train = STATE["df_train"]; target = STATE["target"]

    # Primary leakage signal: any single feature nearly perfectly correlated with target
    leaked_features: list[tuple[str, float]] = []
    if df_train is not None and target is not None:
        for col in df_train.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(df_train[col]):
                try:
                    corr = abs(df_train[col].corr(df_train[target]))
                    if not pd.isna(corr) and corr > 0.99:
                        leaked_features.append((col, float(corr)))
                except Exception:
                    pass
    if leaked_features:
        for col, corr in sorted(leaked_features, key=lambda x: -x[1]):
            issues.append(
                f"Feature '{col}' has |correlation| = {corr:.4f} with target "
                f"'{target}' — almost certainly target leakage."
            )

    if p:
        if p["target_stats"]["std"] < 1e-9:
            issues.append("Target has near-zero variance — regression is degenerate.")
        if abs(p["target_stats"]["skew"]) > 2.0 and not STATE["log_transform_target"]:
            issues.append(
                f"Target skew is {p['target_stats']['skew']:+.2f}; consider log-transform."
            )
    if pre:
        s = pre.summary
        if s["n_features_after"] > s["n_train"]:
            issues.append(
                f"More features ({s['n_features_after']}) than training rows "
                f"({s['n_train']}) — high overfitting risk; prefer regularized models."
            )
    if results:
        for name, r in results.items():
            tr, te = r.metrics.get("R2_train"), r.metrics.get("R2")
            if tr is None or te is None:
                continue
            if te > 0.9999 and tr > 0.9999:
                if leaked_features:
                    issues.append(
                        f"{name}: R² ≈ 1.0 on both splits AND a high-correlation "
                        "feature was found — leakage strongly suggested."
                    )
                else:
                    issues.append(
                        f"{name}: R² ≈ 1.0 on both train and test, but no single "
                        "feature has > 0.99 correlation with the target. This usually "
                        "means the dataset is highly predictable (multiple informative "
                        "features collectively determine the target), not leakage. "
                        "Worth eyeballing — but the model is probably fine."
                    )
            elif tr - te > 0.20:
                issues.append(
                    f"{name}: train R² {tr:.3f} >> test R² {te:.3f} → overfitting."
                )
            if te < 0:
                issues.append(f"{name}: negative test R² → worse than predicting the mean.")
    return "Quality review: no major red flags." if not issues \
        else "Quality review findings:\n  - " + "\n  - ".join(issues)
