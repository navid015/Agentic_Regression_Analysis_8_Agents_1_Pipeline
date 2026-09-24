"""
CrewAI tools — wrap deterministic utilities, return concrete numbers
so the agents can write specific commentary instead of vague summaries.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from crewai.tools import tool

from utils.modeling import (
    BASELINE_MODEL_NAME,
    explain_selection,
    filter_zoo_for_data,
    get_default_model_zoo,
    rank_models,
    target_outlier_share,
    train_and_evaluate,
)
from utils.preprocessing import preprocess, profile_dataframe

# Module-level state populated by the orchestrator before crew.kickoff().
STATE: dict[str, Any] = {
    "df_train": None, "df_val": None, "df_test": None,
    "target": None, "test_size": 0.2, "random_state": 42,
    "selected_models": None, "cv_folds": 5, "cv_strategy": "kfold",
    "split_strategy": "random", "time_column": None, "group_column": None,
    "log_transform_target": False, "auto_datetime_features": True,
    "drop_low_variance": True, "tune_hyperparameters": False,
    "tuning": "off",
    "profile": None, "preprocessing": None, "results": None,
    # filled by the orchestrator so every agent names the SAME winner as the app
    "best_model": None, "selection_note": "", "selection_metric": "rmse",
    "skipped_models": {}, "failed_models": {},
}


_DEFAULT_STATE: dict[str, Any] = dict(STATE)


def reset_state() -> None:
    """Restore STATE to its declared defaults.

    Previously this blanked every key to None and then re-applied only some of
    them by hand, so any default not repeated in that literal was silently lost
    and the two lists could drift apart. Snapshotting the declared defaults once
    keeps them in exactly one place.
    """
    STATE.clear()
    STATE.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULT_STATE.items()})


def _winner(results) -> tuple[str, str]:
    """The app's winner (from the orchestrator) or, if absent, the same rule it uses."""
    if STATE.get("best_model") in (results or {}):
        return STATE["best_model"], STATE.get("selection_note", "")
    return explain_selection(results, selection_metric=STATE.get("selection_metric", "rmse"))


@tool("Profile dataset")
def profile_dataset_tool(_: str = "") -> str:
    """Profile the training set: shape, dtypes, missingness, target stats,
    datetime candidates, low-variance columns, high-cardinality columns.
    """
    df = STATE["df_train"]
    target = STATE["target"]
    if df is None or target is None:
        return "ERROR: dataset and target must be loaded before profiling."
    # Read-through cache. The orchestrator already profiled the data before
    # kicking off the crew; recomputing here doubled the work on every run.
    p = STATE.get("profile")
    if p is None:
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
    # Read-through cache: the orchestrator already preprocessed before kicking
    # off the crew. Recomputing here ran the whole pipeline twice per request.
    result = STATE.get("preprocessing")
    if result is None:
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
    if s.get("id_like_columns_dropped"):
        lines.append(f"Dropped ID/row-index columns: {s['id_like_columns_dropped']}")
    if s.get("duplicate_rows_dropped"):
        lines.append(f"Dropped exact duplicate rows: {s['duplicate_rows_dropped']}")
    if s.get("infinite_values_sanitized"):
        lines.append("Infinite values converted to missing: "
                     f"{s['infinite_values_sanitized']}")
    if s.get("target_transform_requested") and s.get("target_transform") == "none":
        lines.append("NOTE: log-transform was requested but skipped "
                     "(target contains negative values).")
    return "\n".join(lines)


@tool("Train and evaluate models")
def train_models_tool(_: str = "") -> str:
    """Train every regressor; compute MAE/RMSE/R²/MAPE plus CV scores."""
    pre = STATE["preprocessing"]
    if pre is None:
        return "ERROR: must run preprocessing before training."

    # Read-through cache. Retraining every model here doubled the single most
    # expensive operation in the pipeline (and tripled it with tuning on).
    results = STATE.get("results")
    if results is None:
        zoo = get_default_model_zoo(random_state=STATE["random_state"])
        if STATE.get("selected_models"):
            zoo = {k: v for k, v in zoo.items() if k in STATE["selected_models"]}
        zoo, _ = filter_zoo_for_data(zoo, pre.y_train, target_transform=pre.target_transform)
        results = train_and_evaluate(
            pre.X_train, pre.X_test, pre.y_train, pre.y_test,
            models=zoo,
            cv_folds=STATE["cv_folds"],
            cv_strategy=STATE["cv_strategy"],
            groups_train=pre.groups_train,
            target_transform=pre.target_transform,
            tuning=STATE.get("tuning") or ("fast" if STATE["tune_hyperparameters"] else "off"),
            X_val=pre.X_val, y_val=pre.y_val,
            random_state=STATE["random_state"],
        )
        STATE["results"] = results
    metric = STATE.get("selection_metric", "rmse").upper()
    key = f"CV_{metric}_mean"
    order = sorted(results, key=lambda n: results[n].metrics.get(key, float("inf")))
    lines = [f"Trained models (best to worst by cross-validated {metric} on training rows):"]
    for name in order:
        r = results[name]; m = r.metrics
        lines.append(
            f"  {name}: CV {metric}={m.get(key, float('nan')):.4f}, "
            f"CV R²={m.get('CV_R2_mean', float('nan')):.4f} ± {m.get('CV_R2_std', float('nan')):.4f}, "
            f"test RMSE={m['RMSE']:.4f}, test R²={m['R2']:.4f}, fit={r.fit_status}"
            + (f" [auto-fix: {r.remediation}]" if r.remediation else "")
        )
    best, note = _winner(results)
    lines.append(f"\nWinner: {best} — selected by {note}")
    if STATE.get("skipped_models"):
        lines.append("Skipped models: " + "; ".join(f"{k} ({v})" for k, v in STATE["skipped_models"].items()))
    if STATE.get("failed_models"):
        lines.append("Failed models: " + "; ".join(f"{k} ({v})" for k, v in STATE["failed_models"].items()))
    return "\n".join(lines)


@tool("Get best model summary")
def best_model_tool(_: str = "") -> str:
    """Detailed metrics for the winning model."""
    results = STATE["results"]
    if not results:
        return "ERROR: no models trained yet."
    best, note = _winner(results)
    r = results[best]
    m = r.metrics
    lines = [
        f"Best model: {best} (selected by {note})",
        f"  Fit diagnosis: {r.fit_status} — " + "; ".join(r.fit_reasons),
        f"  RMSE: {m['RMSE']:.4f} (train: {m.get('RMSE_train', float('nan')):.4f})",
        f"  MAE:  {m['MAE']:.4f}",
        f"  R²:   {m['R2']:.4f} (train: {m.get('R2_train', float('nan')):.4f})",
        f"  MedianAE: {m['MedianAE']:.4f}",
    ]
    if "CV_R2_mean" in m:
        lines.append(f"  CV R²: {m['CV_R2_mean']:.4f} ± {m['CV_R2_std']:.4f}"
                     + (" (from the tuning search; mildly optimistic)" if r.cv_optimistic else ""))
    if "Skill_vs_baseline_pct" in m:
        lines.append(f"  Error reduction vs. predicting the average: {m['Skill_vs_baseline_pct']:.1f}%")
    if r.best_params:
        lines.append(f"  Final settings: {r.best_params}")
    if r.remediation:
        lines.append(f"  Automatic fix: {r.remediation}")
    if r.ensemble_members:
        lines.append(f"  Ensemble of: {', '.join(r.ensemble_members)}")
    if r.learning_curve:
        lines.append(f"  Learning curve: {r.learning_curve['reading']}")
    pi = r.prediction_interval
    if pi and "test_coverage" in pi:
        lines.append(f"  {pi['coverage_target']:.0%} prediction interval: covers "
                     f"{pi['test_coverage']:.0%} of test rows, average width "
                     f"{pi['mean_width_original_units']:.4g}")
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

    # Columns the pipeline already excluded from training (auto-dropped ID/index
    # columns, low-variance columns, the group column) shouldn't be flagged as
    # leakage — they carry no signal into any trained model, so a high raw
    # correlation there is a non-issue, not a red flag.
    excluded_cols: set = set()
    if pre is not None:
        s = pre.summary
        excluded_cols.update(s.get("id_like_columns_dropped") or [])
        excluded_cols.update(s.get("low_variance_dropped") or [])
        if s.get("group_column"):
            excluded_cols.add(s["group_column"])
    if STATE.get("time_column"):
        excluded_cols.add(STATE["time_column"])

    # Primary leakage signal: any single feature nearly perfectly correlated with target
    leaked_features: list[tuple[str, float]] = []
    if df_train is not None and target is not None:
        for col in df_train.columns:
            if col == target or col in excluded_cols:
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
    if df_train is not None and target is not None:
        y_model = pre.y_train if pre is not None else df_train[target].to_numpy(dtype=float)
        share = target_outlier_share(y_model)
        if share > 0.02:
            issues.append(f"{share:.1%} of target values are extreme outliers; RMSE and R² are "
                          "dominated by them, so the selection used "
                          f"{STATE.get('selection_metric', 'rmse').upper()}. Judge models by "
                          "MAE / MedianAE and consider the Huber model.")

    if results:
        best, note = _winner(results)
        for name, r in results.items():
            if name == BASELINE_MODEL_NAME:
                continue
            tr, te = r.metrics.get("R2_train"), r.metrics.get("R2")
            tag = " (WINNER)" if name == best else ""
            if tr is not None and te is not None and te > 0.9999 and tr > 0.9999:
                if leaked_features:
                    issues.append(f"{name}{tag}: R² ≈ 1.0 on both splits AND a high-correlation "
                                  "feature was found — leakage strongly suggested.")
                else:
                    issues.append(f"{name}{tag}: R² ≈ 1.0 on both train and test, but no single "
                                  "feature has > 0.99 correlation with the target — probably a "
                                  "highly predictable dataset rather than leakage. Worth a look.")
            if r.fit_status == "benign" and name == best:
                issues.append(f"{name}{tag}: train/held-out gap remains, but stronger "
                              "regularisation did not reduce held-out error (harmless gap).")
            if r.fit_status in ("overfit", "underfit", "unstable"):
                fixed = f" Auto-fix: {r.remediation}." if r.remediation else ""
                issues.append(f"{name}{tag}: {r.fit_status.upper()} — "
                              + "; ".join(r.fit_reasons) + "." + fixed)
            if te is not None and te < 0:
                issues.append(f"{name}{tag}: negative test R² → worse than predicting the mean.")
        if STATE.get("split_strategy") == "time" or STATE.get("cv_strategy") == "time":
            tree_like = [n for n in results if any(k in n for k in (
                "Tree", "Forest", "Boost", "XGB", "LightGBM", "KNN"))]
            bad = [n for n in tree_like if results[n].metrics.get("R2", 0) < 0]
            if bad:
                issues.append("Time-ordered data: " + ", ".join(bad) + " score below the mean "
                              "on the future test period. Tree/neighbour models cannot predict "
                              "values outside the range seen in training, so they fail on "
                              "trending series; prefer linear models, or predict the change "
                              "from the previous value instead of the level.")
        w = results[best]
        pi = w.prediction_interval
        if pi and "test_coverage" in pi and pi["test_coverage"] < pi["coverage_target"] - 0.10:
            issues.append(f"Winner's {pi['coverage_target']:.0%} prediction interval only covered "
                          f"{pi['test_coverage']:.0%} of test rows — the test data may differ "
                          "from the training data (distribution shift).")
        if w.cv_optimistic:
            issues.append(f"Winner {best} was tuned; its CV score comes from the search itself "
                          "and is mildly optimistic. Rely on the test score, or enable nested CV.")
        issues.append(f"Winner: {best} — selected by {note}; fit diagnosis: {w.fit_status}.")
    for k, v in (STATE.get("failed_models") or {}).items():
        issues.append(f"{k} failed to train and was left out: {v}")
    red = [i for i in issues if not i.startswith("Winner:")]
    if not red:
        return "Quality review: no major red flags.\n  - " + "\n  - ".join(issues)
    return "Quality review findings:\n  - " + "\n  - ".join(issues)
