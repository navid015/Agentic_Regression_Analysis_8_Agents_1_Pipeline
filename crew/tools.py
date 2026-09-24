"""
CrewAI tools, built PER RUN.

v3 changes
----------
* No module-level STATE. The old global dict was shared by every user of the
  Gradio app, so two concurrent runs overwrote each other's data and agents
  could narrate someone else's results. `build_tools(out)` binds tools to one
  run's output through closures.
* The report functions are plain Python (`*_text(out)`), testable without
  CrewAI or an LLM; the tool wrappers are created lazily.
* Fixed: "log-transform was requested but skipped (target contains negative
  values)" was printed whenever auto mode declined the log, and the skew
  warning never fired because the string "auto" is truthy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.modeling import BASELINE_MODEL_NAME, ENSEMBLE_NAME, target_outlier_share


def _fmt(v, spec=".4f"):
    try:
        return format(float(v), spec) if v is not None and np.isfinite(float(v)) else "n/a"
    except Exception:
        return "n/a"


# ---- report functions ----------------------------------------------------------------

def profile_text(out) -> str:
    p = out.profile
    ts = p["target_stats"]
    lines = [
        f"Shape: {p['n_rows']} rows x {p['n_cols']} cols",
        f"Target: '{p['target']}'  mean={_fmt(ts['mean'], '.3f')}, std={_fmt(ts['std'], '.3f')}, "
        f"min={_fmt(ts['min'], '.3f')}, max={_fmt(ts['max'], '.3f')}, skew={ts['skew']:+.2f}",
        f"Total missing values: {p['missing_total']}",
    ]
    kinds: dict[str, int] = {}
    for c in p["columns"]:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    lines.append("Column kinds: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    for key, label in (("datetime_candidates", "Datetime columns"),
                       ("low_variance_columns", "Low-variance columns (dropped)"),
                       ("high_cardinality_columns", "High-cardinality categoricals"),
                       ("id_like_columns", "Row-counter / ID columns (dropped)"),
                       ("id_suspect_columns", "Unique sorted integer columns (KEPT, check them)"),
                       ("numeric_string_columns", "Numbers stored as text (parsed)"),
                       ("numeric_coded_categoricals", "Integer codes treated as categories")):
        if p.get(key):
            lines.append(f"{label}: {p[key]}")
    high_missing = sorted([(c["name"], c["missing_pct"]) for c in p["columns"] if c["missing_pct"] > 5],
                          key=lambda x: -x[1])[:5]
    if high_missing:
        lines.append("Columns with >5% missing: " + ", ".join(f"{n}({pct:.1f}%)" for n, pct in high_missing))
    return "\n".join(lines)


def preprocess_text(out) -> str:
    s = out.preprocessing.summary
    d = out.target_transform_decision or {}
    lines = [
        f"Train rows: {s['n_train']}, Test rows: {s['n_test']}, Val rows: {s['n_val']}",
        f"Features after encoding: {s['n_features_after']} (numeric {len(s['numeric_cols'])}, "
        f"one-hot {len(s['low_cardinality_categorical'])}, "
        f"{s.get('high_cardinality_encoding', 'target')}-encoded {len(s['high_cardinality_categorical'])})",
        f"Split strategy: {s['split_strategy_used']}",
        "Preprocessing is fitted inside every cross-validation fold (no held-out statistics leak in).",
    ]
    if d.get("method") == "cv":
        errs = d.get("errors", {})
        lines.append(f"Target transform: {s['target_transform']} - chosen by cross-validation "
                     f"(best CV error raw={_fmt(errs.get('none'), '.4g')}, "
                     f"log1p={_fmt(errs.get('log1p'), '.4g')})")
    elif d.get("method") == "rule":
        lines.append(f"Target transform: none - auto mode: {d.get('reason', '')} "
                     f"(training-target skew {s.get('target_skew_train', 0.0):+.2f})")
    else:
        lines.append(f"Target transform: {s['target_transform']} ({s.get('target_transform_reason', '')})")
    for key, label in (("datetime_cols_extracted", "Datetime features extracted from"),
                       ("low_variance_dropped", "Dropped low-variance columns"),
                       ("id_like_columns_dropped", "Dropped row-counter / ID columns"),
                       ("numeric_string_columns_parsed", "Parsed formatted numbers in"),
                       ("categorical_overrides", "Treated as categories"),
                       ("user_dropped_columns", "Dropped at your request")):
        if s.get(key):
            lines.append(f"{label}: {s[key]}")
    if s.get("duplicate_rows_dropped"):
        lines.append(f"Dropped exact duplicate rows: {s['duplicate_rows_dropped']}")
    if s.get("infinite_values_sanitized"):
        lines.append(f"Infinite values converted to missing: {s['infinite_values_sanitized']}")
    return "\n".join(lines)


def models_text(out) -> str:
    M = out.selection_metric.upper()
    key = f"CV_{M}_mean"
    res = out.results
    order = sorted(res, key=lambda n: res[n].metrics.get(key, float("inf")))
    lines = [f"Models, best to worst by cross-validated {M} on the training rows "
             "(all models scored on the SAME folds):"]
    for n in order:
        r, m = res[n], res[n].metrics
        lines.append(f"  {n}: CV {M}={_fmt(m.get(key))}, CV R2={_fmt(m.get('CV_R2_mean'))} "
                     f"+/- {_fmt(m.get('CV_R2_std'))}, fit={r.fit_status}"
                     + (" [tuned]" if r.best_params else "")
                     + (f" [auto-fix: {r.remediation}]" if r.remediation else ""))
    if (out.diagnostics or {}).get("probe_cv_error") is not None:
        lines.append(f"Reference flexible model (capacity probe) CV {M}: "
                     f"{_fmt(out.diagnostics['probe_cv_error'])} - used to decide under- vs low-signal.")
    lines.append(f"\nWinner: {out.best_model} - selected by {out.selection_note}")
    if out.skipped_models:
        lines.append("Skipped: " + "; ".join(f"{k} ({v})" for k, v in out.skipped_models.items()))
    if out.failed_models:
        lines.append("Failed: " + "; ".join(f"{k} ({v})" for k, v in out.failed_models.items()))
    lines.append("Test-set scores are reported separately and were not used for any decision.")
    return "\n".join(lines)


def best_model_text(out) -> str:
    r = out.results[out.best_model]
    m = r.metrics
    lines = [f"Best model: {out.best_model} (selected by {out.selection_note})",
             f"  Fit diagnosis: {r.fit_status} - " + "; ".join(r.fit_reasons),
             f"  Test RMSE {_fmt(m['RMSE'])}, MAE {_fmt(m['MAE'])}, R2 {_fmt(m['R2'])}, "
             f"MedianAE {_fmt(m['MedianAE'])}"]
    if "CV_R2_mean" in m:
        lines.append(f"  CV R2 {_fmt(m['CV_R2_mean'])} +/- {_fmt(m['CV_R2_std'])}"
                     + (" (hyperparameters were chosen on these rows: mildly optimistic)"
                        if r.cv_optimistic else ""))
    if "Skill_vs_baseline_pct" in m:
        lines.append(f"  Error reduction vs predicting the average: {m['Skill_vs_baseline_pct']:.1f}%")
    if r.smearing_factor:
        lines.append(f"  Log-target bias correction (smearing factor {r.smearing_factor:.4f}) applied; "
                     f"uncorrected test RMSE {_fmt(m.get('RMSE_uncorrected'))}")
    if r.best_params:
        lines.append(f"  Final settings: {r.best_params}")
    if r.remediation:
        lines.append(f"  Automatic fix: {r.remediation}")
    if r.ensemble_members:
        lines.append(f"  Ensemble of (one per model family): {', '.join(r.ensemble_members)}")
    if r.learning_curve:
        lines.append(f"  Learning curve: {r.learning_curve['reading']}")
    pi = r.prediction_interval
    if pi and "test_coverage" in pi:
        lines.append(f"  {pi['coverage_target']:.0%} {pi.get('method', 'constant')} prediction interval "
                     f"covers {pi['test_coverage']:.0%} of test rows (average width "
                     f"{_fmt(pi['mean_width_original_units'], '.4g')})")
    if out.final_model_note:
        lines.append(f"  Saved model: {out.final_model_note}")
    return "\n".join(lines)


def quality_findings(out) -> list[str]:
    issues: list[str] = []
    pre, res = out.preprocessing, out.results
    s = pre.summary
    for w in out.warnings or []:
        issues.append(f"Setup: {w}")
    for sus in out.leakage_suspects or []:
        tag = "LIKELY TARGET LEAKAGE" if sus["strong"] else "Very predictive single column"
        issues.append(f"{tag}: '{sus['column']}' {sus['reason']}. If it is only known after the target "
                      "(or is computed from it), drop it and re-run.")
    if s.get("train_test_duplicate_rows"):
        issues.append(f"{s['train_test_duplicate_rows']} test rows are exact copies of training rows: "
                      "test scores are inflated.")
    if s.get("train_val_duplicate_rows"):
        issues.append(f"{s['train_val_duplicate_rows']} validation rows duplicate training rows.")
    if s.get("train_test_group_overlap"):
        issues.append(f"{s['train_test_group_overlap']} groups appear in both train and test files: the "
                      "test measures memorisation of known groups, not generalisation to new ones.")
    if s.get("time_order_ok") is False:
        issues.append("The test (or validation) period starts before the training period ends: "
                      "this is not a forward-in-time evaluation.")
    if s.get("id_suspect_columns_kept"):
        issues.append(f"Kept unique, sorted integer column(s) {s['id_suspect_columns_kept']}: fine if they "
                      "are real measurements (a year, a size); drop them if they are identifiers.")
    ts = out.profile["target_stats"]
    if ts.get("std") is not None and np.isfinite(ts["std"]) and ts["std"] < 1e-9:
        issues.append("Target has near-zero variance - regression is degenerate.")
    if abs(ts.get("skew", 0.0)) > 2.0 and s["target_transform"] == "none":
        why = (out.target_transform_decision or {}).get("method")
        issues.append(f"Target skew is {ts['skew']:+.2f} and no log transform is applied"
                      + (" (cross-validation found the raw scale better)." if why == "cv"
                         else "; consider the log-transform option."))
    if s["n_features_after"] > s["n_train"]:
        issues.append(f"More features ({s['n_features_after']}) than training rows ({s['n_train']}): "
                      "high overfitting risk; prefer regularised models.")
    share = target_outlier_share(pre.y_train)
    if share > 0.02:
        issues.append(f"{share:.1%} of target values are extreme outliers; selection used "
                      f"{out.selection_metric.upper()}.")
    if out.best_model == BASELINE_MODEL_NAME:
        issues.append("NO RELIABLE SIGNAL: " + out.selection_note)
    for name, r in res.items():
        if name == BASELINE_MODEL_NAME:
            continue
        tag = " (WINNER)" if name == out.best_model else ""
        if r.fit_status in ("overfit", "underfit", "unstable", "cv_failed"):
            issues.append(f"{name}{tag}: {r.fit_status.upper()} - " + "; ".join(r.fit_reasons)
                          + (f" Auto-fix: {r.remediation}." if r.remediation else ""))
        te = r.metrics.get("R2")
        if te is not None and te < 0 and name == out.best_model:
            issues.append(f"{name}{tag}: negative test R2 -> worse than predicting the mean on the test rows.")
    if out.options_used.get("split_strategy") == "time":
        bad = [n for n, r in res.items() if r.family in ("tree", "bagging", "boosting", "neighbors")
               and r.metrics.get("R2", 0) < 0]
        if bad:
            issues.append("Time-ordered data: " + ", ".join(bad) + " score below the mean on the future "
                          "test period. Tree / neighbour models cannot extrapolate a trend; prefer linear "
                          "models or predict the change instead of the level.")
    w = res[out.best_model]
    pi = w.prediction_interval
    if pi and "test_coverage" in pi and pi["test_coverage"] < pi["coverage_target"] - 0.10:
        issues.append(f"The winner's {pi['coverage_target']:.0%} interval covered only "
                      f"{pi['test_coverage']:.0%} of test rows - the test data may differ from training.")
    if w.cv_optimistic:
        issues.append(f"{out.best_model}'s hyperparameters were tuned on the training rows, so its CV "
                      "score is mildly optimistic; enable nested CV for a fully honest estimate.")
    for k, v in (out.failed_models or {}).items():
        issues.append(f"{k} failed to train and was left out: {v}")
    return issues


def quality_review_text(out) -> str:
    issues = quality_findings(out)
    w = out.results[out.best_model]
    tail = f"Winner: {out.best_model} - selected by {out.selection_note}; fit diagnosis: {w.fit_status}."
    if not issues:
        return "Quality review: no major red flags.\n  - " + tail
    return "Quality review findings:\n  - " + "\n  - ".join(issues + [tail])


def diagnostics_text(out) -> str:
    d = out.winner_residual_diagnostics or {}
    lines = [f"Residual diagnostics for the winner ({out.best_model}) on the test rows:"]
    for k in ("n", "mean_residual", "bias_pct_of_mean_target", "hetero_spearman", "residual_skew",
              "residual_excess_kurtosis", "share_beyond_2sd", "durbin_watson"):
        if k in d:
            lines.append(f"  {k}: {_fmt(d[k], '.4g')}")
    lines += ["Findings:"] + [f"  - {f}" for f in d.get("findings", [])]
    lines.append("Charts produced: Predicted vs Actual, Residuals vs Predicted, Residual Distribution, "
                 "Q-Q Plot, CV R2 box (fold-consistent R2), Feature Importance (held-out permutation, "
                 "per original column, for the winner), Learning Curve; comparison bars, CV box across "
                 "models, training time, predicted-vs-actual overlay.")
    return "\n".join(lines)


def artifacts_text(out) -> str:
    return (f"regression_pipeline.py and .ipynb: reproduce the run by calling the same library code "
            f"as the app (utils.preprocessing.preprocess -> rebuild the winner '{out.best_model}' with its "
            f"final settings -> evaluate on identical folds -> compare with the app's test metrics -> "
            f"save). They need this project's utils/ folder importable (set REGRESSION_CREW_DIR). "
            f"best_model.joblib: a raw-row feature builder + the fitted model pipeline "
            f"({out.final_model_note or 'trained on the training rows'}); predict with "
            f"utils.modeling.predict_with_interval(bundle, raw_rows).")


def refinement_text(out) -> str:
    if not out.refinement_log:
        return "No refinement loop was run."
    lines = ["Refinement loop (each proposal re-ran the pipeline and was kept only if CV improved):"]
    for e in out.refinement_log:
        lines.append(f"  - round {e['round']} [{e['source']}] {e['actions']}: "
                     f"{'ACCEPTED' if e['accepted'] else 'rejected'} - {e['reason']}")
    return "\n".join(lines)


# ---- CrewAI wrappers --------------------------------------------------------------------

def build_tools(out) -> dict[str, Any]:
    """Tools bound to ONE run's output (closures - nothing shared between users)."""
    from crewai.tools import tool

    @tool("Profile dataset")
    def profile_dataset_tool(_: str = "") -> str:
        """Shape, column kinds, missingness, target statistics and structural findings."""
        return profile_text(out)

    @tool("Preprocess dataset")
    def preprocess_dataset_tool(_: str = "") -> str:
        """What preprocessing did: split, encodings, target transform and why, dropped columns."""
        return preprocess_text(out)

    @tool("Train and evaluate models")
    def train_models_tool(_: str = "") -> str:
        """Every model's cross-validated score on shared folds, fit diagnosis and the winner."""
        return models_text(out)

    @tool("Get best model summary")
    def best_model_tool(_: str = "") -> str:
        """Detailed metrics, fit diagnosis, interval and settings of the winning model."""
        return best_model_text(out)

    @tool("Quality review of pipeline")
    def quality_review_tool(_: str = "") -> str:
        """Leakage, overlap, over/underfitting, low-signal and interval-coverage audit."""
        return quality_review_text(out)

    @tool("Residual diagnostics")
    def diagnostics_tool(_: str = "") -> str:
        """Numeric summaries of the winner's residual charts, with findings."""
        return diagnostics_text(out)

    @tool("Generated artifacts")
    def artifacts_tool(_: str = "") -> str:
        """What the generated script, notebook and model bundle contain."""
        return artifacts_text(out)

    @tool("Refinement log")
    def refinement_tool(_: str = "") -> str:
        """Which improvements the advisor tried and whether cross-validation kept them."""
        return refinement_text(out)

    return {"profile": profile_dataset_tool, "preprocess": preprocess_dataset_tool,
            "train": train_models_tool, "best": best_model_tool, "quality": quality_review_tool,
            "diagnostics": diagnostics_tool, "artifacts": artifacts_tool, "refinement": refinement_tool}
