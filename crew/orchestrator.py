"""
Orchestrator - runs the full pipeline.

Phases
  1. Deterministic ML (always runs, no LLM needed): reconcile split / CV
     settings -> profile -> preprocess -> CV-verified target transform ->
     train / tune on shared folds with in-fold preprocessing -> diagnose ->
     remediate -> diverse ensemble -> select (paired one-SE rule + "is there
     any signal?" check) -> winner extras -> final refit -> audits ->
     reproducible code + model bundle, written to a PER-RUN directory.
  2. Optional refinement loop (`auto_refine`): the advisor proposes changes
     (rules, optionally an LLM); each is tested by re-running phase 1 and kept
     only if cross-validated error improves beyond fold noise.
  3. Optional CrewAI narration of the FINAL run, with tools bound to that run.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from utils.code_generator import generate_notebook, generate_python_script
from utils.diagnostics import leakage_suspects, residual_diagnostics
from utils.modeling import (
    BASELINE_MODEL_NAME,
    ENSEMBLE_NAME,
    ModelResult,
    _compute_metrics,
    _oof_from_folds,
    _plain_params,
    _set_n_jobs,
    apply_smearing,
    compare_target_transforms,
    compute_learning_curve,
    compute_permutation_importance,
    compute_prediction_interval,
    explain_selection,
    filter_zoo_for_data,
    get_default_model_zoo,
    make_folds,
    rank_models,
    resolve_selection_metric,
    selection_basis,
    smearing_factor,
    train_and_evaluate,
)
from utils.preprocessing import PreprocessingResult, preprocess, profile_dataframe

from . import advisor as crew_advisor
from . import tools as crew_tools


@dataclass
class PipelineOutput:
    profile: dict[str, Any]
    preprocessing: PreprocessingResult
    results: dict[str, ModelResult]
    best_model: str
    ranking: list[tuple[str, float, float]]
    generated_script: str
    generated_notebook: str
    options_used: dict[str, Any] = field(default_factory=dict)
    crew_narrative: str | None = None
    agent_outputs: dict[str, str] = field(default_factory=dict)
    model_bundle_path: str | None = None
    selection_basis: str = "cv"
    selection_note: str = ""
    selection_metric: str = "rmse"
    selection_metric_reason: str = ""
    skipped_models: dict[str, str] = field(default_factory=dict)
    failed_models: dict[str, str] = field(default_factory=dict)
    # ---- v3 ----
    output_dir: str | None = None
    script_path: str | None = None
    notebook_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    leakage_suspects: list[dict] = field(default_factory=list)
    target_transform_decision: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    final_model_note: str = ""
    final_model_metrics: dict = field(default_factory=dict)
    winner_residual_diagnostics: dict = field(default_factory=dict)
    refinement_log: list[dict] = field(default_factory=list)
    pending_confirmation: list[dict] = field(default_factory=list)
    run_options: dict = field(default_factory=dict)
    run_inputs: dict = field(default_factory=dict)
    task_report: dict = field(default_factory=dict)


# ---- LLM ------------------------------------------------------------------------------

#: Claude 3.5 Sonnet (the old default) was retired in October 2025.
DEFAULT_ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def _build_llm():
    try:
        from crewai import LLM
    except Exception:
        return None
    if os.getenv("OPENAI_API_KEY"):
        return LLM(model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL), temperature=0.2)
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLM(model=os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL), temperature=0.2)
    return None


def _llm_call_fn():
    llm = _build_llm()
    if llm is None:
        return None

    def call(prompt: str) -> str:
        return str(llm.call([{"role": "user", "content": prompt}]))
    return call


# ---- helpers ----------------------------------------------------------------------------------

def _winner_spec(name: str, results: dict[str, ModelResult]) -> dict[str, Any]:
    r = results[name]
    if r.ensemble_members:
        return {"name": name, "members": [
            {"name": m, "params": _plain_params(results[m].estimator)}
            for m in r.ensemble_members if m in results]}
    return {"name": name, "params": _plain_params(r.estimator)}


def _normalise_tuning(tuning, tune_hyperparameters) -> str:
    if tuning is None:
        return "fast" if tune_hyperparameters else "off"
    t = str(tuning).lower()
    for key in ("off", "fast", "thorough"):
        if t.startswith(key):
            return key
    return "off"


def reconcile_strategies(split_strategy: str, cv_strategy: str, time_column, group_column,
                         warnings: list[str]) -> tuple[str, str]:
    """Split and CV must agree. A time-aware split validated with GroupKFold or
    shuffled K-fold trains on the future; TimeSeriesSplit on shuffled rows is
    meaningless; group CV without a group column silently becomes K-fold."""
    if split_strategy == "time":
        if not time_column:
            raise ValueError("A time-aware split needs a time column.")
        if cv_strategy != "time":
            warnings.append(f"CV strategy '{cv_strategy}' replaced by time-series CV: a time-aware split "
                            "must also be validated forward in time (group / shuffled K-fold folds would "
                            "train on the future).")
        return "time", "time"
    if cv_strategy == "time":
        if time_column:
            warnings.append("Time-series CV requested: the train/test split is made time-aware too "
                            "(TimeSeriesSplit on randomly shuffled rows is meaningless).")
            return "time", "time"
        warnings.append("Time-series CV requested without a time column: using K-fold instead.")
        return split_strategy, "kfold"
    if cv_strategy == "group" and not group_column:
        warnings.append("Group CV requested without a group column: using K-fold instead.")
        return split_strategy, "kfold"
    if group_column and cv_strategy == "kfold":
        return split_strategy, "group"
    return split_strategy, cv_strategy


def _concat(parts):
    if isinstance(parts[0], pd.DataFrame):
        return pd.concat(parts, ignore_index=True)
    return np.concatenate(parts)


# ---- one deterministic run -------------------------------------------------------------------------

def _run_once(df_train, df_val, df_test, target: str, o: dict, progress_callback=None) -> PipelineOutput:
    def say(stage, msg):
        if progress_callback:
            progress_callback(stage, msg)

    warns: list[str] = []
    rs, n_jobs = int(o["random_state"]), int(o.get("n_jobs", -1))
    split_strategy, cv_strategy = reconcile_strategies(o["split_strategy"], o["cv_strategy"],
                                                       o.get("time_column"), o.get("group_column"), warns)
    tuning_mode = _normalise_tuning(o.get("tuning"), o.get("tune_hyperparameters"))

    say("profile", "Profiling dataset...")
    profile = profile_dataframe(df_train, target)

    pre_kwargs = dict(
        test_size=o["test_size"], random_state=rs, split_strategy=split_strategy,
        time_column=o.get("time_column"), group_column=o.get("group_column"),
        auto_datetime_features=o["auto_datetime_features"], drop_low_variance=o["drop_low_variance"],
        auto_drop_id_columns=o["auto_drop_id_columns"], drop_duplicate_rows=o["drop_duplicate_rows"],
        add_missing_indicators=o["add_missing_indicators"], add_interactions=o["add_interactions"],
        high_cardinality_encoding=o.get("high_cardinality_encoding", "target"),
        categorical_columns=list(o.get("categorical_columns") or []),
        drop_columns=list(o.get("drop_columns") or []),
    )
    say("preprocess", "Preprocessing...")
    log_req = o["log_transform_target"]
    decision: dict[str, Any] = {"requested": log_req}
    if isinstance(log_req, str) and log_req.lower() == "auto":
        pre = preprocess(df_train, target, df_val=df_val, df_test=df_test,
                         log_transform_target=False, **pre_kwargs)
        s = pre.summary
        if s["log_candidate"]:
            say("preprocess", "Checking raw vs log target by cross-validation...")
            folds0 = make_folds(cv_strategy, o["cv_folds"], pre.y_train, pre.groups_train, rs)
            m0 = o["selection_metric"] if o["selection_metric"] in ("rmse", "mae") else "rmse"
            chosen, errs = compare_target_transforms(
                pre.X_train_raw, pre.y_train_original, pre.preprocessor_templates, folds0,
                selection_metric=m0, count_like=s["target_count_like"], random_state=rs,
                time_ordered=cv_strategy == "time", n_jobs=n_jobs)
            decision.update(method="cv", chosen=chosen, errors=errs, metric=m0)
            if chosen == "log1p":
                pre = preprocess(df_train, target, df_val=df_val, df_test=df_test,
                                 log_transform_target=True, **pre_kwargs)
        else:
            decision.update(method="rule", chosen="none",
                            reason="the target is not a skewed non-negative quantity")
    else:
        pre = preprocess(df_train, target, df_val=df_val, df_test=df_test,
                         log_transform_target=bool(log_req), **pre_kwargs)
        decision.update(method="user", chosen=pre.target_transform)
    s = pre.summary
    log_t = pre.target_transform
    inv = np.expm1 if log_t == "log1p" else (lambda a: a)

    zoo = get_default_model_zoo(rs, time_ordered=cv_strategy == "time", target_transform=log_t)
    if o.get("selected_models"):
        zoo = {k: v for k, v in zoo.items()
               if k == BASELINE_MODEL_NAME or k in o["selected_models"]}
    zoo, skipped = filter_zoo_for_data(zoo, pre.y_train_original, target_transform=log_t)
    if not zoo:
        raise ValueError("None of the selected models can be used on this dataset: "
                         + "; ".join(f"{k}: {v}" for k, v in skipped.items()))
    metric, metric_reason = resolve_selection_metric(o["selection_metric"], pre.y_train)

    def _model_progress(idx, total, name):
        say("train", f"Training {name} ({idx}/{total})...")

    failures: dict[str, str] = {}
    diag: dict[str, Any] = {}
    results = train_and_evaluate(
        pre.X_train_raw, pre.X_test_raw, pre.y_train, pre.y_test,
        models=zoo, cv_folds=o["cv_folds"], cv_strategy=cv_strategy, groups_train=pre.groups_train,
        target_transform=log_t, tuning=tuning_mode, nested_cv=o["nested_cv"],
        auto_remediate=o["auto_remediate"], build_ensemble=o["build_ensemble"],
        progress_callback=_model_progress, X_val=pre.X_val_raw, y_val=pre.y_val, random_state=rs,
        failures=failures, selection_metric=metric, preprocessor=pre.preprocessor_templates,
        n_jobs=n_jobs, diagnostics=diag)
    if not results:
        raise RuntimeError("Every model failed to train: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))
    folds = diag["folds"]
    best, note = explain_selection(results, one_se_rule=o["one_se_rule"], selection_metric=metric,
                                   no_signal_check=o.get("allow_baseline_winner", True))
    basis = selection_basis(results)
    winner = results[best]
    y_test_orig = pre.y_test_original

    # ---- log-target bias correction, kept only if it helps OUT-OF-FOLD ----
    if log_t == "log1p" and metric == "rmse" and best != BASELINE_MODEL_NAME:
        S = smearing_factor(pre.y_train, winner.fold_test_pred, folds)
        if S:
            oof, mask = _oof_from_folds(len(pre.y_train), folds["splits"], winner.fold_test_pred)
            yo = pre.y_train_original[mask]
            rmse_plain = float(np.sqrt(np.mean((yo - np.expm1(oof[mask])) ** 2)))
            rmse_corr = float(np.sqrt(np.mean((yo - apply_smearing(oof[mask], S)) ** 2)))
            if rmse_corr < rmse_plain:
                winner.smearing_factor = S
                winner.metrics["RMSE_uncorrected"] = winner.metrics["RMSE"]
                winner.metrics["MAE_uncorrected"] = winner.metrics["MAE"]
                winner.y_pred_test = apply_smearing(winner.y_pred_test_model, S)
                winner.metrics.update(_compute_metrics(y_test_orig, winner.y_pred_test))
                winner.metrics["smearing_factor"] = S

    if o["compute_permutation_importances"]:
        say("importance", "Computing permutation importance per original column...")
        imp, imp_std = compute_permutation_importance(winner.estimator, pre.X_test_raw, pre.y_test,
                                                      random_state=rs, target_transform=log_t,
                                                      selection_metric=metric)
        winner.permutation_importances, winner.permutation_importances_std = imp, imp_std
        winner.permutation_feature_names = list(pre.X_test_raw.columns)

    if o["compute_learning_curves"]:
        say("learning_curve", "Computing learning curve for the winner...")
        winner.learning_curve = compute_learning_curve(winner.spec_estimator, pre.X_train_raw, pre.y_train,
                                                       target_transform=log_t, folds=folds)

    if o["compute_intervals"]:
        say("interval", "Computing prediction interval...")
        winner.prediction_interval = compute_prediction_interval(
            winner.estimator, pre.X_train_raw, pre.y_train, target_transform=log_t, random_state=rs,
            coverage=o["interval_coverage"], X_test=pre.X_test_raw, y_test=pre.y_test,
            fold_test_pred=winner.fold_test_pred, folds=folds,
            adaptive=o.get("adaptive_intervals", True),
            sigma_preprocessor=pre.preprocessor_templates["default"])

    # ---- the model that is SAVED ----
    mode = o.get("final_refit", "auto")
    if mode == "auto":
        mode = "train+val" if pre.X_val_raw is not None else "train"
    final_est, final_note, final_metrics = winner.estimator, "fitted on the training rows", {}
    try:
        if mode == "train+val" and pre.X_val_raw is not None:
            final_est = _set_n_jobs(clone(winner.spec_estimator), -1).fit(
                _concat([pre.X_train_raw, pre.X_val_raw]), _concat([pre.y_train, pre.y_val]))
            p = np.asarray(final_est.predict(pre.X_test_raw), float)
            p = apply_smearing(p, winner.smearing_factor) if log_t == "log1p" else p
            final_metrics = _compute_metrics(y_test_orig, p)
            final_note = (f"refitted on training + validation rows (after selection); its test RMSE is "
                          f"{final_metrics['RMSE']:.4g}")
        elif mode == "all":
            Xs = [pre.X_train_raw] + ([pre.X_val_raw] if pre.X_val_raw is not None else []) + [pre.X_test_raw]
            ys = [pre.y_train] + ([pre.y_val] if pre.y_val is not None else []) + [pre.y_test]
            final_est = _set_n_jobs(clone(winner.spec_estimator), -1).fit(_concat(Xs), _concat(ys))
            final_note = ("refitted on ALL labelled rows (train + validation + test) for deployment; "
                          "the reported test scores describe the train-only fit")
    except Exception as e:
        final_est, final_note = winner.estimator, f"refit failed ({type(e).__name__}); fitted on training rows"

    say("audit", "Auditing for leakage...")
    suspects = leakage_suspects(pre.X_train_raw, pre.y_train_original, target,
                                exclude=s.get("time_features_from_split_column") or [], random_state=rs)
    resid = residual_diagnostics(y_test_orig, winner.y_pred_test, time_ordered=cv_strategy == "time")
    if s.get("time_order_ok") is False:
        warns.append("The validation/test period is not strictly after the training period.")

    # ---- reproducible code + bundle, in a directory of THIS run ----
    say("code", "Generating reproducible code...")
    out_dir = o.get("output_dir") or tempfile.mkdtemp(prefix="regression_crew_")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    interval_literal = {k: v for k, v in (winner.prediction_interval or {}).items()
                        if k in ("coverage_target", "halfwidth", "space", "method")}
    gen_kwargs = dict(
        target=target, model_names=[n for n in results if n != ENSEMBLE_NAME],
        test_size=o["test_size"], random_state=rs, cv_folds=o["cv_folds"], cv_strategy=cv_strategy,
        csv_path=o.get("csv_filename") or "train.csv", val_csv_path=o.get("val_csv_filename"),
        test_csv_path=o.get("test_csv_filename"), summary=s, cv_repr=folds["repr"],
        selection_metric=metric, selection_note=note, winner_spec=_winner_spec(best, results),
        interval=interval_literal, smearing_factor=winner.smearing_factor,
        app_test_metrics=winner.metrics, split_strategy=split_strategy,
        preprocess_options={**pre_kwargs, "log_transform_target": log_t == "log1p"},
        task_spec=o.get("task_spec"), reserve_final_test=o.get("reserve_final_test", False),
    )
    script, nb = generate_python_script(**gen_kwargs), generate_notebook(**gen_kwargs)
    script_path, nb_path = Path(out_dir) / "regression_pipeline.py", Path(out_dir) / "regression_pipeline.ipynb"
    script_path.write_text(script)
    nb_path.write_text(nb)
    bundle_path: str | None = str(Path(out_dir) / "best_model.joblib")
    try:
        joblib.dump({
            "format_version": 3,
            "preprocessor": pre.raw_feature_builder,   # raw rows -> engineered frame
            "model": final_est,                        # full pipeline (encoding + model)
            "model_name": best, "target": target, "target_transform": log_t,
            "smearing_factor": winner.smearing_factor,
            "raw_feature_columns": list(pre.X_train_raw.columns),
            "task_spec": o.get("task_spec"),
            "task_input_columns": o.get("task_input_columns"),
            "feature_names": pre.feature_names, "selection_basis": basis, "selection_metric": metric,
            "selection_note": note, "fit_status": winner.fit_status, "metrics": winner.metrics,
            "best_params": winner.best_params, "prediction_interval": winner.prediction_interval,
            "trained_on": final_note, "sklearn_version": sklearn.__version__,
        }, bundle_path)
    except Exception:
        bundle_path = None

    status_counts: dict[str, int] = {}
    for n, r in results.items():
        if n != BASELINE_MODEL_NAME:
            status_counts[r.fit_status] = status_counts.get(r.fit_status, 0) + 1

    return PipelineOutput(
        profile=profile, preprocessing=pre, results=results, best_model=best,
        ranking=rank_models(results), generated_script=script, generated_notebook=nb,
        options_used={
            "target": target, "n_models_trained": len(results), "cv_folds": o["cv_folds"],
            "cv_strategy": cv_strategy, "cv_scheme": folds["repr"], "split_strategy": split_strategy,
            "log_transform_target": log_t == "log1p", "log_transform_requested": log_req,
            "tuning": tuning_mode, "nested_cv": o["nested_cv"], "tune_hyperparameters": tuning_mode != "off",
            "auto_remediate": o["auto_remediate"], "build_ensemble": o["build_ensemble"],
            "one_se_rule": o["one_se_rule"], "test_size": o["test_size"], "random_state": rs,
            "multi_file": df_test is not None, "selection_basis": basis,
            "drop_duplicate_rows": o["drop_duplicate_rows"],
            "add_missing_indicators": o["add_missing_indicators"],
            "add_interactions": s.get("interaction_features_added", False),
            "high_cardinality_encoding": o.get("high_cardinality_encoding", "target"),
            "final_refit": mode, "fit_status_counts": status_counts,
        },
        model_bundle_path=bundle_path, selection_basis=basis, selection_note=note,
        selection_metric=metric, selection_metric_reason=metric_reason,
        skipped_models=skipped, failed_models=failures,
        output_dir=out_dir, script_path=str(script_path), notebook_path=str(nb_path),
        warnings=warns, leakage_suspects=suspects, target_transform_decision=decision,
        diagnostics={k: v for k, v in diag.items() if k != "folds"} | {"folds": folds},
        final_model_note=final_note, final_model_metrics=final_metrics,
        winner_residual_diagnostics=resid, run_options=dict(o),
        run_inputs={"df_train": df_train, "df_val": df_val, "df_test": df_test, "target": target},
    )


# ---- public entry point --------------------------------------------------------------------------

def _experiment_context(out: PipelineOutput) -> str:
    """Only training profile and CV scores enter the experiment planner prompt."""
    metric = out.selection_metric.upper()
    rows = sorted(
        ((name, result.metrics.get(f"CV_{metric}_mean")) for name, result in out.results.items()),
        key=lambda row: float("inf") if row[1] is None else row[1],
    )
    scores = ", ".join(f"{name}: {score:.4g}" for name, score in rows
                       if score is not None and np.isfinite(score))
    history = out.refinement_log[-10:] if out.refinement_log else []
    return (f"Training data: {out.profile['n_rows']} rows, {out.profile['n_cols']} columns; "
            f"CV metric: {metric}; CV scores: {scores}. "
            f"Missing cells: {out.profile.get('missing_total', 0)}. "
            f"Previous experiments: {history}")


def run_full_pipeline(
    *,
    df_train: pd.DataFrame,
    target: str,
    df_val: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    selected_models: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    cv_strategy: str = "kfold",            # "kfold" | "time" | "group"
    split_strategy: str = "random",        # "random" | "time"
    time_column: str | None = None,
    group_column: str | None = None,
    log_transform_target: bool | str = "auto",
    auto_datetime_features: bool = True,
    drop_low_variance: bool = True,
    auto_drop_id_columns: bool = True,
    drop_duplicate_rows: bool = True,
    add_missing_indicators: bool = True,
    add_interactions: bool = False,
    compute_permutation_importances: bool = True,
    tune_hyperparameters: bool = False,
    tuning: str | None = None,
    nested_cv: bool = False,
    auto_remediate: bool = True,
    build_ensemble: bool = True,
    one_se_rule: bool = True,
    selection_metric: str = "auto",
    compute_learning_curves: bool = True,
    compute_intervals: bool = True,
    interval_coverage: float = 0.90,
    use_agents: bool = False,
    progress_callback=None,
    csv_filename: str = "your_dataset.csv",
    val_csv_filename: str | None = None,
    test_csv_filename: str | None = None,
    # ---- v3 ----
    high_cardinality_encoding: str = "target",
    categorical_columns: list[str] | None = None,
    drop_columns: list[str] | None = None,
    adaptive_intervals: bool = True,
    final_refit: str = "auto",             # "auto" | "train" | "train+val" | "all"
    allow_baseline_winner: bool = True,
    auto_refine: bool = False,
    refine_rounds: int = 1,
    use_llm_advisor: bool = False,
    eight_agent_mode: bool = False,
    regression_task=None,
    output_dir: str | None = None,
    n_jobs: int = -1,
) -> PipelineOutput:
    _args = dict(locals())
    options = {k: v for k, v in _args.items()
               if k not in ("df_train", "df_val", "df_test", "target", "progress_callback",
                            "use_agents", "auto_refine", "refine_rounds", "use_llm_advisor", "eight_agent_mode", "regression_task") }

    def run(opts):
        return _run_once(df_train, df_val, df_test, target, opts, progress_callback)

    if eight_agent_mode:
        from .laboratory import run_eight_agent_pipeline
        out = run_eight_agent_pipeline(df_train=df_train, df_test=df_test, df_val=df_val,
                                       target=target, task=regression_task, progress_callback=progress_callback,
                                       **options)
        if use_agents:
            out.crew_narrative, out.agent_outputs = _run_crew_narration(out, progress_callback)
        return out

    out = run(options)
    if auto_refine:
        llm_call = _llm_call_fn() if use_llm_advisor else None
        out = crew_advisor.refine(run, out, rounds=refine_rounds, llm_call=llm_call,
                                  context_fn=_experiment_context,
                                  progress=progress_callback)
    else:
        out.pending_confirmation = [p.as_dict() for p in crew_advisor.rule_based_proposals(out)
                                    if not p.auto_ok]
    if use_agents:
        out.crew_narrative, out.agent_outputs = _run_crew_narration(out, progress_callback)
    return out


def _run_crew_narration(out: PipelineOutput, progress_callback=None) -> tuple[str | None, dict[str, str]]:
    llm = _build_llm()
    if llm is None:
        return ("Agent narration skipped - no OPENAI_API_KEY or ANTHROPIC_API_KEY set. "
                "The deterministic pipeline still ran.", {})
    try:
        from crewai import Crew, Process

        from .agents import build_agents
        from .tasks import build_tasks
        if progress_callback:
            progress_callback("agents", "CrewAI agents narrating results...")
        agents = build_agents(llm, crew_tools.build_tools(out))
        tasks = build_tasks(agents, out.options_used["target"], out.profile["n_cols"], out.profile["n_rows"])
        result = Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential,
                      verbose=False).kickoff()
        names = ["planner", "eda", "preprocessor", "modeler", "chart", "quality", "code", "insight"]
        agent_outputs = {}
        for task, name in zip(tasks, names):
            try:
                agent_outputs[name] = str(task.output.raw) if task.output else ""
            except Exception:
                agent_outputs[name] = ""
        return str(result), agent_outputs
    except Exception as e:
        return f"Agent narration failed: {type(e).__name__}: {e}", {}
