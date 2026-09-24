"""
Orchestrator — runs the full pipeline. Multi-file aware.

Two phases:
  1. Deterministic ML (always runs, no LLM needed): profile -> preprocess ->
     train/tune -> diagnose over/underfitting -> remediate -> ensemble ->
     select -> winner extras (importance, learning curve, prediction
     interval) -> reproducible code + model bundle.
  2. Optional CrewAI agent narration (runs if an LLM key is configured).
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
from sklearn.compose import TransformedTargetRegressor

from utils.code_generator import generate_notebook, generate_python_script
from utils.modeling import (
    BASELINE_MODEL_NAME,
    ENSEMBLE_NAME,
    ModelResult,
    _make_cv,
    compute_learning_curve,
    compute_permutation_importance,
    compute_prediction_interval,
    explain_selection,
    filter_zoo_for_data,
    get_default_model_zoo,
    rank_models,
    resolve_selection_metric,
    selection_basis,
    train_and_evaluate,
)
from utils.preprocessing import PreprocessingResult, preprocess, profile_dataframe

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


def _build_llm():
    try:
        from crewai import LLM
    except Exception:
        return None
    if os.getenv("OPENAI_API_KEY"):
        return LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLM(
            model=os.getenv("ANTHROPIC_MODEL", "anthropic/claude-3-5-sonnet-20241022"),
            temperature=0.2,
        )
    return None


def _plain_params(est) -> dict[str, Any]:
    """JSON/Python-literal-safe hyperparameters of a (possibly wrapped) model."""
    inner = est.regressor if isinstance(est, TransformedTargetRegressor) else est
    out = {}
    for k, v in inner.get_params(deep=False).items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
    return out


def _winner_spec(name: str, results: dict[str, ModelResult]) -> dict[str, Any]:
    r = results[name]
    if name == ENSEMBLE_NAME and r.ensemble_members:
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
    log_transform_target: bool | str = "auto",   # False | True | "auto"
    auto_datetime_features: bool = True,
    drop_low_variance: bool = True,
    auto_drop_id_columns: bool = True,
    drop_duplicate_rows: bool = True,
    add_missing_indicators: bool = False,
    add_interactions: bool = False,
    compute_permutation_importances: bool = True,
    tune_hyperparameters: bool = False,
    tuning: str | None = None,             # "off" | "fast" | "thorough"
    nested_cv: bool = False,
    auto_remediate: bool = True,
    build_ensemble: bool = True,
    one_se_rule: bool = True,
    selection_metric: str = "auto",        # "auto" | "rmse" | "mae"
    compute_learning_curves: bool = True,
    compute_intervals: bool = True,
    interval_coverage: float = 0.90,
    use_agents: bool = False,
    progress_callback=None,
    csv_filename: str = "your_dataset.csv",
    val_csv_filename: str | None = None,
    test_csv_filename: str | None = None,
) -> PipelineOutput:
    tuning_mode = _normalise_tuning(tuning, tune_hyperparameters)
    crew_tools.reset_state()
    crew_tools.STATE.update({
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "target": target, "test_size": test_size, "random_state": random_state,
        "selected_models": selected_models, "cv_folds": cv_folds,
        "cv_strategy": cv_strategy, "split_strategy": split_strategy,
        "time_column": time_column, "group_column": group_column,
        "log_transform_target": log_transform_target,
        "auto_datetime_features": auto_datetime_features,
        "drop_low_variance": drop_low_variance,
        "tune_hyperparameters": tuning_mode != "off",
        "tuning": tuning_mode,
    })

    def say(stage, msg):
        if progress_callback:
            progress_callback(stage, msg)

    say("profile", "Profiling dataset...")
    profile = profile_dataframe(df_train, target)

    say("preprocess", "Preprocessing...")
    pre = preprocess(
        df_train, target,
        df_val=df_val, df_test=df_test,
        test_size=test_size, random_state=random_state,
        split_strategy=split_strategy, time_column=time_column,
        group_column=group_column, log_transform_target=log_transform_target,
        auto_datetime_features=auto_datetime_features,
        drop_low_variance=drop_low_variance,
        auto_drop_id_columns=auto_drop_id_columns,
        drop_duplicate_rows=drop_duplicate_rows,
        add_missing_indicators=add_missing_indicators,
        add_interactions=add_interactions,
    )
    inv = np.expm1 if pre.target_transform == "log1p" else (lambda a: a)

    zoo = get_default_model_zoo(random_state=random_state)
    if selected_models:
        zoo = {k: v for k, v in zoo.items() if k in selected_models}
    zoo, skipped = filter_zoo_for_data(zoo, inv(pre.y_train),
                                       target_transform=pre.target_transform)
    if not zoo:
        raise ValueError("None of the selected models can be used on this dataset: "
                         + "; ".join(f"{k}: {v}" for k, v in skipped.items()))

    # judged on the target the models actually learn (after any log transform):
    # a skewed target that log1p has tamed is not an outlier problem
    metric, metric_reason = resolve_selection_metric(selection_metric, pre.y_train)

    def _model_progress(idx, total, name):
        say("train", f"Training {name} ({idx}/{total})...")

    failures: dict[str, str] = {}
    results = train_and_evaluate(
        pre.X_train, pre.X_test, pre.y_train, pre.y_test,
        models=zoo, cv_folds=cv_folds, cv_strategy=cv_strategy,
        groups_train=pre.groups_train, target_transform=pre.target_transform,
        tuning=tuning_mode, nested_cv=nested_cv, auto_remediate=auto_remediate,
        build_ensemble=build_ensemble, progress_callback=_model_progress,
        X_val=pre.X_val, y_val=pre.y_val, random_state=random_state,
        failures=failures, selection_metric=metric,
    )
    if not results:
        raise RuntimeError("Every model failed to train: " +
                           "; ".join(f"{k}: {v}" for k, v in failures.items()))
    ranking = rank_models(results)
    best, note = explain_selection(results, one_se_rule=one_se_rule, selection_metric=metric)
    basis = selection_basis(results)
    winner = results[best]

    if compute_permutation_importances:
        say("importance", "Computing permutation importance...")
        imp, imp_std = compute_permutation_importance(winner.estimator, pre.X_test, pre.y_test,
                                                      random_state=random_state)
        winner.permutation_importances, winner.permutation_importances_std = imp, imp_std

    if compute_learning_curves:
        say("learning_curve", "Computing learning curve for the winner...")
        winner.learning_curve = compute_learning_curve(
            winner.estimator, pre.X_train, pre.y_train, cv_strategy=cv_strategy,
            cv_folds=cv_folds, groups=pre.groups_train,
            target_transform=pre.target_transform, random_state=random_state)

    if compute_intervals:
        say("interval", "Computing prediction interval...")
        winner.prediction_interval = compute_prediction_interval(
            winner.estimator, pre.X_train, pre.y_train, cv_strategy=cv_strategy,
            cv_folds=cv_folds, groups=pre.groups_train,
            target_transform=pre.target_transform, random_state=random_state,
            coverage=interval_coverage, X_test=pre.X_test, y_test=pre.y_test)

    say("code", "Generating reproducible code...")
    cv_obj, _ = _make_cv(cv_strategy, cv_folds, pre.groups_train, random_state,
                         n_samples=len(pre.y_train))
    common_kwargs = dict(
        target=target, model_names=[n for n in results if n != ENSEMBLE_NAME],
        test_size=test_size, random_state=random_state,
        cv_folds=cv_folds, cv_strategy=cv_strategy,
        csv_path=csv_filename, val_csv_path=val_csv_filename,
        test_csv_path=test_csv_filename,
        log_transform_target=pre.target_transform == "log1p",
        auto_datetime_features=auto_datetime_features,
        drop_low_variance=drop_low_variance,
        auto_drop_id_columns=auto_drop_id_columns,
        drop_duplicate_rows=drop_duplicate_rows,
        # the script only time-splits when the app did
        time_column=time_column if split_strategy == "time" else None,
        group_column=group_column, summary=pre.summary,
        add_missing_indicators=add_missing_indicators,
        add_interactions=pre.summary.get("interaction_features_added", False),
        cv_repr=repr(cv_obj), selection_metric=metric, selection_note=note,
        winner_spec=_winner_spec(best, results),
        interval=winner.prediction_interval,
    )
    script = generate_python_script(**common_kwargs)
    nb = generate_notebook(**common_kwargs)

    bundle_path: str | None = None
    try:
        out_dir = Path(tempfile.gettempdir()) / "regression_crew_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = str(out_dir / "best_model.joblib")
        joblib.dump({
            # full raw-row pipeline: date features + imputation/encoding/scaling
            "preprocessor": pre.raw_preprocessor or pre.preprocessor,
            "model": winner.estimator,
            "model_name": best,
            "target": target,
            "target_transform": pre.target_transform,
            "feature_names": pre.feature_names,
            "selection_basis": basis,
            "selection_metric": metric,
            "selection_note": note,
            "fit_status": winner.fit_status,
            "metrics": winner.metrics,
            "best_params": winner.best_params,
            "prediction_interval": winner.prediction_interval,
            "sklearn_version": sklearn.__version__,
        }, bundle_path)
    except Exception:
        bundle_path = None

    crew_tools.STATE.update({
        "profile": profile, "preprocessing": pre, "results": results,
        "best_model": best, "selection_note": note, "selection_metric": metric,
        "skipped_models": skipped, "failed_models": failures,
    })

    status_counts: dict[str, int] = {}
    for n, r in results.items():
        if n != BASELINE_MODEL_NAME:
            status_counts[r.fit_status] = status_counts.get(r.fit_status, 0) + 1

    out = PipelineOutput(
        profile=profile, preprocessing=pre, results=results,
        best_model=best, ranking=ranking,
        generated_script=script, generated_notebook=nb,
        options_used={
            "target": target,
            "n_models_trained": len(results),
            "cv_folds": cv_folds, "cv_strategy": cv_strategy,
            "cv_scheme": repr(cv_obj),
            "split_strategy": split_strategy,
            "log_transform_target": pre.target_transform == "log1p",
            "log_transform_requested": log_transform_target,
            "tuning": tuning_mode, "nested_cv": nested_cv,
            "tune_hyperparameters": tuning_mode != "off",
            "auto_remediate": auto_remediate, "build_ensemble": build_ensemble,
            "one_se_rule": one_se_rule,
            "test_size": test_size, "random_state": random_state,
            "multi_file": df_test is not None,
            "selection_basis": basis,
            "drop_duplicate_rows": drop_duplicate_rows,
            "add_missing_indicators": add_missing_indicators,
            "add_interactions": pre.summary.get("interaction_features_added", False),
            "fit_status_counts": status_counts,
        },
        model_bundle_path=bundle_path,
        selection_basis=basis, selection_note=note,
        selection_metric=metric, selection_metric_reason=metric_reason,
        skipped_models=skipped, failed_models=failures,
    )

    if use_agents:
        narrative, agent_out = _run_crew_narration(
            target=target,
            n_rows=profile["n_rows"], n_features=profile["n_cols"],
            progress_callback=progress_callback,
        )
        out.crew_narrative = narrative
        out.agent_outputs = agent_out
    return out


def _run_crew_narration(target, n_rows, n_features, progress_callback=None
                        ) -> tuple[str | None, dict[str, str]]:
    llm = _build_llm()
    if llm is None:
        return ("Agent narration skipped — no OPENAI_API_KEY or ANTHROPIC_API_KEY set. "
                "The deterministic pipeline still ran.", {})
    try:
        from crewai import Crew, Process

        from .agents import build_agents
        from .tasks import build_tasks
        if progress_callback:
            progress_callback("agents", "CrewAI agents narrating results...")
        agents = build_agents(llm)
        tasks = build_tasks(agents, target, n_features, n_rows)
        crew = Crew(
            agents=list(agents.values()), tasks=tasks,
            process=Process.sequential, verbose=False,
        )
        result = crew.kickoff()
        agent_outputs: dict[str, str] = {}
        names = ["planner", "eda", "preprocessor", "modeler",
                 "chart", "quality", "code", "insight"]
        for task, name in zip(tasks, names):
            try:
                agent_outputs[name] = str(task.output.raw) if task.output else ""
            except Exception:
                agent_outputs[name] = ""
        return str(result), agent_outputs
    except Exception as e:
        return f"Agent narration failed: {type(e).__name__}: {e}", {}
