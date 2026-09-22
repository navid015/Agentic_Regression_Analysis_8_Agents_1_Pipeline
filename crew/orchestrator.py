"""
Orchestrator — runs the full pipeline. Multi-file aware.

Two phases:
  1. Deterministic ML (always runs, no LLM needed).
  2. Optional CrewAI agent narration (runs if an LLM key is configured).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

import joblib

from utils.code_generator import generate_notebook, generate_python_script
from utils.modeling import (
    BASELINE_MODEL_NAME,
    ModelResult,
    compute_permutation_importance,
    get_default_model_zoo,
    pick_best,
    rank_models,
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
    selection_basis: str = "test"


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
    log_transform_target: bool = False,
    auto_datetime_features: bool = True,
    drop_low_variance: bool = True,
    auto_drop_id_columns: bool = True,
    drop_duplicate_rows: bool = True,
    add_missing_indicators: bool = False,
    compute_permutation_importances: bool = True,
    tune_hyperparameters: bool = False,
    use_agents: bool = False,
    progress_callback=None,
    csv_filename: str = "your_dataset.csv",
    val_csv_filename: str | None = None,
    test_csv_filename: str | None = None,
) -> PipelineOutput:
    # populate the shared crew state
    crew_tools.STATE.update({
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "target": target, "test_size": test_size, "random_state": random_state,
        "selected_models": selected_models, "cv_folds": cv_folds,
        "cv_strategy": cv_strategy, "split_strategy": split_strategy,
        "time_column": time_column, "group_column": group_column,
        "log_transform_target": log_transform_target,
        "auto_datetime_features": auto_datetime_features,
        "drop_low_variance": drop_low_variance,
        "tune_hyperparameters": tune_hyperparameters,
    })

    if progress_callback: progress_callback("profile", "Profiling dataset...")
    profile = profile_dataframe(df_train, target)

    if progress_callback: progress_callback("preprocess", "Preprocessing...")
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
    )

    zoo = get_default_model_zoo(random_state=random_state)
    if selected_models:
        zoo = {k: v for k, v in zoo.items() if k in selected_models}

    def _model_progress(idx, total, name):
        if progress_callback:
            progress_callback("train", f"Training {name} ({idx}/{total})...")

    results = train_and_evaluate(
        pre.X_train, pre.X_test, pre.y_train, pre.y_test,
        models=zoo, cv_folds=cv_folds, cv_strategy=cv_strategy,
        groups_train=pre.groups_train, target_transform=pre.target_transform,
        tune_hyperparameters=tune_hyperparameters,
        progress_callback=_model_progress,
        X_val=pre.X_val, y_val=pre.y_val, random_state=random_state,
    )
    ranking = rank_models(results)
    best = pick_best(results)
    basis = selection_basis(results)

    # Permutation importance on held-out data, for the winner only (it is the
    # expensive part). Unlike impurity importance this is unbiased w.r.t.
    # cardinality and works even for KNN / SVR, which expose nothing native.
    if compute_permutation_importances and best in results:
        if progress_callback:
            progress_callback("importance", "Computing permutation importance...")
        imp, imp_std = compute_permutation_importance(
            results[best].estimator, pre.X_test, pre.y_test,
            random_state=random_state,
        )
        results[best].permutation_importances = imp
        results[best].permutation_importances_std = imp_std

    if progress_callback: progress_callback("code", "Generating reproducible code...")
    common_kwargs = dict(
        target=target, model_names=list(results.keys()),
        test_size=test_size, random_state=random_state,
        cv_folds=cv_folds, cv_strategy=cv_strategy,
        csv_path=csv_filename,
        val_csv_path=val_csv_filename,
        test_csv_path=test_csv_filename,
        log_transform_target=log_transform_target,
        auto_datetime_features=auto_datetime_features,
        drop_low_variance=drop_low_variance,
        auto_drop_id_columns=auto_drop_id_columns,
        time_column=time_column, group_column=group_column,
        summary=pre.summary,
    )
    script = generate_python_script(**common_kwargs)
    nb     = generate_notebook(**common_kwargs)

    # Persist preprocessor + winning model as ONE loadable bundle.
    # The README advertised best_model.joblib as a generated artifact, but only
    # the generated script ever wrote one - the app itself did not. Because the
    # frequency lookups now live inside the ColumnTransformer, this bundle is
    # genuinely self-contained: load it and call transform + predict on raw data.
    bundle_path: str | None = None
    try:
        out_dir = Path(tempfile.gettempdir()) / "regression_crew_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = str(out_dir / "best_model.joblib")
        joblib.dump(
            {
                "preprocessor": pre.preprocessor,
                "model": results[best].estimator,
                "model_name": best,
                "target": target,
                "target_transform": pre.target_transform,
                "feature_names": pre.feature_names,
                "selection_basis": basis,
                "metrics": results[best].metrics,
                "sklearn_version": sklearn.__version__,
            },
            bundle_path,
        )
    except Exception:
        bundle_path = None

    crew_tools.STATE["profile"] = profile
    crew_tools.STATE["preprocessing"] = pre
    crew_tools.STATE["results"] = results

    out = PipelineOutput(
        profile=profile, preprocessing=pre, results=results,
        best_model=best, ranking=ranking,
        generated_script=script, generated_notebook=nb,
        options_used={
            "target": target,
            "n_models_trained": len(results),
            "cv_folds": cv_folds, "cv_strategy": cv_strategy,
            "split_strategy": split_strategy,
            "log_transform_target": pre.target_transform == "log1p",
            "tune_hyperparameters": tune_hyperparameters,
            "test_size": test_size, "random_state": random_state,
            "multi_file": df_test is not None,
            "selection_basis": basis,
            "drop_duplicate_rows": drop_duplicate_rows,
            "add_missing_indicators": add_missing_indicators,
        },
        model_bundle_path=bundle_path,
        selection_basis=basis,
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
