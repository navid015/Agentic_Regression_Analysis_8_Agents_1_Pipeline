"""
Orchestrator — runs the full pipeline. Multi-file aware.

Two phases:
  1. Deterministic ML (always runs, no LLM needed).
  2. Optional CrewAI agent narration (runs if an LLM key is configured).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from utils.code_generator import generate_notebook, generate_python_script
from utils.modeling import (
    ModelResult,
    get_default_model_zoo,
    pick_best,
    rank_models,
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
    )
    ranking = rank_models(results)
    best = pick_best(results)

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
        },
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
