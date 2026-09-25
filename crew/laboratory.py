"""Eight independent hypothesis agents, one locked regression evaluator.

Only structured, validated proposals reach the runner. Agents see training
summaries and CV scores, never the reserved final-test frame or its metrics.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, asdict
from typing import Callable

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .advisor import Proposal, _parse_llm_actions, _improves, apply_proposals
from utils.modeling import available_model_names
from utils.task_contract import RegressionTask, prepare_task_frames, assess_target, shift_report, subgroup_error_report


@dataclass(frozen=True)
class Specialist:
    name: str
    role: str
    scope: tuple[str, ...]
    instruction: str


SPECIALISTS = (
    Specialist("data", "Data Scientist", ("treat_as_categorical", "add_missing_indicators"),
               "Investigate feature types and missingness."),
    Specialist("validation", "Validation Scientist", ("tuning",),
               "Audit the locked validation design. Never change splits; suggest tuning only if justified."),
    Specialist("features", "Feature Engineer", ("add_interactions", "log_transform_target"),
               "Test a concrete feature or target representation hypothesis."),
    Specialist("linear", "Linear and GLM Scientist", ("selected_models",),
               "Consider only LinearRegression, Ridge, Lasso, ElasticNet, Huber, PoissonRegressor, GammaRegressor, TweedieRegressor."),
    Specialist("trees", "Tree Scientist", ("selected_models",),
               "Consider only DecisionTree, RandomForest, ExtraTrees."),
    Specialist("boosting", "Boosting Scientist", ("selected_models",),
               "Consider only GradientBoosting, HistGradientBoosting, XGBoost, LightGBM."),
    Specialist("ensemble", "Ensemble Scientist", ("build_ensemble",),
               "Test whether the existing diverse ensemble helps."),
    Specialist("critic", "Regression Critic", ("high_cardinality_encoding", "tuning"),
               "Challenge the leading configuration and test one robustness alternative."),
)
FAMILIES = {
    "linear": {"LinearRegression", "Ridge", "Lasso", "ElasticNet", "Huber", "PoissonRegressor", "GammaRegressor", "TweedieRegressor"},
    "trees": {"DecisionTree", "RandomForest", "ExtraTrees"},
    "boosting": {"GradientBoosting", "HistGradientBoosting", "XGBoost", "LightGBM"},
}


def reserve_test(df: pd.DataFrame, *, test_size: float, seed: int, split: str,
                 time_column: str | None, group_column: str | None):
    """Reserve untouched rows before creating any agent or fitting an experiment."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5 in eight-agent mode")
    if split == "time":
        if not time_column or time_column not in df:
            raise ValueError("A valid time column is required for time-ordered experiments")
        ordered = df.sort_values(time_column, kind="stable")
        cut = int(len(ordered) * (1 - test_size))
        while 0 < cut < len(ordered) and ordered[time_column].iloc[cut] == ordered[time_column].iloc[cut - 1]:
            cut += 1
        if cut >= len(ordered):
            raise ValueError("Cannot reserve a later test period: timestamps overlap the boundary")
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    elif group_column:
        if group_column not in df:
            raise ValueError("Group column is missing")
        indices = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                       .split(df, groups=df[group_column]))
        train, test = df.iloc[indices[0]], df.iloc[indices[1]]
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=seed)
    if len(train) < 30 or len(test) < 5:
        raise ValueError("Eight-agent mode needs at least 30 development rows and 5 test rows")
    return train.reset_index(drop=True).copy(), test.reset_index(drop=True).copy()


def valid_proposal(raw: str, specialist: Specialist, columns: set[str]) -> Proposal | None:
    actions = _parse_llm_actions(raw, columns)
    if len(actions) != 1 or actions[0].action not in specialist.scope:
        return None
    proposal = actions[0]
    if specialist.name in FAMILIES and proposal.action == "selected_models":
        if not set(proposal.value) <= FAMILIES[specialist.name]:
            return None
    if proposal.action == "treat_as_categorical" and len(proposal.value) > 5:
        return None
    return proposal if proposal.auto_ok else None


def crew_proposer(specialist: Specialist, prompt: str, llm) -> str:
    from crewai import Agent, Crew, Process, Task
    agent = Agent(role=specialist.role, goal="Propose one testable regression experiment",
                  backstory=specialist.instruction, llm=llm, allow_delegation=False,
                  max_iter=2, verbose=False)
    task = Task(description=prompt, expected_output='One JSON object with an "actions" array', agent=agent)
    result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    return str(result)


def run_eight_agent_pipeline(*, df_train: pd.DataFrame, target: str,
                             df_test: pd.DataFrame | None = None,
                             df_val: pd.DataFrame | None = None,
                             proposer: Callable[[Specialist, str], str] | None = None,
                             task: RegressionTask | None = None,
                             progress_callback=None, **options):
    """Run eight specialist proposals, then evaluate the winner once on final test.

    `proposer` is injectable for offline testing. Without it, eight distinct
    CrewAI agents are instantiated sequentially with the configured LLM.
    """
    if df_val is not None:
        raise ValueError("Eight-agent mode does not use a separate validation file; combine it with training only if appropriate")
    from .orchestrator import _build_llm, _run_once, _experiment_context, reconcile_strategies
    llm = None if proposer is not None else _build_llm()
    if proposer is None and llm is None:
        raise ValueError("Eight-agent mode requires OPENAI_API_KEY or ANTHROPIC_API_KEY")
    if target not in df_train or (df_test is not None and target not in df_test):
        raise ValueError("Target must exist in training and final-test files")
    opts = dict(options)
    task = task or RegressionTask()
    original_feature_cols = [c for c in df_train.columns if c != target]
    if task.available_features is not None:
        allowed = set(task.available_features) | {c for c in (opts.get("time_column"), opts.get("group_column")) if c}
        original_feature_cols = [c for c in original_feature_cols if c in allowed]
    df_train, df_test = prepare_task_frames(task, df_train, df_test, target,
        time_column=opts.get("time_column"), group_column=opts.get("group_column"))
    opts["final_refit"] = "train"
    opts["output_dir"] = None
    split, cv = reconcile_strategies(opts["split_strategy"], opts["cv_strategy"],
                                      opts.get("time_column"), opts.get("group_column"), [])
    opts["split_strategy"], opts["cv_strategy"] = split, cv
    if df_test is None:
        dev, locked_test = reserve_test(df_train, test_size=opts["test_size"],
            seed=opts["random_state"], split=split, time_column=opts.get("time_column"),
            group_column=opts.get("group_column"))
    else:
        dev, locked_test = df_train.copy(), df_test.copy()
        if split == "time":
            col = opts["time_column"]
            if pd.to_datetime(dev[col]).max() >= pd.to_datetime(locked_test[col]).min():
                raise ValueError("Final test must be strictly later than development data")
        if opts.get("group_column") and opts["group_column"] in locked_test:
            col = opts["group_column"]
            if set(dev[col].dropna()) & set(locked_test[col].dropna()):
                raise ValueError("Final test groups overlap development groups")
    if len(dev) < 40:
        raise ValueError("At least 40 development rows are required")
    if df_test is not None:
        shared = [c for c in dev if c in locked_test]
        if shared and len(pd.merge(dev[shared].drop_duplicates(),
                                   locked_test[shared].drop_duplicates(), on=shared)):
            raise ValueError("Final test contains rows duplicated in development data")
    assessment = assess_target(dev[target], task)
    requested = opts.get("selected_models")
    if requested:
        eligible = [name for name in requested if name in assessment.candidate_models]
        if not eligible:
            raise ValueError("No selected models are suitable for the declared target type")
    else:
        eligible = assessment.candidate_models
    opts["selected_models"] = eligible
    opts["selection_metric"] = task.objective if task.objective != "auto" else opts.get("selection_metric", "auto")
    drift = shift_report(dev, locked_test, target)
    # Each trial gets its own artifact directory; no agent receives those paths.
    def run_trial(config):
        with tempfile.TemporaryDirectory(prefix="regression_experiment_") as tmp:
            return _run_once(dev, None, None, target,
                             {**config, "output_dir": tmp, "compute_learning_curves": False,
                              "compute_permutation_importances": False,
                              "compute_intervals": False}, progress_callback)

    baseline = run_trial(opts)
    if baseline.selection_basis != "cv":
        raise ValueError("Eight-agent mode requires complete cross-validation before final-test evaluation")
    best, history = baseline, []
    locked_metric = baseline.selection_metric
    columns = set(map(str, dev.columns)) - {target}
    for specialist in SPECIALISTS:
        if progress_callback:
            progress_callback("agents", f"{specialist.role}: proposing an experiment")
        context = (_experiment_context(best) + f" Target kind: {assessment.kind}. "
                   f"Eligible models: {eligible}. Warnings: {assessment.warnings}")
        prompt = (f"{specialist.instruction} Available models: {available_model_names()}. "
                  f"Allowed actions: {specialist.scope}. Columns: {sorted(columns)[:80]}. "
                  "Choose exactly ONE supported action with a concrete hypothesis, or an empty actions array. "
                  "Return JSON only: {\"actions\":[{\"action\":\"...\",\"value\":...,\"reason\":\"...\"}]}. "
                  "Never request holdout metrics, code execution, or a changed split. "
                  f"Training/CV context: {context[:3500]}")
        try:
            raw = proposer(specialist, prompt) if proposer else crew_proposer(specialist, prompt, llm)
            proposal = valid_proposal(raw, specialist, columns)
        except Exception as exc:
            history.append({"agent": specialist.name, "accepted": False,
                            "reason": f"proposal failed: {type(exc).__name__}: {exc}"})
            continue
        if proposal is not None and proposal.action == "selected_models":
            if not set(proposal.value) <= set(assessment.candidate_models):
                proposal = None
        if proposal is None:
            history.append({"agent": specialist.name, "accepted": False,
                            "reason": "no valid proposal"})
            continue
        candidate_opts = apply_proposals(best.run_options, [proposal])
        candidate_opts["selection_metric"] = locked_metric
        try:
            candidate = run_trial(candidate_opts)
            accepted, reason = _improves(candidate, best)
        except Exception as exc:
            accepted, reason = False, f"experiment failed: {type(exc).__name__}: {exc}"
            candidate = None
        history.append({"agent": specialist.name, "action": proposal.action,
                        "value": proposal.value, "hypothesis": proposal.reason,
                        "accepted": accepted, "reason": reason})
        if accepted:
            best = candidate
            best.refinement_log = list(history)
    # This is the only call that receives final-test labels. Never send its
    # resulting scores back to a proposer, and never rerun based on them.
    final_options = {**best.run_options, "selection_metric": locked_metric,
                     "task_spec": asdict(task), "task_input_columns": original_feature_cols,
                     "reserve_final_test": df_test is None,
                     "final_refit": "train", "output_dir": options.get("output_dir")}
    final = _run_once(dev, None, locked_test, target, final_options, progress_callback)
    if final.selection_basis != "cv":
        raise ValueError("Final model lacks complete CV; final-test selection is forbidden")
    final.refinement_log = history
    groups_for_report = (locked_test[opts["group_column"]].to_numpy()
                         if split != "time" and opts.get("group_column") in locked_test else None)
    reliability = subgroup_error_report(final.preprocessing.y_test_original,
        final.results[final.best_model].y_pred_test, groups_for_report,
        time_ordered=split == "time")
    final.task_report = {"target_kind": assessment.kind, "objective": locked_metric,
                         "candidate_models": eligible, "warnings": assessment.warnings,
                         "feature_shift_alerts": drift, "subgroup_error": reliability,
                         "available_features_confirmed": task.available_features is not None}
    final.run_inputs = {"df_train": dev, "df_val": None, "df_test": locked_test, "target": target}
    final.warnings = list(final.warnings or []) + assessment.warnings
    if drift:
        final.warnings.append(f"Feature shift alerts in {len(drift)} final-test columns; inspect task_report")
    return final
