# Eight regression specialists

This experimental mode turns eight CrewAI agents into experiment proposers. Each receives the current training summary, cross-validation scores, and previous accepted experiments. A shared Python evaluator validates and runs its proposal. The agents never receive final-test labels or final-test scores. One final evaluation occurs after all eight proposals are complete.

## Roles

| Agent | Proposals it may submit |
| --- | --- |
| Data scientist | Categorical column treatment, missing indicators |
| Validation scientist | Tuning intensity; audits the locked split |
| Feature engineer | Interactions or log target |
| Linear/GLM scientist | Registered linear/GLM models |
| Tree scientist | Registered tree/forest models |
| Boosting scientist | Registered boosting models |
| Ensemble scientist | Existing ensemble on/off |
| Regression critic | Encoding or tuning robustness experiment |

Each specialist submits at most one experiment per run. Failed or invalid proposals are logged. A proposal is accepted only when paired CV improvement clears the existing noise threshold. Model names, actions, and columns are allowlisted. The evaluator maintains the original split and metric. No agent can execute shell commands or Python code.

## Start

Install `requirements.txt`, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and select **Eight independent regression specialists** in the Setup tab. The mode is also available from Python:

```python
from crew.orchestrator import run_full_pipeline

out = run_full_pipeline(
    df_train=data, target="target", eight_agent_mode=True,
    selected_models=["Ridge", "RandomForest", "XGBoost"],
    use_agents=False,  # optionally add eight post-run reporting agents
    tuning="off", cv_folds=5,
)
print(out.best_model)
for experiment in out.refinement_log:
    print(experiment)
```

For temporal data use `split_strategy="time"`, `cv_strategy="time"`, `time_column="date"`. For repeated entities use `cv_strategy="group"`, `group_column="entity"`. A supplied `df_test` remains reserved for the final call. A separate `df_val` is not supported in this mode. The saved model is fitted on development data only; it does not refit using final-test labels.

## Practical limits

The current proposals use existing preprocessing operations and model registry. The agents cannot create arbitrary transformers, add CatBoost, tune arbitrary hyperparameters, or implement stacking. The eight experiments run sequentially and can be expensive. Repeated use of the same CV folds may overfit model selection; treat a single untouched final test as the estimate of generalization. The existing pipeline still runs an internal development holdout during each experiment, but those scores are never sent to agents or used by the accept/reject gate.
