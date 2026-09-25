# Agent-driven regression experiments

The existing deterministic regression evaluator now accepts an LLM-selected subset of registered models through the refinement planner. Each accepted proposal reruns the entire pipeline on the same split configuration; decisions use paired cross-validation errors. The baseline remains included. Rejected and accepted proposals appear in `PipelineOutput.refinement_log`.

## Run

Install dependencies with `pip install -r requirements.txt`. Set the credential expected by CrewAI (`OPENAI_API_KEY` or the provider credentials in your CrewAI configuration). The LLM is optional: rule-based refinement works without credentials.

```python
import pandas as pd
from crew.orchestrator import run_full_pipeline

df = pd.read_csv("data.csv")
out = run_full_pipeline(
    df_train=df, target="target", auto_refine=True,
    use_llm_advisor=True, refine_rounds=5,
    use_agents=True, tuning="fast", n_jobs=2,
)
print(out.best_model, out.refinement_log)
```

For data ordered in time, set `split_strategy="time"`, `cv_strategy="time"`, and `time_column="date"`. For repeated entities, use `cv_strategy="group"` and `group_column="entity"`. Explicit validation and test files can be passed as `df_val` and `df_test`.

The planner may request registered model subsets, tuning, categorical treatment, encoding, interactions, missing indicators, target transform, and ensemble on/off. It cannot run shell commands or alter split rules. Suspected leak column drops remain proposals for user review. At most three experiments run per round and at most twenty rounds run in one call. Only training summaries and CV scores are sent to the proposal model.

## Current limits

The implementation runs the existing full evaluation for each experiment, so it is expensive and the internal holdout is recomputed each time. The advisor does not receive holdout scores, but this is not a fully isolated one-time final test. Repeated CV comparisons may overfit the validation protocol. The agents do not write new Python feature transformers or install model packages, and this release does not add CatBoost, stacking, causal temporal target encoding, or arbitrary user-defined metrics. For a publishable unbiased estimate, evaluate the finalized model once on an untouched external test set.

Temporal high-cardinality categoricals use frequency encoding even when the target-encoding option is requested: scikit-learn's TargetEncoder uses regular KFold internally and cannot accept TimeSeriesSplit through its `cv` integer parameter.
