# Broader tabular regression mode

The eight-specialist mode now accepts a `RegressionTask`: the set of columns known when a prediction is made, the kind of target, a selection objective (`auto`, `rmse`, or `mae`), optional row-wise numeric ratios, and bounds for a bounded outcome. It reserves the test set before profiling the development targets or running agents. The system proposes candidate models based on the **development target only**, with Gamma and Tweedie candidates for appropriate positive outcomes, and reports unseen categories or unusual numeric ranges in the final test **without reading its target**. The generated script reproduces the feature contract and reserved split from the original input files. The saved model bundle recreates ratio features from raw prediction rows. Final-test subgroup or early/late-period errors are reported only after selection.

## Python example

```python
from crew.orchestrator import run_full_pipeline
from utils.task_contract import RegressionTask

# Supply only features that exist when a prediction is made.
task = RegressionTask(
    available_features=("income", "household_size", "region"),
    target_kind="positive",    # or auto, continuous, count, nonnegative, bounded
    objective="mae",           # or rmse / auto
    ratio_features=(("income", "household_size"),),
)
out = run_full_pipeline(
    df_train=data, target="annual_cost", regression_task=task,
    eight_agent_mode=True, selected_models=None,
    cv_folds=5, tuning="fast", use_agents=False,
)
print(out.task_report)
print(out.refinement_log)
```

In the UI, select **Eight independent regression specialists** and fill in **Features available at prediction time**. If you leave this blank, all non-target columns are assumed available and a warning is shown. Ratio features use one `numerator/denominator` pair per line. A bounded target also requires `lower,upper` bounds. The bounds are checked against training data, but predictions are not clipped or guaranteed to stay inside them.

If there are repeated entities, select a group column. If data are time ordered, select a time column and time-aware split. These are different validation designs; the current system does not support purged simultaneous group-and-time CV. The saved model fits only development rows, never the final-test target. A supplied external final test must be later in time or have disjoint groups where those settings apply.

## What this does not cover yet

Selection still optimizes RMSE or MAE. RMSLE, asymmetric costs, weighted errors, quantile loss, and arbitrary objectives are not implemented. There is no automatic lag/rolling feature builder or true forecast horizon specification, no hurdle/zero-inflated model, and no constrained bounded-outcome model. The target-type heuristic may confuse integer monetary amounts with counts: specify `target_kind` yourself when you know the meaning. The feature-shift report is a basic alert, not a statistical guarantee. The system cannot discover semantic leakage from names; explicitly provide only prediction-time features. Small datasets and repeated experiments remain uncertain even with locked final testing.
