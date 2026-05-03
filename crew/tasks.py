"""
CrewAI tasks — sharper expected outputs to drive concrete commentary.
"""

from __future__ import annotations

from crewai import Task


def build_tasks(agents: dict, target: str, n_features: int, n_rows: int) -> list[Task]:
    plan_task = Task(
        description=(
            f"Plan an end-to-end regression analysis on a dataset with {n_rows} rows "
            f"and {n_features} columns, targeting `{target}`. "
            "Produce a numbered plan with 6–10 steps. Each step: what happens, why, "
            "what artifact it produces. Use plain markdown — bullets / numbered lists, "
            "no code blocks."
        ),
        expected_output="A numbered, well-structured plan with 6–10 concrete steps.",
        agent=agents["planner"],
    )

    eda_task = Task(
        description=(
            "Use the `Profile dataset` tool. Then write a specific EDA summary that:\n"
            "  1. States the dataset shape (rows × cols).\n"
            "  2. Reports target mean, std, min, max, skew — call out skewness if |skew| > 1.5.\n"
            "  3. Names the top 3–5 columns with the highest missing-percentages.\n"
            "  4. Names any datetime columns detected.\n"
            "  5. Names any low-variance or high-cardinality columns.\n"
            "  6. Flags any columns whose names suggest possible target leakage "
            "(e.g. 'price_predicted', 'target_x100', columns ending in '_log').\n"
            "Cite actual column names and numbers from the tool output. "
            "Avoid vague phrases like 'the data looks reasonable'."
        ),
        expected_output=(
            "A 6–10 line EDA summary citing specific column names and numerical values."
        ),
        agent=agents["eda"], context=[plan_task],
    )

    preprocess_task = Task(
        description=(
            "Run the `Preprocess dataset` tool. Then in 4–6 lines, explain:\n"
            "  - how missing values were handled,\n"
            "  - how categoricals were encoded,\n"
            "  - whether log-transform was applied,\n"
            "  - whether datetime features were extracted,\n"
            "  - the final feature count after encoding."
        ),
        expected_output="Preprocessing report in 4–6 lines with specific numbers.",
        agent=agents["preprocessor"], context=[eda_task],
    )

    modeling_task = Task(
        description=(
            "Run `Train and evaluate models`, then `Get best model summary`. "
            "Compare the top three by RMSE in 4–6 lines. Discuss tradeoffs "
            "(accuracy vs train time vs interpretability). Flag any overfitting signs."
        ),
        expected_output="Modeling report with concrete numbers and a winner recommendation.",
        agent=agents["modeler"], context=[preprocess_task],
    )

    chart_task = Task(
        description=(
            "List the per-model diagnostic charts the system produces (Predicted vs "
            "Actual, Residuals vs Predicted, Residual Distribution, Q-Q Plot, CV R² Box, "
            "Feature Importance) and the comparison charts (RMSE/MAE/R² bars, grouped "
            "metrics, training-time, predicted-vs-actual overlay, CV R² box across "
            "models). Briefly note what each chart family reveals."
        ),
        expected_output="A bulleted inventory with one-line interpretations.",
        agent=agents["chart"], context=[modeling_task],
    )

    quality_task = Task(
        description=(
            "Run `Quality review of pipeline`. Then assign one of three verdicts:\n"
            "  ✅ GO — pipeline can be trusted as-is\n"
            "  ⚠️ CAUTION — usable but with named caveats\n"
            "  ❌ NO-GO — re-run with changes (state which)\n"
            "Justify your verdict by citing the specific findings from the tool."
        ),
        expected_output="A clear verdict (GO / CAUTION / NO-GO) with cited justification.",
        agent=agents["quality"], context=[modeling_task],
    )

    code_task = Task(
        description=(
            "In 3–5 lines, describe what the auto-generated `regression_pipeline.py` "
            "and `.ipynb` contain — the steps reproduced, the saved artifacts "
            "(`best_model.joblib` etc.), and where the user should run them."
        ),
        expected_output="Brief description of the generated code artifacts.",
        agent=agents["code"], context=[modeling_task],
    )

    insight_task = Task(
        description=(
            "Write the final executive summary (~10 lines) for a non-technical reader. "
            "Cover: what the data showed, which model won and by how much, how confident "
            "we should be (cite test R² and CV stability), and what the user should do next "
            "(deploy / collect more data / try feature engineering / be wary of leakage)."
        ),
        expected_output="A polished, plain-language summary (~10 lines).",
        agent=agents["insight"], context=[eda_task, modeling_task, quality_task],
    )

    return [plan_task, eda_task, preprocess_task, modeling_task,
            chart_task, quality_task, code_task, insight_task]
