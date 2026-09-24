"""CrewAI tasks - concrete expected outputs grounded in the run's tools."""

from __future__ import annotations


def build_tasks(agents: dict, target: str, n_features: int, n_rows: int):
    from crewai import Task

    plan = Task(
        description=(f"Use `Profile dataset`, `Preprocess dataset` and `Refinement log`. For the analysis "
                     f"of `{target}` ({n_rows} rows, {n_features} columns), list the 6-10 steps that were "
                     "ACTUALLY executed, naming each automatic decision (split, encodings, target "
                     "transform, refinement changes kept or rejected). End with 2-3 concrete next steps. "
                     "Plain markdown, no code blocks."),
        expected_output="A numbered list of executed steps with decisions, then next steps.",
        agent=agents["planner"])
    eda = Task(
        description=("Use `Profile dataset`. Write a specific EDA summary: shape; target mean/std/min/max/"
                     "skew (flag |skew| > 1.5); top missing columns; datetime, ID, numeric-text and "
                     "code columns; any column names suggesting target leakage. Cite real names and numbers."),
        expected_output="A 6-10 line EDA summary citing specific columns and values.",
        agent=agents["eda"], context=[plan])
    prep = Task(
        description=("Use `Preprocess dataset`. In 4-6 lines explain missing-value handling, categorical "
                     "encoding, the target transform and how it was chosen, datetime features, and the "
                     "final feature count."),
        expected_output="Preprocessing report in 4-6 lines with specific numbers.",
        agent=agents["preprocessor"], context=[eda])
    model = Task(
        description=("Use `Train and evaluate models`, then `Get best model summary`. Compare the top three "
                     "by CROSS-VALIDATED error (not test error) in 4-6 lines; discuss accuracy vs time vs "
                     "interpretability; state each one's fit diagnosis. If the baseline won, explain that "
                     "no reliable signal was found and what that means."),
        expected_output="Modeling report with concrete numbers and a winner explanation.",
        agent=agents["modeler"], context=[prep])
    chart = Task(
        description=("Use `Residual diagnostics`. For each chart family say what it shows FOR THIS RUN, "
                     "citing the numbers (bias, fanning, skew, tails, autocorrelation)."),
        expected_output="A bulleted list: chart -> what it shows here, with numbers.",
        agent=agents["chart"], context=[model])
    quality = Task(
        description=("Use `Quality review of pipeline` and `Refinement log`. Assign one verdict: GO, CAUTION "
                     "(name the caveats) or NO-GO (state what to change and re-run). Cite the findings."),
        expected_output="A clear verdict (GO / CAUTION / NO-GO) with cited justification.",
        agent=agents["quality"], context=[model])
    code = Task(
        description=("Use `Generated artifacts`. In 3-5 lines describe the script, notebook and "
                     "best_model.joblib and how to run / load them."),
        expected_output="Brief description of the generated artifacts.",
        agent=agents["code"], context=[model])
    insight = Task(
        description=("Write the executive summary (~10 lines) for a non-technical reader: what the data "
                     "showed, which model won and by how much versus predicting the average, how confident "
                     "to be (CV stability, interval coverage), and what to do next."),
        expected_output="A polished, plain-language summary (~10 lines).",
        agent=agents["insight"], context=[eda, model, quality])
    return [plan, eda, prep, model, chart, quality, code, insight]
