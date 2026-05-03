"""
CrewAI agents — eight specialists.
"""

from __future__ import annotations

from crewai import Agent

from .tools import (
    best_model_tool,
    preprocess_dataset_tool,
    profile_dataset_tool,
    quality_review_tool,
    train_models_tool,
)


def build_agents(llm) -> dict[str, Agent]:
    planner = Agent(
        role="ML Project Planner",
        goal="Lay out a concrete, sequenced plan for the regression analysis.",
        backstory="A senior ML engineer who thinks in steps and writes plans others can execute.",
        llm=llm, verbose=False, allow_delegation=False,
    )

    eda_agent = Agent(
        role="Exploratory Data Analyst",
        goal=("Profile the dataset and produce SPECIFIC observations citing actual "
              "column names, missing percentages, target stats, and skew values."),
        backstory=(
            "A data scientist who never trusts a CSV until inspecting every column. "
            "You write commentary that names columns, quotes percentages, and flags "
            "concrete risks — never vague phrasing like 'some columns may have issues'."
        ),
        tools=[profile_dataset_tool],
        llm=llm, verbose=False, allow_delegation=False,
    )

    preprocessor = Agent(
        role="Data Preprocessing Engineer",
        goal="Transform the raw data into clean numeric arrays and explain the choices.",
        backstory=(
            "You build robust scikit-learn pipelines. Median imputation for numerics, "
            "mode + one-hot for low-card categoricals, frequency encoding for high-card, "
            "datetime decomposition where appropriate, standardization at the end."
        ),
        tools=[preprocess_dataset_tool],
        llm=llm, verbose=False, allow_delegation=False,
    )

    modeler = Agent(
        role="ML Modeler & Evaluator",
        goal="Train multiple regressors and identify the best by RMSE on the held-out test set.",
        backstory=(
            "You believe in trying many models and letting metrics decide. "
            "You report the top three with concrete numbers and call out overfitting."
        ),
        tools=[train_models_tool, best_model_tool],
        llm=llm, verbose=False, allow_delegation=False,
    )

    chart_agent = Agent(
        role="Visualization Specialist",
        goal="Confirm the chart inventory and explain what each diagnostic plot reveals.",
        backstory=(
            "Visualization-first practitioner who believes a residual plot tells more "
            "than any single number."
        ),
        llm=llm, verbose=False, allow_delegation=False,
    )

    code_agent = Agent(
        role="Code Generator",
        goal="Describe the standalone Python script and Jupyter notebook produced.",
        backstory="You write code other engineers want to read.",
        llm=llm, verbose=False, allow_delegation=False,
    )

    insight_reporter = Agent(
        role="Insight Reporter",
        goal=("Translate the metrics into a plain-language executive summary that a "
              "non-technical reader can act on."),
        backstory="You write the executive summary at the top of every ML report.",
        tools=[best_model_tool],
        llm=llm, verbose=False, allow_delegation=False,
    )

    quality_reviewer = Agent(
        role="Quality Reviewer",
        goal=("Audit for target leakage, overfitting, degenerate targets, and skew. "
              "Give a clear go / caution / no-go recommendation."),
        backstory="The last line of defense before a model goes live.",
        tools=[quality_review_tool],
        llm=llm, verbose=False, allow_delegation=False,
    )

    return {
        "planner": planner, "eda": eda_agent, "preprocessor": preprocessor,
        "modeler": modeler, "chart": chart_agent, "code": code_agent,
        "insight": insight_reporter, "quality": quality_reviewer,
    }
