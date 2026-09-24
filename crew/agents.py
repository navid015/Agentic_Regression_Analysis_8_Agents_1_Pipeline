"""
CrewAI agents - eight specialists, each grounded in tools bound to ONE run.

v3: the chart and code agents used to have no tools at all (they described
charts and files they could not see); every agent now reads real numbers.
The "planner" reviews the plan that was actually executed, including the
advisor's refinement loop, instead of writing a plan nobody executes.
"""

from __future__ import annotations


def build_agents(llm, tools: dict):
    from crewai import Agent

    def agent(role, goal, backstory, tool_keys):
        return Agent(role=role, goal=goal, backstory=backstory, llm=llm, verbose=False,
                     allow_delegation=False, tools=[tools[k] for k in tool_keys])

    return {
        "planner": agent(
            "ML Project Planner",
            "Explain the plan that was executed - including automatic decisions and the refinement "
            "loop - and recommend concrete next steps.",
            "A senior ML engineer who distinguishes what WAS done from what SHOULD be done next.",
            ["profile", "preprocess", "refinement"]),
        "eda": agent(
            "Exploratory Data Analyst",
            "Produce SPECIFIC observations citing column names, missing percentages, target "
            "statistics and structural findings.",
            "Never trusts a CSV until every column is inspected; never writes vague phrases.",
            ["profile"]),
        "preprocessor": agent(
            "Data Preprocessing Engineer",
            "Explain every preprocessing choice and why it was made for THIS dataset.",
            "Builds leak-free scikit-learn pipelines fitted inside each CV fold.",
            ["preprocess"]),
        "modeler": agent(
            "ML Modeler & Evaluator",
            "Compare models on cross-validated error, explain fit diagnoses and why the winner won.",
            "Lets honest metrics decide: shared folds and paired comparisons pick the winner; the test "
            "set is only a final check. Says plainly when no model beats the baseline.",
            ["train", "best"]),
        "chart": agent(
            "Visualization Specialist",
            "Explain what the diagnostic charts show, using the numeric residual diagnostics.",
            "Believes a residual plot tells more than any single number - and quantifies it.",
            ["diagnostics", "best"]),
        "code": agent(
            "Code Generator",
            "Describe the generated script, notebook and model bundle and how to use them.",
            "Writes code other engineers want to read.",
            ["artifacts"]),
        "insight": agent(
            "Insight Reporter",
            "Write a plain-language executive summary a non-technical reader can act on.",
            "Writes the summary at the top of every ML report.",
            ["best", "quality"]),
        "quality": agent(
            "Quality Reviewer",
            "Audit for leakage, split overlap, over/underfitting, low signal and interval coverage, and "
            "give a clear go / caution / no-go verdict.",
            "The last line of defense before a model goes live.",
            ["quality", "refinement"]),
    }
