"""
Plotly-based visualizations for regression model results.

Two families:
  - per_model_charts(): everything for ONE model
  - comparison_charts(): how models stack up

All figures use the 'plotly_white' template; in dark themes the chart areas
remain readable because we set explicit text colors on the layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# slightly larger fonts + explicit text colors so charts read in both themes
_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(size=12, color="#111827"),
    title_font=dict(size=15, color="#111827"),
)


def _apply_defaults(fig: go.Figure, height: int = 420) -> go.Figure:
    # Right-margin bumped to 80px — model names like "GradientBoosting" + bar
    # value labels were overflowing into the legend at narrower margins.
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=80, t=55, b=60),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---- per-model charts -------------------------------------------------------


def predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(size=6, opacity=0.6, color="#3b82f6"), name="Predictions",
    ))
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#dc2626", dash="dash"), name="Ideal (y=x)",
    ))
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted")
    return _apply_defaults(fig)


def residuals_vs_predicted(y_true, y_pred, title="Residuals vs Predicted") -> go.Figure:
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(size=6, opacity=0.6, color="#10b981"), name="Residuals",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#dc2626")
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Residual (y - ŷ)")
    return _apply_defaults(fig)


def residual_distribution(y_true, y_pred, title="Residual Distribution") -> go.Figure:
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color="#8b5cf6", opacity=0.85))
    fig.update_layout(title=title, xaxis_title="Residual", yaxis_title="Frequency", bargap=0.05)
    return _apply_defaults(fig, height=380)


def qq_plot(y_true, y_pred, title="Q-Q Plot of Residuals") -> go.Figure:
    residuals = y_true - y_pred
    (theo_q, samp_q), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    line_x = np.array([theo_q.min(), theo_q.max()])
    line_y = slope * line_x + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo_q, y=samp_q, mode="markers",
                             marker=dict(color="#f59e0b", size=6), name="Residuals"))
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                             line=dict(color="#dc2626", dash="dash"), name="Normal fit"))
    fig.update_layout(title=title, xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
    return _apply_defaults(fig, height=380)


def feature_importance_chart(importances, feature_names, title="Feature Importance",
                             top_k=20) -> go.Figure | None:
    if importances is None or len(importances) != len(feature_names):
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(top_k)
    fig = px.bar(df, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Blues")
    fig.update_layout(title=title, coloraxis_showscale=False)
    return _apply_defaults(fig, height=max(380, 22 * len(df)))


def cv_score_box(model_name, cv_r2) -> go.Figure | None:
    if not cv_r2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Box(y=cv_r2, name=model_name, marker_color="#0ea5e9", boxmean=True))
    fig.update_layout(title=f"{model_name} — CV R² distribution", yaxis_title="R² (per fold)")
    return _apply_defaults(fig, height=360)


def per_model_charts(model_name, y_test, y_pred_test, cv_r2,
                     importances, feature_names) -> dict[str, go.Figure]:
    charts: dict[str, go.Figure] = {
        "Predicted vs Actual":     predicted_vs_actual(y_test, y_pred_test,
                                                        f"{model_name} — Predicted vs Actual"),
        "Residuals vs Predicted":  residuals_vs_predicted(y_test, y_pred_test,
                                                           f"{model_name} — Residuals vs Predicted"),
        "Residual Distribution":   residual_distribution(y_test, y_pred_test,
                                                          f"{model_name} — Residual Distribution"),
        "Q-Q Plot":                qq_plot(y_test, y_pred_test, f"{model_name} — Q-Q Plot"),
    }
    cv_fig = cv_score_box(model_name, cv_r2)
    if cv_fig is not None:
        charts["CV R² Distribution"] = cv_fig
    fi_fig = feature_importance_chart(importances, feature_names,
                                      f"{model_name} — Feature Importance")
    if fi_fig is not None:
        charts["Feature Importance"] = fi_fig
    return charts


# ---- comparison charts -----------------------------------------------------


def metrics_comparison_bar(metrics_df, metric, ascending=True) -> go.Figure:
    df = metrics_df[["Model", metric]].sort_values(metric, ascending=ascending)
    fig = px.bar(df, x="Model", y=metric, color=metric,
                 color_continuous_scale="Viridis", text_auto=".4f")
    fig.update_layout(title=f"Model comparison — {metric}", coloraxis_showscale=False)
    return _apply_defaults(fig, height=420)


def all_metrics_grouped(metrics_df) -> go.Figure:
    show = ["MAE", "RMSE", "MedianAE"]
    have = [m for m in show if m in metrics_df.columns]
    long = metrics_df.melt(id_vars="Model", value_vars=have,
                           var_name="Metric", value_name="Value")
    fig = px.bar(long, x="Model", y="Value", color="Metric", barmode="group", text_auto=".3f")
    fig.update_layout(title="Error metrics — grouped (lower is better)")
    return _apply_defaults(fig, height=460)


def cv_comparison_box(results) -> go.Figure | None:
    rows = []
    for name, r in results.items():
        for s in r.cv_scores.get("R2", []):
            rows.append({"Model": name, "R2": s})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.box(df, x="Model", y="R2", color="Model", points="all")
    fig.update_layout(title="Cross-validation R² — all models", showlegend=False)
    return _apply_defaults(fig, height=460)


def training_time_chart(results) -> go.Figure:
    df = pd.DataFrame(
        [{"Model": n, "Train time (s)": r.train_time_sec} for n, r in results.items()]
    ).sort_values("Train time (s)")
    fig = px.bar(df, x="Model", y="Train time (s)", color="Train time (s)",
                 color_continuous_scale="Reds", text_auto=".3f")
    fig.update_layout(title="Training time per model", coloraxis_showscale=False)
    return _apply_defaults(fig, height=400)


def overlay_pred_vs_actual(y_test, results) -> go.Figure:
    fig = go.Figure()
    lo = float(min(y_test.min(), min(r.y_pred_test.min() for r in results.values())))
    hi = float(max(y_test.max(), max(r.y_pred_test.max() for r in results.values())))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#dc2626", dash="dash"), name="Ideal (y=x)"))
    palette = px.colors.qualitative.Bold
    for i, (name, r) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=y_test, y=r.y_pred_test, mode="markers", name=name,
            marker=dict(size=5, opacity=0.55, color=palette[i % len(palette)]),
        ))
    fig.update_layout(title="Predicted vs Actual — all models overlaid",
                      xaxis_title="Actual", yaxis_title="Predicted")
    return _apply_defaults(fig, height=520)


def comparison_charts(results, y_test) -> dict[str, go.Figure]:
    metrics_df = pd.DataFrame([{"Model": n, **r.metrics} for n, r in results.items()])
    charts: dict[str, go.Figure] = {
        "RMSE comparison":              metrics_comparison_bar(metrics_df, "RMSE", ascending=True),
        "MAE comparison":               metrics_comparison_bar(metrics_df, "MAE",  ascending=True),
        "R² comparison":                metrics_comparison_bar(metrics_df, "R2",   ascending=False),
        "All error metrics grouped":    all_metrics_grouped(metrics_df),
        "Training time":                training_time_chart(results),
        "Predicted vs Actual (overlay)": overlay_pred_vs_actual(y_test, results),
    }
    cv_fig = cv_comparison_box(results)
    if cv_fig is not None:
        charts["CV R² (all models)"] = cv_fig
    return charts


def target_distribution_chart(y: np.ndarray, target_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y, nbinsx=40, marker_color="#3b82f6", opacity=0.85))
    fig.update_layout(title=f"Target distribution — {target_name}",
                      xaxis_title=target_name, yaxis_title="Count", bargap=0.05)
    return _apply_defaults(fig, height=320)
