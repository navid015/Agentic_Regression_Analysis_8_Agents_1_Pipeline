"""
Regression Crew — Gradio UI.

Fixes from the previous Streamlit version:
  - All metric cards now use explicit text/background colors that work in
    light AND dark themes (no more invisible white-on-white).
  - The metrics table highlights best values with a colored BORDER instead of
    a low-contrast fill, so values stay readable in dark mode.
  - Adds: multi-file upload (train / val / test), dataset preview (10 rows),
    configurable CV folds, log-transform toggle, time/group-aware split,
    optional XGBoost/LightGBM, optional hyperparameter tuning, Scope tab.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crew.orchestrator import PipelineOutput, run_full_pipeline
from utils.modeling import available_model_names
from utils.visualization import (
    comparison_charts,
    per_model_charts,
    target_distribution_chart,
)


# ---------------------------------------------------------------------------
# Styling — works on both light and dark Gradio themes
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ---------- BASE: deep dark background with subtle indigo glow ---------- */
body, gradio-app, .gradio-container {
    background: #0a0a14 !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(99, 102, 241, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 80% 60% at 100% 100%, rgba(139, 92, 246, 0.10) 0%, transparent 60%) !important;
    background-attachment: fixed !important;
    color: #e2e8f0 !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 28px 28px !important;
}

/* ---------- HEADER ---------- */
.main-header {
    background: linear-gradient(135deg, rgba(76, 29, 149, 0.35) 0%, rgba(15, 15, 36, 0.85) 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    color: #e2e8f0 !important;
    padding: 32px 36px;
    border-radius: 16px;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.main-header h1 {
    color: #c4b5fd !important;
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.main-header p {
    color: #94a3b8 !important;
    margin: 12px 0 0;
    font-size: 0.95rem;
    line-height: 1.65;
    max-width: 1100px;
}

/* ---------- TOP-OF-PAGE ALERT/NOTICE BOXES ---------- */
.scope-notice {
    background: rgba(6, 182, 212, 0.06);
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-left: 4px solid #06b6d4;
    padding: 14px 22px;
    border-radius: 8px;
    margin-bottom: 14px;
    color: #cbd5e1 !important;
    font-size: 0.92rem;
    line-height: 1.55;
}
.scope-notice .label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #22d3ee;
    margin-right: 12px;
}
.scope-notice b { color: #e2e8f0; }

/* Status banner — sits above tabs, updates throughout the pipeline */
.status-banner {
    border-radius: 8px;
    padding: 13px 22px;
    margin-bottom: 18px;
    font-family: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', 'Source Code Pro', monospace;
    font-size: 0.92rem;
    border: 1px solid rgba(255,255,255,0.05);
}
.status-banner.idle    { background: rgba(100, 116, 139, 0.08); border-left: 4px solid #64748b; color: #94a3b8 !important; }
.status-banner.running { background: rgba(245, 158, 11, 0.08); border-left: 4px solid #f59e0b; color: #fbbf24 !important; }
.status-banner.success { background: rgba(34, 197, 94, 0.08);  border-left: 4px solid #22c55e; color: #86efac !important; }
.status-banner.error   { background: rgba(239, 68, 68, 0.08);  border-left: 4px solid #ef4444; color: #fca5a5 !important; }
.status-banner b { color: #ffffff; }

/* ---------- TABS ---------- */
.tab-nav {
    border-bottom: 1px solid rgba(139, 92, 246, 0.2) !important;
    margin-bottom: 12px !important;
}
.tab-nav button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 18px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
.tab-nav button.selected {
    color: #c4b5fd !important;
    border-bottom: 2px solid #8b5cf6 !important;
}
.tab-nav button:hover { color: #e2e8f0 !important; }

/* ---------- METRIC CARDS — dark glass ---------- */
.metric-card {
    background: rgba(20, 20, 38, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    color: #e2e8f0 !important;
    transition: all 0.2s ease;
    backdrop-filter: blur(4px);
}
.metric-card:hover {
    border-color: rgba(139, 92, 246, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.15);
}
.metric-card .label {
    color: #94a3b8 !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
    font-weight: 600;
}
.metric-card .value {
    color: #e2e8f0 !important;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.metric-card.gold {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(20, 20, 38, 0.7) 100%);
    border-color: rgba(245, 158, 11, 0.45);
}
.metric-card.gold .value { color: #fbbf24 !important; }

/* ---------- SECTION HEADERS — vertical bar + small caps ---------- */
.section-header {
    padding: 6px 0 6px 14px;
    border-left: 4px solid #8b5cf6;
    color: #c4b5fd !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin: 22px 0 12px;
}

/* ---------- TIMELINE ---------- */
.timeline-container {
    background: rgba(20, 20, 38, 0.55);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0 16px;
    font-family: 'JetBrains Mono', ui-monospace, 'Cascadia Mono', monospace;
    font-size: 0.9rem;
    color: #cbd5e1;
    min-height: 60px;
    backdrop-filter: blur(4px);
}
.timeline-empty {
    color: #64748b;
    font-style: italic;
    padding: 8px 0;
}
.timeline-entry {
    padding: 7px 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.08);
    display: flex;
    align-items: baseline;
    gap: 12px;
}
.timeline-entry:last-child { border-bottom: none; }
.timeline-entry .ico { font-family: ui-sans-serif, system-ui, sans-serif; flex-shrink: 0; width: 1.2em; }
.timeline-entry.done   { color: #86efac; }
.timeline-entry.active { color: #fbbf24; animation: pulse 1.5s ease-in-out infinite; }
.timeline-entry.error  { color: #fca5a5; }
.timeline-entry.pending { color: #64748b; }
.timeline-entry .elapsed { color: #64748b; margin-left: auto; font-size: 0.82rem; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.65; }
}

/* ---------- ACTIVE-MODEL PANEL ---------- */
.model-panel {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(20, 20, 38, 0.4) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0 14px;
}
.model-panel h3 {
    color: #e2e8f0 !important;
    margin: 0 0 4px;
    font-size: 1.08rem;
    font-weight: 600;
}
.model-panel .desc {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.5;
}

/* ---------- AGENT/EDA CARDS ---------- */
.agent-card {
    background: rgba(245, 158, 11, 0.06);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-left: 4px solid #eab308;
    padding: 14px 20px;
    border-radius: 6px;
    color: #fde68a !important;
}

/* Inline scope boxes still used in Setup tab + per-model warnings */
.scope-good   { background: rgba(34, 197, 94, 0.08);   border: 1px solid rgba(34, 197, 94, 0.25);
                border-left: 4px solid #22c55e;
                padding: 12px 18px; border-radius: 6px; color: #86efac !important; }
.scope-warn   { background: rgba(245, 158, 11, 0.08);  border: 1px solid rgba(245, 158, 11, 0.25);
                border-left: 4px solid #f59e0b;
                padding: 12px 18px; border-radius: 6px; color: #fcd34d !important; }
.scope-bad    { background: rgba(239, 68, 68, 0.08);   border: 1px solid rgba(239, 68, 68, 0.25);
                border-left: 4px solid #ef4444;
                padding: 12px 18px; border-radius: 6px; color: #fca5a5 !important; }

/* ---------- DATAFRAME — dark with purple header ---------- */
.gradio-container .gr-dataframe table {
    color: #e2e8f0 !important;
    background: rgba(20, 20, 38, 0.5) !important;
    border-color: rgba(99, 102, 241, 0.15) !important;
}
.gradio-container .gr-dataframe th {
    background: rgba(99, 102, 241, 0.25) !important;
    color: #e2e8f0 !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
}
.gradio-container .gr-dataframe td {
    border-color: rgba(99, 102, 241, 0.1) !important;
}

/* Run button — make it pop */
button.lg.primary {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35) !important;
}
button.lg.primary:hover {
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
    transform: translateY(-1px);
}
"""


# ---------------------------------------------------------------------------
# State container — held in gr.State across events
# ---------------------------------------------------------------------------
def _empty_state() -> dict:
    return {"output": None, "df_train": None, "df_val": None, "df_test": None,
            "train_filename": None, "val_filename": None, "test_filename": None}


# ---------------------------------------------------------------------------
# Helpers — file I/O, target column detection
# ---------------------------------------------------------------------------
def _load_csv(filepath: str | None) -> pd.DataFrame | None:
    if filepath is None or not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)


def _numeric_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None: return []
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _all_columns(df: pd.DataFrame | None) -> list[str]:
    return list(df.columns) if df is not None else []


# ---------------------------------------------------------------------------
# UI event handlers
# ---------------------------------------------------------------------------
def on_mode_change(mode: str):
    """Toggle visibility of single-file vs multi-file uploaders."""
    is_single = (mode == "Single file (auto-split)")
    return (
        gr.update(visible=is_single),       # single file box
        gr.update(visible=not is_single),   # multi file box
    )


def on_files_uploaded(mode, single_file, train_file, val_file, test_file, state):
    """Load CSVs, update preview + dropdowns + state."""
    state = state or _empty_state()

    if mode == "Single file (auto-split)":
        df_train = _load_csv(single_file)
        df_val = df_test = None
        train_name = os.path.basename(single_file) if single_file else None
        val_name = test_name = None
    else:
        df_train = _load_csv(train_file)
        df_val   = _load_csv(val_file)
        df_test  = _load_csv(test_file)
        train_name = os.path.basename(train_file) if train_file else None
        val_name   = os.path.basename(val_file)   if val_file   else None
        test_name  = os.path.basename(test_file)  if test_file  else None

    state.update({
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "train_filename": train_name, "val_filename": val_name, "test_filename": test_name,
    })

    if df_train is None:
        return (state, gr.update(value=None),
                gr.update(value="*Upload a CSV to see a preview.*"),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None))

    preview = df_train.head(10)
    info_md = (f"**Train:** {df_train.shape[0]:,} rows × {df_train.shape[1]} cols"
               f" — `{train_name}`")
    if df_val is not None:
        info_md += f"  \n**Val:** {df_val.shape[0]:,} rows — `{val_name}`"
    if df_test is not None:
        info_md += f"  \n**Test:** {df_test.shape[0]:,} rows — `{test_name}`"

    num_cols = _numeric_columns(df_train)
    all_cols = _all_columns(df_train)
    target_default = num_cols[-1] if num_cols else None

    return (
        state,
        gr.update(value=preview),
        gr.update(value=info_md),
        gr.update(choices=num_cols, value=target_default),                           # target
        gr.update(choices=["(none)"] + all_cols, value="(none)"),                    # time col
        gr.update(choices=["(none)"] + all_cols, value="(none)"),                    # group col
    )


import html as _html_mod
import threading
import time as _time_mod


def _render_timeline(events: list[dict]) -> str:
    """Render the timeline events as HTML."""
    if not events:
        return ('<div class="timeline-container">'
                '<div class="timeline-empty">Waiting to start...</div>'
                '</div>')
    icon_for = {"done": "✅", "active": "🔄", "pending": "⏳", "error": "❌"}
    rows = []
    for ev in events:
        cls = ev["status"]
        ico = icon_for.get(cls, "•")
        text = _html_mod.escape(ev["text"])
        elapsed = ""
        if ev.get("elapsed") is not None and ev["elapsed"] >= 0.1:
            elapsed = f'<span class="elapsed">{ev["elapsed"]:.1f}s</span>'
        rows.append(
            f'<div class="timeline-entry {cls}">'
            f'<span class="ico">{ico}</span>'
            f'<span>{text}</span>'
            f'{elapsed}'
            '</div>'
        )
    return '<div class="timeline-container">' + "".join(rows) + "</div>"


def _status_banner(kind: str, message: str) -> str:
    return f'<div class="status-banner {kind}">{message}</div>'


# How many output components come AFTER (status_banner, state, timeline_display).
# These are all the result components from _render_results — we provide gr.update()
# placeholders during in-progress yields so they don't get reset.
_NUM_RESULT_OUTPUTS = 31


def on_run(
    mode, target, time_col, group_col,
    test_size, random_state, n_folds,
    selected_models, split_strategy, log_target,
    auto_datetime, drop_lowvar, drop_id_cols, tune_hyperparams, use_agents,
    state,
):
    """Generator: streams timeline updates while the pipeline runs on a thread."""
    state = state or _empty_state()
    df_train = state.get("df_train")

    placeholders = [gr.update()] * _NUM_RESULT_OUTPUTS

    # ---- input validation ----
    if df_train is None:
        yield (_status_banner("error", "❌ <b>No training data</b> — upload a CSV in the Setup tab."),
               state, gr.update(), *placeholders)
        return
    if not target:
        yield (_status_banner("error", "❌ <b>No target column selected</b> — pick one in Setup."),
               state, gr.update(), *placeholders)
        return
    if not selected_models:
        yield (_status_banner("error", "❌ <b>No models selected</b> — pick at least one in Setup."),
               state, gr.update(), *placeholders)
        return

    # ---- timeline state (mutated from both this function and the worker thread) ----
    events: list[dict] = []
    lock = threading.Lock()

    def _add(text: str, status: str = "active") -> None:
        """Add a new event. Marks any previous active event as done first."""
        with lock:
            for ev in events:
                if ev["status"] == "active":
                    ev["status"] = "done"
                    ev["elapsed"] = _time_mod.time() - ev["t0"]
            events.append({"text": text, "status": status,
                           "t0": _time_mod.time(), "elapsed": None})

    def _finish_last(status: str = "done") -> None:
        with lock:
            if events and events[-1]["status"] == "active":
                events[-1]["status"] = status
                events[-1]["elapsed"] = _time_mod.time() - events[-1]["t0"]

    # The orchestrator's progress_callback signature is (stage, msg).
    # We use that to drive timeline updates.
    def progress_cb(stage: str, msg: str) -> None:
        _add(msg, "active")

    # ---- start the timeline ----
    _add("Validating inputs...", "active")
    _finish_last("done")
    _add("Starting pipeline...", "active")

    yield (_status_banner("running",
                          "⚡ <b>Pipeline running...</b> see the Overview tab for live progress."),
           state, gr.update(value=_render_timeline(events)), *placeholders)

    # ---- run the pipeline on a worker thread so we can yield UI updates ----
    result_holder: dict = {"output": None, "error": None, "traceback": None}

    def worker():
        try:
            result_holder["output"] = run_full_pipeline(
                df_train=df_train,
                df_val=state.get("df_val"),
                df_test=state.get("df_test"),
                target=target,
                selected_models=selected_models,
                test_size=test_size, random_state=int(random_state), cv_folds=int(n_folds),
                cv_strategy=("group" if group_col not in (None, "(none)") else
                             "time"  if split_strategy == "Time-aware" else "kfold"),
                split_strategy=("time" if split_strategy == "Time-aware" else "random"),
                time_column=None if time_col in (None, "(none)") else time_col,
                group_column=None if group_col in (None, "(none)") else group_col,
                log_transform_target=log_target,
                auto_datetime_features=auto_datetime,
                drop_low_variance=drop_lowvar,
                auto_drop_id_columns=drop_id_cols,
                tune_hyperparameters=tune_hyperparams,
                use_agents=use_agents,
                progress_callback=progress_cb,
                csv_filename=state.get("train_filename") or "train.csv",
                val_csv_filename=state.get("val_filename"),
                test_csv_filename=state.get("test_filename"),
            )
        except Exception as e:
            import traceback
            result_holder["error"] = e
            result_holder["traceback"] = traceback.format_exc()

    t0_total = _time_mod.time()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Poll the event list while the worker runs; yield whenever it grows.
    last_count = len(events)
    while thread.is_alive():
        _time_mod.sleep(0.5)
        if len(events) != last_count:
            yield (_status_banner("running",
                                  f"⚡ <b>Pipeline running...</b> {len(events)} steps so far"),
                   state, gr.update(value=_render_timeline(events)), *placeholders)
            last_count = len(events)
    thread.join()

    # ---- error path ----
    if result_holder["error"]:
        _finish_last("error")
        with lock:
            events.append({"text": f"Error: {type(result_holder['error']).__name__}: "
                                   f"{result_holder['error']}",
                           "status": "error", "t0": _time_mod.time(), "elapsed": None})
        yield (_status_banner("error", f"❌ <b>{type(result_holder['error']).__name__}</b>: "
                                       f"{_html_mod.escape(str(result_holder['error']))}"),
               state, gr.update(value=_render_timeline(events)), *placeholders)
        return

    # ---- success path ----
    out = result_holder["output"]
    state["output"] = out
    total_elapsed = _time_mod.time() - t0_total

    _finish_last("done")
    with lock:
        events.append({
            "text": f"Done — best model: {out.best_model} "
                    f"(RMSE = {out.results[out.best_model].metrics['RMSE']:.4f})",
            "status": "done", "t0": _time_mod.time(),
            "elapsed": None,
        })

    yield (_status_banner(
                "success",
                f"✅ <b>Complete</b> in {total_elapsed:.1f}s — best model: "
                f"<b>{out.best_model}</b> "
                f"(RMSE: {out.results[out.best_model].metrics['RMSE']:.4f}, "
                f"R²: {out.results[out.best_model].metrics['R2']:.4f})"),
           state,
           gr.update(value=_render_timeline(events)),
           *_render_results(out, df_train))


# ---------------------------------------------------------------------------
# Result rendering — turns the PipelineOutput into all the UI components
# ---------------------------------------------------------------------------
def _metric_card(label: str, value: str, gold: bool = False) -> str:
    cls = "metric-card gold" if gold else "metric-card"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div></div>'


def _render_results(out: PipelineOutput, df_preview: pd.DataFrame):
    best = out.best_model
    bm = out.results[best].metrics

    # ---- Overview ----
    cards_html = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;">
        {_metric_card("Rows", f"{out.profile['n_rows']:,}")}
        {_metric_card("Features", f"{out.preprocessing.summary['n_features_after']}")}
        {_metric_card("Models trained", f"{len(out.results)}")}
        {_metric_card("Best R²", f"{bm['R2']:.4f}", gold=True)}
    </div>
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px;">
        {_metric_card("Best model", best, gold=True)}
        {_metric_card("RMSE", f"{bm['RMSE']:.4f}")}
        {_metric_card("MAE",  f"{bm['MAE']:.4f}")}
        {_metric_card("MedianAE", f"{bm['MedianAE']:.4f}")}
        {_metric_card("MAPE", f"{bm.get('MAPE_pct', float('nan')):.2f}%")}
    </div>
    """

    # plan
    plan_md = out.agent_outputs.get("planner") if out.agent_outputs else None
    if not plan_md:
        plan_md = (
            "*(Agent narration disabled — set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to enable.)*\n\n"
            "**Default plan executed by the deterministic pipeline:**\n\n"
            "1. Profile dataset (types, missingness, target stats, datetime/leakage candidates)\n"
            "2. Auto-extract datetime features and drop low-variance columns\n"
            "3. Split into train/test (random or time-aware) — or use user-supplied splits\n"
            "4. Impute, encode, scale\n"
            "5. Optionally log-transform the target\n"
            f"6. Train {len(out.results)} regressors with {out.options_used['cv_folds']}-fold CV\n"
            "7. Compute MAE / RMSE / R² / MedianAE / MAPE\n"
            "8. Pick best by lowest test RMSE\n"
            "9. Generate per-model and comparison charts\n"
            "10. Emit standalone `.py` script and `.ipynb` notebook"
        )

    # ---- Data & Preprocessing ----
    profile_rows = []
    for c in out.profile["columns"]:
        profile_rows.append({
            "column": c["name"], "kind": c["kind"], "dtype": c["dtype"],
            "missing": c["missing"], "missing_%": c["missing_pct"], "unique": c["unique"],
        })
    profile_df = pd.DataFrame(profile_rows)

    s = out.preprocessing.summary
    pre_md = (
        f"- **Train rows:** {s['n_train']:,} &nbsp;·&nbsp; "
        f"**Test rows:** {s['n_test']:,} &nbsp;·&nbsp; "
        f"**Val rows:** {s['n_val']:,}\n"
        f"- **Features after encoding:** {s['n_features_after']}\n"
        f"- **Numeric columns:** {len(s['numeric_cols'])}\n"
        f"- **One-hot encoded categoricals:** {len(s['low_cardinality_categorical'])}\n"
        f"- **Frequency-encoded high-cardinality:** "
        f"{len(s['high_cardinality_categorical_freq_encoded'])}\n"
        f"- **Split strategy:** {s['split_strategy_used']}\n"
        f"- **Target transform:** {s['target_transform']}\n"
    )
    if s.get("datetime_cols_extracted"):
        pre_md += f"- **Datetime features extracted from:** {s['datetime_cols_extracted']}\n"
    if s.get("low_variance_dropped"):
        pre_md += f"- **Low-variance columns dropped:** {s['low_variance_dropped']}\n"
    if s.get("id_like_columns_dropped"):
        pre_md += (f"- **ID/row-index columns auto-dropped (would leak if kept):** "
                   f"{s['id_like_columns_dropped']}\n")

    target_dist_fig = target_distribution_chart(
        out.preprocessing.y_train, out.options_used["target"]
    )

    eda_md = out.agent_outputs.get("eda") or (
        "*Agent commentary unavailable. Showing raw profile above.*\n\n"
        f"**Quick takeaways from the profile tool:**\n"
        f"- Target `{out.options_used['target']}` ranges from "
        f"{out.profile['target_stats']['min']:.3f} to "
        f"{out.profile['target_stats']['max']:.3f}, "
        f"skew = {out.profile['target_stats']['skew']:+.2f}\n"
        f"- {out.profile['missing_total']} missing values in total across all columns\n"
        f"- {len(out.profile['datetime_candidates'])} datetime candidate columns, "
        f"{len(out.profile['high_cardinality_columns'])} high-cardinality categoricals"
    )
    preproc_md = out.agent_outputs.get("preprocessor") or "*Agent commentary unavailable.*"

    # ---- Model comparison ----
    metrics_rows = []
    for n, r in out.results.items():
        m = r.metrics
        metrics_rows.append({
            "Model": n,
            "MAE":  round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "R²":   round(m["R2"], 4),
            "MedianAE": round(m["MedianAE"], 4),
            "MAPE %":   round(m.get("MAPE_pct", float("nan")), 2),
            "R² (train)":   round(m.get("R2_train", float("nan")), 4),
            "RMSE (train)": round(m.get("RMSE_train", float("nan")), 4),
            "CV R² mean":   round(m.get("CV_R2_mean", float("nan")), 4),
            "CV R² std":    round(m.get("CV_R2_std", float("nan")), 4),
            "Train time (s)": round(r.train_time_sec, 3),
        })
    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)

    # build a markdown header above the table indicating winners — avoids
    # the unreadable-highlight issue from the Streamlit version
    best_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
    best_mae  = metrics_df.loc[metrics_df["MAE"].idxmin(),  "Model"]
    best_r2   = metrics_df.loc[metrics_df["R²"].idxmax(),   "Model"]
    winners_md = (
        f"🏆 **Best by RMSE:** `{best_rmse}` &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Best by MAE:** `{best_mae}` &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Best by R²:** `{best_r2}`"
    )

    cc = comparison_charts(out.results, out.preprocessing.y_test)
    cc_keys = list(cc.keys())
    cc_figs = [cc[k] for k in cc_keys]
    # pad to fixed slots so Gradio outputs match
    while len(cc_figs) < 7:
        cc_figs.append(None)

    modeler_md = out.agent_outputs.get("modeler") or "*Agent commentary unavailable.*"

    # ---- Per-model dropdown ----
    model_names = list(out.results.keys())
    default_model = best
    pm_charts = per_model_charts(
        default_model, out.preprocessing.y_test,
        out.results[default_model].y_pred_test,
        out.results[default_model].cv_scores.get("R2"),
        out.results[default_model].feature_importances,
        out.preprocessing.feature_names,
    )
    pm_keys = list(pm_charts.keys())
    pm_figs = [pm_charts.get(k) for k in
               ["Predicted vs Actual", "Residuals vs Predicted",
                "Residual Distribution", "Q-Q Plot",
                "CV R² Distribution", "Feature Importance"]]

    pm_metric_html = _per_model_metric_html(default_model, out.results[default_model])

    # ---- Generated code ----
    tmp_dir = Path(tempfile.gettempdir()) / "regression_crew_outputs"
    tmp_dir.mkdir(exist_ok=True)
    py_path  = tmp_dir / "regression_pipeline.py"
    nb_path  = tmp_dir / "regression_pipeline.ipynb"
    py_path.write_text(out.generated_script)
    nb_path.write_text(out.generated_notebook)

    code_md = out.agent_outputs.get("code") or "*Agent commentary unavailable.*"

    # ---- Agent narrative ----
    nar_md = _build_narrative_md(out)

    return (
        gr.update(value=cards_html),                # overview cards
        gr.update(value=plan_md),                   # plan
        gr.update(value=df_preview.head(10)),       # data preview (re-shown for context)
        gr.update(value=profile_df),                # profile table
        gr.update(value=pre_md),                    # preprocessing summary
        gr.update(value=target_dist_fig),           # target distribution
        gr.update(value=eda_md),                    # EDA commentary
        gr.update(value=preproc_md),                # preprocessor commentary
        gr.update(value=metrics_df),                # metrics table
        gr.update(value=winners_md),                # winners line
        *[gr.update(value=f) for f in cc_figs],     # 7 comparison plots
        gr.update(value=modeler_md),                # modeler commentary
        gr.update(choices=model_names, value=default_model),  # model dropdown
        gr.update(value=pm_metric_html),            # per-model metric cards
        *[gr.update(value=f) for f in pm_figs],     # 6 per-model plots
        gr.update(value=out.generated_script),      # code preview
        gr.update(value=str(py_path)),              # py download
        gr.update(value=str(nb_path)),              # ipynb download
        gr.update(value=code_md),                   # code commentary
        gr.update(value=nar_md),                    # full narrative
    )


def _per_model_metric_html(name: str, result) -> str:
    m = result.metrics
    overfit = ""
    if "R2_train" in m and m["R2_train"] - m["R2"] > 0.15:
        overfit = (f'<div class="scope-warn" style="margin-top:8px;">⚠️ Possible overfitting: '
                   f'train R² = {m["R2_train"]:.3f}, test R² = {m["R2"]:.3f}</div>')
    if m.get("R2", 0) > 0.9999 and m.get("R2_train", 0) > 0.9999:
        overfit = ('<div class="scope-warn" style="margin-top:8px;">'
                   'ℹ️ R² ≈ 1.0 on both train AND test. This <i>can</i> mean '
                   'target leakage (a feature is a near-duplicate of the target), '
                   'but it can also just mean the dataset is highly predictable '
                   '(e.g. diamonds, where carat alone explains ~85% of price '
                   'variance). The Quality Reviewer in the Narrative tab does a '
                   'more careful check using feature-target correlations.'
                   '</div>')
    cv = ""
    if "CV_R2_mean" in m:
        cv = f"<small style='color:#6b7280;'>CV R²: {m['CV_R2_mean']:.4f} ± {m['CV_R2_std']:.4f}</small>"
    cards = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
      {_metric_card("Model", name, gold=True)}
      {_metric_card("RMSE", f"{m['RMSE']:.4f}")}
      {_metric_card("MAE",  f"{m['MAE']:.4f}")}
      {_metric_card("R²",   f"{m['R2']:.4f}")}
    </div>
    <div style="margin-top:6px;">{cv}</div>
    {overfit}
    """
    return cards


def on_model_dropdown_change(model_name: str, state: dict):
    out: PipelineOutput | None = (state or {}).get("output")
    if out is None or model_name not in out.results:
        return (gr.update(),) * 7
    res = out.results[model_name]
    charts = per_model_charts(
        model_name, out.preprocessing.y_test, res.y_pred_test,
        res.cv_scores.get("R2"), res.feature_importances,
        out.preprocessing.feature_names,
    )
    figs = [charts.get(k) for k in
            ["Predicted vs Actual", "Residuals vs Predicted",
             "Residual Distribution", "Q-Q Plot",
             "CV R² Distribution", "Feature Importance"]]
    return (gr.update(value=_per_model_metric_html(model_name, res)),
            *[gr.update(value=f) for f in figs])


def _build_narrative_md(out: PipelineOutput) -> str:
    if not out.agent_outputs:
        return ("*Agent narration was not run. Set `OPENAI_API_KEY` or "
                "`ANTHROPIC_API_KEY` and tick the agent checkbox in **Setup** to enable.*")
    sections = [
        ("📝 Insight Reporter — Executive summary", out.agent_outputs.get("insight", "")),
        ("🧐 Quality Reviewer — Verdict",          out.agent_outputs.get("quality", "")),
        ("🗺️ Planner — Plan",                      out.agent_outputs.get("planner", "")),
        ("🔬 EDA Analyst",                          out.agent_outputs.get("eda", "")),
        ("🧹 Preprocessor",                         out.agent_outputs.get("preprocessor", "")),
        ("🤖 Modeler & Evaluator",                  out.agent_outputs.get("modeler", "")),
        ("📊 Visualization Specialist",            out.agent_outputs.get("chart", "")),
        ("💻 Code Generator",                       out.agent_outputs.get("code", "")),
    ]
    parts = []
    for title, body in sections:
        if body.strip():
            parts.append(f"### {title}\n\n{body}\n")
    return "\n---\n".join(parts) if parts else "*No agent output captured.*"


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------
def build_ui():
    has_llm = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    llm_provider = ("OpenAI" if os.getenv("OPENAI_API_KEY")
                    else "Anthropic" if os.getenv("ANTHROPIC_API_KEY") else None)

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
                   css=CUSTOM_CSS, title="Regression Crew") as demo:
        state = gr.State(_empty_state())

        gr.HTML(
            '<div class="main-header">'
            '<h1>📈 Agentic Regression Analysis — 8 Agents, 1 Pipeline</h1>'
            '<p>Agentic end-to-end regression analysis — upload a CSV (or train/val/test), '
            'pick a target, explore models &amp; charts, and download reproducible code. '
            'Eight CrewAI agents profile your data, train and rank up to 11 models, '
            'audit for leakage and overfitting, and write a plain-language report — '
            'so you get both the numbers and the narrative.</p>'
            '</div>'
        )

        # Scope notice — always visible, sits above the tabs
        gr.HTML(
            '<div class="scope-notice">'
            '<span class="label">Scope Notice</span>'
            'Built for <b>standard tabular regression</b> on small-to-medium datasets. '
            'Adapts to time-series, group-aware CV, skewed targets, and basic leakage. '
            '<b>Not</b> a universal regression solver — see the <b>📜 Scope</b> tab for the full list.'
            '</div>'
        )

        # Status banner — updates as the pipeline runs (visible across all tabs)
        status_banner = gr.HTML(
            '<div class="status-banner idle">'
            '⏸️  <b>Idle</b> — configure in the <b>Setup</b> tab and click <b>🚀 Run Analysis</b>.'
            '</div>'
        )

        with gr.Tabs() as tabs:

            # =================== SETUP TAB ===================
            with gr.Tab("⚙️ Setup"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. Upload your data")
                        upload_mode = gr.Radio(
                            ["Single file (auto-split)", "Multiple files (train / val / test)"],
                            value="Single file (auto-split)", label="Upload mode",
                        )
                        with gr.Group(visible=True) as single_box:
                            single_file = gr.File(label="Training CSV",
                                                  file_types=[".csv"], type="filepath")
                        with gr.Group(visible=False) as multi_box:
                            train_file = gr.File(label="Training CSV (required)",
                                                 file_types=[".csv"], type="filepath")
                            val_file   = gr.File(label="Validation CSV (optional)",
                                                 file_types=[".csv"], type="filepath")
                            test_file  = gr.File(label="Test CSV (optional)",
                                                 file_types=[".csv"], type="filepath")

                        upload_info = gr.Markdown("*Upload a CSV to see a preview.*")

                        gr.Markdown("### 2. Dataset preview (first 10 rows)")
                        preview_df = gr.DataFrame(
                            value=None, label=None, interactive=False, wrap=True,
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 3. Choose target & options")
                        target_dd = gr.Dropdown(label="🎯 Target column (numeric)",
                                                choices=[], value=None, interactive=True)

                        with gr.Accordion("Optional: time / group columns", open=False):
                            time_dd  = gr.Dropdown(label="Time column (for time-aware split)",
                                                   choices=["(none)"], value="(none)")
                            group_dd = gr.Dropdown(label="Group column (for group-aware CV)",
                                                   choices=["(none)"], value="(none)")

                        gr.Markdown("### 4. Training settings")
                        with gr.Row():
                            test_size = gr.Slider(0.10, 0.40, value=0.20, step=0.05,
                                                  label="Test size (single-file mode)")
                            random_state = gr.Number(value=42, label="Random state",
                                                     precision=0)
                        with gr.Row():
                            n_folds = gr.Slider(2, 10, value=5, step=1,
                                                label="Cross-validation folds")
                            split_strategy = gr.Radio(
                                ["Random", "Time-aware"], value="Random",
                                label="Split strategy (single-file mode)",
                            )

                        all_models = available_model_names()
                        models_cb = gr.CheckboxGroup(
                            choices=all_models, value=all_models,
                            label="Models to train",
                        )

                        with gr.Accordion("Advanced options", open=False):
                            log_target_cb     = gr.Checkbox(False, label="Log-transform target (skewed targets)")
                            auto_dt_cb        = gr.Checkbox(True,  label="Auto-extract datetime features")
                            drop_lowvar_cb    = gr.Checkbox(True,  label="Drop low-variance columns")
                            drop_id_cb        = gr.Checkbox(True,  label="Auto-drop ID/row-index columns (recommended — prevents subtle leakage when data is sorted)")
                            tune_cb           = gr.Checkbox(False, label="Hyperparameter tuning (slower)")

                        gr.Markdown("### 5. CrewAI agent narration")
                        if has_llm:
                            gr.HTML(f'<div class="scope-good">✅ <b>{llm_provider}</b> '
                                    f'API key detected. Agents available.</div>')
                            use_agents_cb = gr.Checkbox(True, label="Run CrewAI agents")
                        else:
                            gr.HTML(
                                '<div class="scope-warn">🔑 No LLM key detected. The pipeline '
                                'still runs without agents. Set <code>OPENAI_API_KEY</code> or '
                                '<code>ANTHROPIC_API_KEY</code> in your environment to enable.</div>'
                            )
                            use_agents_cb = gr.Checkbox(False, label="Run CrewAI agents",
                                                        interactive=False)

                        run_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")

            # =================== OVERVIEW TAB ===================
            with gr.Tab("📊 Overview"):
                with gr.Row():
                    run_btn_overview = gr.Button(
                        "🚀 Run Analysis",
                        variant="primary", size="lg", scale=1,
                    )
                    gr.HTML(
                        '<div style="color:#94a3b8; font-size:0.88rem; padding:10px 12px; '
                        'line-height:1.5;">Run the full pipeline using the configuration '
                        'from the <b>Setup</b> tab. The timeline below streams live updates.</div>',
                    )

                gr.HTML('<div class="section-header">Pipeline Timeline</div>')
                timeline_display = gr.HTML(
                    '<div class="timeline-container">'
                    '<div class="timeline-empty">Waiting to start — click <b>🚀 Run Analysis</b> '
                    'above (or in the Setup tab) to begin.</div>'
                    '</div>'
                )

                gr.HTML('<div class="section-header">Best Model Summary</div>')
                overview_cards = gr.HTML()

                gr.HTML('<div class="section-header">Plan</div>')
                overview_plan  = gr.Markdown()

            # =================== DATA & PREPROCESSING TAB ===================
            with gr.Tab("🧹 Data"):
                gr.Markdown("### Dataset preview (first 10 rows)")
                data_preview = gr.DataFrame(value=None, interactive=False, wrap=True)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Column profile")
                        profile_table = gr.DataFrame(interactive=False, wrap=True)
                    with gr.Column():
                        gr.Markdown("### Preprocessing summary")
                        preproc_summary = gr.Markdown()
                        gr.Markdown("### Target distribution")
                        target_dist_plot = gr.Plot()

                with gr.Accordion("🔬 EDA Agent commentary — what the agent found", open=True):
                    eda_commentary = gr.Markdown()
                with gr.Accordion("🧹 Preprocessor Agent commentary — what was done and why", open=False):
                    preproc_commentary = gr.Markdown()

            # =================== MODEL COMPARISON TAB ===================
            with gr.Tab("🏁 Models"):
                comparison_winners = gr.Markdown()
                gr.HTML('<div class="section-header">All models — metrics</div>')
                metrics_table = gr.DataFrame(interactive=False, wrap=True)

                gr.HTML('<div class="section-header">Comparison charts</div>')
                # One chart per row at full width — bar charts with model names
                # need horizontal space, and the legends on grouped charts get
                # cut off when squeezed into a 2-column grid.
                cmp_plot1 = gr.Plot()
                cmp_plot2 = gr.Plot()
                cmp_plot3 = gr.Plot()
                cmp_plot4 = gr.Plot()
                cmp_plot5 = gr.Plot()
                cmp_plot6 = gr.Plot()
                cmp_plot7 = gr.Plot()

                with gr.Accordion("🤖 Modeler Agent commentary", open=False):
                    modeler_commentary = gr.Markdown()

            # =================== PER-MODEL TAB ===================
            with gr.Tab("🔍 Per-Model"):
                gr.HTML('<div class="section-header">Pick a model to inspect</div>')
                model_dropdown = gr.Dropdown(choices=[], value=None, interactive=True,
                                             label="Model")
                pm_metrics = gr.HTML()
                with gr.Row():
                    pm_plot1 = gr.Plot()
                    pm_plot2 = gr.Plot()
                with gr.Row():
                    pm_plot3 = gr.Plot()
                    pm_plot4 = gr.Plot()
                with gr.Row():
                    pm_plot5 = gr.Plot()
                    pm_plot6 = gr.Plot()

            # =================== CODE TAB ===================
            with gr.Tab("💻 Code"):
                gr.Markdown(
                    "### Reproducible code\n\n"
                    "Drop your CSV next to either file (or update `TRAIN_CSV` at the top of the "
                    "script), and the pipeline reproduces end-to-end. Runs in plain Jupyter, "
                    "VS Code, or Cursor."
                )
                with gr.Row():
                    py_download = gr.File(label="📜 regression_pipeline.py", interactive=False)
                    nb_download = gr.File(label="📓 regression_pipeline.ipynb", interactive=False)
                code_preview = gr.Code(language="python", label="Generated Python script", lines=25)
                with gr.Accordion("💻 Code Generator Agent commentary", open=False):
                    code_commentary = gr.Markdown()

            # =================== AGENT NARRATIVE TAB ===================
            with gr.Tab("🤖 Narrative"):
                narrative_md = gr.Markdown()

            # =================== SCOPE TAB ===================
            with gr.Tab("📜 Scope"):
                gr.HTML(_SCOPE_HTML)

        # -------- wire up events --------
        upload_mode.change(on_mode_change, [upload_mode], [single_box, multi_box])

        for f in [single_file, train_file, val_file, test_file]:
            f.change(
                on_files_uploaded,
                [upload_mode, single_file, train_file, val_file, test_file, state],
                [state, preview_df, upload_info, target_dd, time_dd, group_dd],
            )

        run_outputs = [
            status_banner, state, timeline_display,
            # Overview
            overview_cards, overview_plan,
            # Data & Preprocessing
            data_preview, profile_table, preproc_summary, target_dist_plot,
            eda_commentary, preproc_commentary,
            # Comparison
            metrics_table, comparison_winners,
            cmp_plot1, cmp_plot2, cmp_plot3, cmp_plot4,
            cmp_plot5, cmp_plot6, cmp_plot7,
            modeler_commentary,
            # Per-model
            model_dropdown, pm_metrics,
            pm_plot1, pm_plot2, pm_plot3, pm_plot4, pm_plot5, pm_plot6,
            # Code
            code_preview, py_download, nb_download, code_commentary,
            # Narrative
            narrative_md,
        ]

        run_inputs = [
            upload_mode, target_dd, time_dd, group_dd,
            test_size, random_state, n_folds,
            models_cb, split_strategy, log_target_cb,
            auto_dt_cb, drop_lowvar_cb, drop_id_cb, tune_cb, use_agents_cb, state,
        ]

        # Wire both run buttons (Setup tab and Overview tab) to the same handler
        run_btn.click(on_run, run_inputs, run_outputs)
        run_btn_overview.click(on_run, run_inputs, run_outputs)

        model_dropdown.change(
            on_model_dropdown_change,
            [model_dropdown, state],
            [pm_metrics, pm_plot1, pm_plot2, pm_plot3, pm_plot4, pm_plot5, pm_plot6],
        )

    return demo


# ---------------------------------------------------------------------------
# Scope / Disclaimer content
# ---------------------------------------------------------------------------
_SCOPE_HTML = """
<div style="max-width: 1100px; margin: 0 auto; padding: 8px 4px;">

<!-- Header -->
<div style="background: rgba(20,20,38,0.6); border: 1px solid rgba(139,92,246,0.25);
            padding: 24px 28px; border-radius: 12px;
            color: #e2e8f0; margin-bottom: 18px; backdrop-filter: blur(4px);">
  <h1 style="color: #c4b5fd; margin: 0 0 8px 0; font-size: 1.7rem;">📜 Scope &amp; Disclaimer</h1>
  <p style="color: #94a3b8; margin: 0; font-size: 1rem; line-height: 1.6;">
    An honest reckoning of what this project handles well, what it handles with
    caveats, and what it is not designed for.
  </p>
</div>

<!-- WORKS WELL -->
<div style="background: rgba(34,197,94,0.07); border: 1px solid rgba(34,197,94,0.25);
            border-left: 6px solid #22c55e; border-radius: 10px;
            padding: 22px 28px; margin-bottom: 16px;">
  <h2 style="color: #86efac; margin: 0 0 14px; font-size: 1.3rem;">✅ Works well for</h2>
  <ul style="color: #d1fae5; line-height: 1.85; margin: 0; padding-left: 22px;">
    <li><b style="color: #ffffff;">Standard tabular regression</b> — numeric target, mix of numeric and categorical features</li>
    <li><b style="color: #ffffff;">Independent rows</b> — each row is its own observation</li>
    <li><b style="color: #ffffff;">Reasonable size</b> — a few hundred to ~100k rows, up to a few hundred features</li>
    <li><b style="color: #ffffff;">Some missing values</b> — gets median (numeric) or mode (categorical) imputation</li>
    <li><b style="color: #ffffff;">Mixed feature types</b> including high-cardinality categoricals (frequency-encoded)</li>
    <li><b style="color: #ffffff;">Datetime columns</b> — automatically decomposed into year/month/day/weekday/hour</li>
    <li><b style="color: #ffffff;">User-supplied train/val/test splits</b> — no random split applied when you upload three files</li>
  </ul>
  <p style="color: #d1fae5; margin-top: 14px; line-height: 1.65;">
    Covers the majority of business and scientific regression problems: house prices,
    sales (with independent observations), customer metrics, sensor readings,
    biological measurements, exam scores.
  </p>
</div>

<!-- USE WITH CAUTION -->
<div style="background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.25);
            border-left: 6px solid #f59e0b; border-radius: 10px;
            padding: 22px 28px; margin-bottom: 16px;">
  <h2 style="color: #fbbf24; margin: 0 0 14px; font-size: 1.3rem;">⚠️ Use with caution for</h2>
  <ul style="color: #fde68a; line-height: 1.85; margin: 0; padding-left: 22px;">
    <li><b style="color: #ffffff;">Time series data</b> — even with the time-aware split option, this is not a forecasting framework. Use <code style="background:rgba(245,158,11,0.2);padding:2px 7px;border-radius:4px;color:#fde68a;font-size:0.9em;">statsmodels</code>, <code style="background:rgba(245,158,11,0.2);padding:2px 7px;border-radius:4px;color:#fde68a;font-size:0.9em;">Prophet</code>, <code style="background:rgba(245,158,11,0.2);padding:2px 7px;border-radius:4px;color:#fde68a;font-size:0.9em;">sktime</code>, or <code style="background:rgba(245,158,11,0.2);padding:2px 7px;border-radius:4px;color:#fde68a;font-size:0.9em;">darts</code> for ARIMA / state-space / exponential-smoothing. The time-aware split only ensures evaluation respects time order; it does not engineer lag features.</li>
    <li><b style="color: #ffffff;">Highly skewed targets</b> — turn on the <b style="color:#ffffff;">log-transform target</b> option, but inspect residuals on the original scale. Income, claims, traffic data often need this.</li>
    <li><b style="color: #ffffff;">Grouped data</b> — multiple rows per patient / store / user. Use the <b style="color:#ffffff;">group column</b> option so CV respects groups, otherwise CV scores will be optimistically biased.</li>
    <li><b style="color: #ffffff;">Wide datasets</b> (features ≫ rows) — one-hot encoding can blow up dimensions. Stick to regularized models (Ridge / Lasso / ElasticNet).</li>
    <li><b style="color: #ffffff;">Tiny datasets</b> (&lt; 50 rows) — 5-fold CV becomes unreliable. Lower the fold count and treat results with skepticism.</li>
    <li><b style="color: #ffffff;">R² ≈ 1.0 on every tree-based model</b> — usually means target leakage (a feature is a near-duplicate of the target), <i style="color:#fde68a;">but not always</i>. Some datasets like diamonds — where carat alone explains ~85% of price variance — are genuinely that predictable. The Quality Reviewer checks both feature-target correlation AND model perfection. Treat its verdict as advisory, not gospel.</li>
  </ul>
</div>

<!-- NOT DESIGNED FOR -->
<div style="background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.25);
            border-left: 6px solid #ef4444; border-radius: 10px;
            padding: 22px 28px; margin-bottom: 16px;">
  <h2 style="color: #fca5a5; margin: 0 0 14px; font-size: 1.3rem;">❌ Not designed for</h2>
  <ul style="color: #fecaca; line-height: 1.85; margin: 0; padding-left: 22px;">
    <li><b style="color: #ffffff;">Classification</b> — the target must be numeric. There is a hard check that rejects non-numeric targets.</li>
    <li><b style="color: #ffffff;">Multi-output regression</b> — only single-target.</li>
    <li><b style="color: #ffffff;">Survival analysis</b> (time-to-event with censoring) — use <code style="background:rgba(239,68,68,0.2);padding:2px 7px;border-radius:4px;color:#fecaca;font-size:0.9em;">lifelines</code> or <code style="background:rgba(239,68,68,0.2);padding:2px 7px;border-radius:4px;color:#fecaca;font-size:0.9em;">scikit-survival</code>.</li>
    <li><b style="color: #ffffff;">Quantile regression</b> — predicting intervals rather than point estimates.</li>
    <li><b style="color: #ffffff;">Image, text, audio, or graph data</b> — tabular only. No CNNs, transformers, or graph neural nets.</li>
    <li><b style="color: #ffffff;">Datasets too large for memory</b> — no batching or out-of-core training. Use <code style="background:rgba(239,68,68,0.2);padding:2px 7px;border-radius:4px;color:#fecaca;font-size:0.9em;">dask-ml</code> or <code style="background:rgba(239,68,68,0.2);padding:2px 7px;border-radius:4px;color:#fecaca;font-size:0.9em;">vaex</code>.</li>
    <li><b style="color: #ffffff;">Production deployment</b> — generates a <code style="background:rgba(239,68,68,0.2);padding:2px 7px;border-radius:4px;color:#fecaca;font-size:0.9em;">joblib</code> artifact, but no serving infrastructure, monitoring, or model registry.</li>
    <li><b style="color: #ffffff;">Causal inference</b> — these models tell you what is correlated, not what is causal. For uplift modeling or treatment effects use <code style="background:rgba(239,68,68,0.2);padding:2px 7px;border-radius:4px;color:#fecaca;font-size:0.9em;">econml</code> or <code style="background:rgba(239,68,68,0.2);padding:2px 7px;border-radius:4px;color:#fecaca;font-size:0.9em;">dowhy</code>.</li>
  </ul>
</div>

<!-- WHAT THE AGENTS DO -->
<div style="background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.3);
            border-left: 6px solid #6366f1; border-radius: 10px;
            padding: 22px 28px; margin-bottom: 16px;">
  <h2 style="color: #a5b4fc; margin: 0 0 14px; font-size: 1.3rem;">🔧 What the agents do (and do not do)</h2>
  <p style="color: #c7d2fe; line-height: 1.7; margin: 0 0 12px;">
    The CrewAI agents <b style="color:#ffffff;">narrate and review</b> a pipeline that runs deterministically.
    They do not write ML code on the fly — every metric, model, and chart you see was
    produced by tested Python utilities. The agents:
  </p>
  <ul style="color: #c7d2fe; line-height: 1.85; margin: 0; padding-left: 22px;">
    <li>Plan the run</li>
    <li>Comment on EDA findings with specific column names and percentages</li>
    <li>Explain preprocessing choices</li>
    <li>Rank models and call out overfitting</li>
    <li>Audit for leakage and dimensional issues</li>
    <li>Write an executive summary</li>
  </ul>
  <p style="color: #c7d2fe; line-height: 1.7; margin: 12px 0 0;">
    If you set no LLM key, the pipeline still runs end-to-end. You just get the
    deterministic outputs without the agent commentary.
  </p>
</div>

<!-- WHAT I WOULD ADD -->
<div style="background: rgba(139,92,246,0.07); border: 1px solid rgba(139,92,246,0.3);
            border-left: 6px solid #8b5cf6; border-radius: 10px;
            padding: 22px 28px;">
  <h2 style="color: #c4b5fd; margin: 0 0 14px; font-size: 1.3rem;">🛣️ What I would add to make it more universal</h2>
  <ul style="color: #ddd6fe; line-height: 1.85; margin: 0; padding-left: 22px;">
    <li>Time-series-native models (ARIMA / Prophet / NeuralForecast)</li>
    <li>Optuna for smarter hyperparameter search</li>
    <li>Quantile regression / prediction intervals</li>
    <li>Automated feature interactions (polynomial features, target encoding)</li>
    <li>Weighted-loss objectives (Tweedie, Poisson) for count and claims data</li>
    <li>Drift detection on the test split</li>
  </ul>
</div>

</div>
"""


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True, show_api=False)
