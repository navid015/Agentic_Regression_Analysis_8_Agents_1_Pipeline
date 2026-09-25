"""
Advisor: the part of the "agentic" pipeline that ACTS.

Previously every agent ran after the pipeline had finished and could only
narrate. Here an advisor (deterministic rules, optionally plus an LLM)
proposes concrete changes from a whitelisted action set; the orchestrator
re-runs the pipeline with them and keeps a change only if the winner's
cross-validated error improves by more than the corrected fold-to-fold noise.

Safety rules
------------
* Only whitelisted actions with validated values are executed - an LLM can
  never run code or change anything else.
* Rows never change between rounds, so the evaluation folds are identical
  and the comparison is paired.
* The selection metric is locked for the whole loop (switching RMSE <-> MAE
  would make "better" meaningless).
* Dropping columns (suspected leakage) is NEVER automatic: removing a real
  leak always makes CV worse, so a CV-driven loop would refuse it, and a
  very predictive column is not always a leak. Such proposals are returned
  as "needs your confirmation".
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from utils.modeling import BASELINE_MODEL_NAME, _fold_errors, corrected_paired_difference, available_model_names

AUTO_ACTIONS = {
    "tuning": {"fast", "thorough"},
    "add_interactions": {True, False},
    "add_missing_indicators": {True, False},
    "high_cardinality_encoding": {"target", "frequency"},
    "log_transform_target": {"auto", True, False},
    "treat_as_categorical": "columns",
    "build_ensemble": {True, False},
}
CONFIRM_ACTIONS = {"drop_columns": "columns"}
MODEL_ACTION = "selected_models"


@dataclass
class Proposal:
    action: str
    value: Any
    reason: str
    source: str = "rules"
    auto_ok: bool = True

    def as_dict(self):
        return asdict(self)


def _winner_cv(out) -> float:
    M = out.selection_metric.upper()
    return float(out.results[out.best_model].metrics.get(f"CV_{M}_mean", np.inf))


def _columns(out) -> set[str]:
    cols = set(map(str, out.run_inputs["df_train"].columns)) if out.run_inputs else set()
    return cols - {out.options_used.get("target")}


def rule_based_proposals(out) -> list[Proposal]:
    opts, s = out.run_options, out.preprocessing.summary
    res = out.results
    props: list[Proposal] = []
    if (opts.get("tuning") or "off") == "off" and s["n_train"] <= 50_000:
        props.append(Proposal("tuning", "fast", "hyperparameters were not tuned"))
    lin = [n for n, r in res.items() if r.family == "linear" and np.isfinite(r.metrics.get("CV_RMSE_mean", np.nan))]
    if lin and not opts.get("add_interactions") and 2 <= len(s["numeric_cols"]) <= 15:
        M = out.selection_metric.upper()
        best_lin = min(res[n].metrics[f"CV_{M}_mean"] for n in lin)
        if best_lin <= _winner_cv(out) * 1.10:
            props.append(Proposal("add_interactions", True,
                                  "a linear model is (nearly) the best - pairwise interactions may "
                                  "capture effects it currently misses"))
    if s.get("high_cardinality_categorical") and s.get("high_cardinality_encoding") == "frequency":
        props.append(Proposal("high_cardinality_encoding", "target",
                              "frequency encoding discards category identity"))
    if out.profile.get("missing_total", 0) > 0 and not opts.get("add_missing_indicators", True):
        props.append(Proposal("add_missing_indicators", True, "missingness is often informative"))
    if opts.get("log_transform_target") is False and s.get("log_candidate"):
        props.append(Proposal("log_transform_target", "auto",
                              f"target is non-negative and skewed ({s['target_skew_train']:+.2f})"))
    for sus in out.leakage_suspects or []:
        if sus["strong"]:
            props.append(Proposal("drop_columns", [sus["column"]], f"possible target leakage: {sus['reason']}",
                                  auto_ok=False))
    return props


_LLM_PROMPT = """You are a senior ML engineer improving a tabular REGRESSION pipeline.
A deterministic pipeline already ran. Propose at most 3 changes that could lower the
cross-validated error. You may ONLY use these actions (anything else is ignored):
- {{"action": "tuning", "value": "fast" | "thorough"}}
- {{"action": "add_interactions", "value": true | false}}
- {{"action": "add_missing_indicators", "value": true | false}}
- {{"action": "high_cardinality_encoding", "value": "target" | "frequency"}}
- {{"action": "log_transform_target", "value": "auto" | true | false}}
- {{"action": "treat_as_categorical", "value": ["column", ...]}}   (integer codes that are categories)
- {{"action": "selected_models", "value": ["Ridge", "RandomForest", ...]}} (available names only; baseline is always included)
- {{"action": "build_ensemble", "value": true | false}}
- {{"action": "drop_columns", "value": ["column", ...]}}   (ONLY for probable target leakage; a human confirms)
Each action is a separate experiment, evaluated on the locked CV protocol. Propose a hypothesis and reason for every action. Never request test labels or a different split.
Reply with JSON only: {{"actions": [{{"action": ..., "value": ..., "reason": "..."}}]}}

Available columns: {columns}
Current options: {options}

{context}
"""


def _parse_llm_actions(text: str, valid_columns: set[str]) -> list[Proposal]:
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for a in (data.get("actions") or [])[:3]:
        if not isinstance(a, dict):
            continue
        act, val, why = a.get("action"), a.get("value"), str(a.get("reason", ""))[:300]
        allowed = AUTO_ACTIONS.get(act, CONFIRM_ACTIONS.get(act))
        if act == MODEL_ACTION:
            allowed = "models"
        if allowed is None:
            continue
        if allowed == "models":
            if not isinstance(val, list) or not val or any(not isinstance(n, str) or n not in available_model_names() for n in val):
                continue
            val = list(dict.fromkeys(val))
        elif allowed == "columns":
            if not isinstance(val, list):
                continue
            cols = [str(c) for c in val if str(c) in valid_columns]
            if not cols:
                continue
            val = cols
        elif val not in allowed:
            continue
        out.append(Proposal(act, val, why or "suggested by the LLM advisor", source="llm",
                            auto_ok=act in AUTO_ACTIONS or act == MODEL_ACTION))
    return out


def llm_proposals(out, llm_call: Callable[[str], str], context: str) -> list[Proposal]:
    safe_opts = {k: v for k, v in out.run_options.items()
                 if isinstance(v, (bool, int, float, str, type(None), list))}
    prompt = _LLM_PROMPT.format(columns=sorted(_columns(out))[:200], options=json.dumps(safe_opts, default=str),
                                context=context[:6000])
    try:
        return _parse_llm_actions(llm_call(prompt), _columns(out))
    except Exception:
        return []


def apply_proposals(options: dict, proposals: list[Proposal]) -> dict:
    new = dict(options)
    for p in proposals:
        if p.action == "treat_as_categorical":
            new["categorical_columns"] = sorted(set(new.get("categorical_columns") or []) | set(p.value))
        elif p.action == "drop_columns":
            new["drop_columns"] = sorted(set(new.get("drop_columns") or []) | set(p.value))
        else:
            new[p.action] = p.value
    return new


def _already_applied(options: dict, p: Proposal) -> bool:
    if p.action == "treat_as_categorical":
        return set(p.value) <= set(options.get("categorical_columns") or [])
    if p.action == "drop_columns":
        return set(p.value) <= set(options.get("drop_columns") or [])
    return options.get(p.action) == p.value


def _improves(new_out, old_out) -> tuple[bool, str]:
    M = old_out.selection_metric.upper()
    old_cv, new_cv = _winner_cv(old_out), _winner_cv(new_out)
    a = _fold_errors(old_out.results[old_out.best_model], M)
    b = _fold_errors(new_out.results[new_out.best_model], M)
    ratio = old_out.results[old_out.best_model].metrics.get("CV_test_train_ratio", 0.25)
    if a is not None and b is not None and len(a) == len(b):
        gain, se, _ = corrected_paired_difference(a, b, ratio)
        ok = bool(gain > max(se, 0.005 * old_cv))
        return ok, (f"winner CV {M} {old_cv:.4g} \u2192 {new_cv:.4g} "
                    f"(paired gain {gain:.4g} vs noise {se:.3g})")
    ok = bool(new_cv < old_cv * 0.99)
    return ok, f"winner CV {M} {old_cv:.4g} \u2192 {new_cv:.4g}"


def refine(run_fn: Callable[[dict], Any], first_out, *, rounds: int = 1,
           llm_call: Callable[[str], str] | None = None, context_fn: Callable[[Any], str] | None = None,
           progress: Callable[[str, str], None] | None = None):
    """Propose -> re-run -> keep only improvements. Returns the best output,
    with `refinement_log` and `pending_confirmation` filled in."""
    best = first_out
    log: list[dict] = []
    pending: list[dict] = []
    locked_metric = first_out.selection_metric
    tried: set[str] = set()
    for rnd in range(1, min(max(rounds, 0), 20) + 1):
        props = rule_based_proposals(best)
        if llm_call is not None:
            props += llm_proposals(best, llm_call, context_fn(best) if context_fn else "")
        uniq, seen = [], set()
        for p in props:
            key = (p.action, json.dumps(p.value, default=str))
            if key not in seen and key not in tried and not _already_applied(best.run_options, p):
                seen.add(key)
                uniq.append(p)
        for p in uniq:
            if not p.auto_ok and all(d["action"] != p.action or d["value"] != p.value for d in pending):
                pending.append(p.as_dict())
        auto = [p for p in uniq if p.auto_ok]
        if not auto:
            break
        attempts = [[p] for p in auto[:3]]
        accepted = False
        for group in attempts:
            for p in group:
                tried.add((p.action, json.dumps(p.value, default=str)))
            if progress:
                progress("refine", f"Refinement round {rnd}: trying {[(p.action, p.value) for p in group]}")
            opts = apply_proposals(best.run_options, group)
            opts["selection_metric"] = locked_metric
            try:
                cand = run_fn(opts)
            except Exception as e:
                log.append({"round": rnd, "source": "/".join(sorted({p.source for p in group})),
                            "actions": [(p.action, p.value) for p in group], "accepted": False,
                            "reason": f"re-run failed: {type(e).__name__}: {e}"})
                continue
            ok, why = _improves(cand, best)
            log.append({"round": rnd, "source": "/".join(sorted({p.source for p in group})),
                        "actions": [(p.action, p.value) for p in group], "accepted": ok,
                        "reason": why, "why_proposed": [p.reason for p in group]})
            if ok:
                best, accepted = cand, True
                break
        # Record rejected ideas and continue to the next hypothesis, within the budget.
        if not accepted and llm_call is None:
            break
    best.refinement_log = (first_out.refinement_log or []) + log
    best.pending_confirmation = pending
    if best is not first_out and best.best_model == BASELINE_MODEL_NAME:
        best.warnings = list(best.warnings or []) + ["refinement kept a change but no reliable signal was found"]
    return best
