"""
Audits that turn "look at the charts" into numbers.

* `leakage_suspects` - the old audit only flagged a NUMERIC column with
  Pearson |r| > 0.99. A perfect leak such as log(target) has Pearson r ~ 0.92
  on a skewed target and passed. Here every column gets a rank correlation
  and a single-feature model R2 (a shallow tree for numbers, cross-fitted
  target encoding for categories), which catches monotone and non-linear
  leaks. Suspects are REPORTED; dropping one is always the user's decision,
  because genuinely deterministic data exist.
* `residual_diagnostics` - bias, heteroscedasticity, skew, tail share and
  (for time-ordered data) autocorrelation of the winner's test residuals,
  with plain-English findings. The chart agent reads these instead of
  guessing what a plot it cannot see might show.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import TargetEncoder
from sklearn.tree import DecisionTreeRegressor

from utils.preprocessing import _as_category_strings, _name_tokens

LEAK_R2_STRONG = 0.98
LEAK_R2_REPORT = 0.90
LEAK_RANK_STRONG = 0.995


def leakage_suspects(X_raw: pd.DataFrame, y: np.ndarray, target_name: str, *,
                     exclude: list[str] | None = None, max_rows: int = 5000,
                     max_cols: int = 200, random_state: int = 42) -> list[dict[str, Any]]:
    """Columns that, ON THEIR OWN, predict the target almost perfectly."""
    if X_raw is None or len(X_raw) < 30:
        return []
    y = np.asarray(y, float)
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y)) if len(y) <= max_rows else np.sort(rng.choice(len(y), max_rows, replace=False))
    Xs, ys = X_raw.iloc[idx], y[idx]
    target_tokens = _name_tokens(target_name)
    cv = KFold(3, shuffle=True, random_state=random_state)
    out = []
    for col in [c for c in X_raw.columns if c not in set(exclude or [])][:max_cols]:
        s = Xs[col]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if pd.api.types.is_numeric_dtype(s):
                    v = s.astype(float)
                    v = v.fillna(v.median() if v.notna().any() else 0.0).to_numpy().reshape(-1, 1)
                    rank = abs(float(stats.spearmanr(v.ravel(), ys).statistic))
                    pred = cross_val_predict(DecisionTreeRegressor(max_depth=6, min_samples_leaf=5,
                                                                   random_state=random_state),
                                             v, ys, cv=cv)
                else:
                    rank = float("nan")
                    cats = _as_category_strings(s).to_frame()
                    pred = TargetEncoder(target_type="continuous", random_state=random_state) \
                        .fit_transform(cats, ys).ravel()
                r2 = float(r2_score(ys, pred))
        except Exception:
            continue
        name_hint = bool(target_tokens and target_tokens <= _name_tokens(col))
        strong = r2 >= LEAK_R2_STRONG or (np.isfinite(rank) and rank >= LEAK_RANK_STRONG) \
            or (name_hint and r2 >= LEAK_R2_REPORT)
        if strong or r2 >= LEAK_R2_REPORT:
            why = [f"alone predicts the target with R\u00b2 {r2:.3f}"]
            if np.isfinite(rank) and rank >= 0.9:
                why.append(f"rank correlation {rank:.3f}")
            if name_hint:
                why.append(f"its name contains the target name '{target_name}'")
            out.append({"column": col, "single_feature_r2": r2, "rank_corr": rank,
                        "name_hint": name_hint, "strong": bool(strong), "reason": "; ".join(why)})
    return sorted(out, key=lambda d: -d["single_feature_r2"])


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, *,
                         time_ordered: bool = False) -> dict[str, Any]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r = y_true - y_pred
    ok = np.isfinite(r)
    r, yt, yp = r[ok], y_true[ok], y_pred[ok]
    if len(r) < 10:
        return {"findings": ["too few test rows for residual diagnostics"]}
    sd = float(np.std(r)) or 1e-12
    scale = float(np.mean(np.abs(yt))) or 1.0
    if np.ptp(yp) > 0 and np.ptp(np.abs(r)) > 0:
        het = stats.spearmanr(yp, np.abs(r))
        het_stat, het_p = float(het.statistic), float(het.pvalue)
    else:                       # constant predictions (e.g. the baseline won)
        het_stat, het_p = 0.0, 1.0
    d: dict[str, Any] = {
        "n": int(len(r)),
        "mean_residual": float(np.mean(r)),
        "bias_pct_of_mean_target": float(100 * np.mean(r) / scale),
        "hetero_spearman": het_stat, "hetero_p": het_p,
        "residual_skew": float(stats.skew(r)),
        "residual_excess_kurtosis": float(stats.kurtosis(r)),
        "share_beyond_2sd": float(np.mean(np.abs(r - r.mean()) > 2 * sd)),
    }
    findings = []
    t_bias = stats.ttest_1samp(r, 0.0)
    if t_bias.pvalue < 0.01 and abs(d["bias_pct_of_mean_target"]) > 1:
        findings.append(f"Predictions are biased: on average the model is off by "
                        f"{d['mean_residual']:+.4g} ({d['bias_pct_of_mean_target']:+.1f}% of the mean "
                        f"target){' - it under-predicts' if d['mean_residual'] > 0 else ' - it over-predicts'}.")
    if het_p < 0.01 and abs(het_stat) > 0.15:
        findings.append("Error size grows with the predicted value (heteroscedasticity): the "
                        "Residuals-vs-Predicted chart fans out. A log target or an adaptive "
                        "prediction interval is appropriate.")
    if abs(d["residual_skew"]) > 1:
        findings.append(f"Residuals are skewed ({d['residual_skew']:+.2f}); the Q-Q plot will bend "
                        "away from the line at one end.")
    if d["residual_excess_kurtosis"] > 3 or d["share_beyond_2sd"] > 0.08:
        findings.append(f"Heavy tails: {d['share_beyond_2sd']:.1%} of residuals lie beyond 2 standard "
                        "deviations (about 5% expected for normal errors) - a few large misses dominate RMSE.")
    if time_ordered and len(r) > 20:
        dw = float(np.sum(np.diff(r) ** 2) / np.sum(r ** 2))
        d["durbin_watson"] = dw
        if dw < 1.5:
            findings.append(f"Consecutive test errors are correlated (Durbin-Watson {dw:.2f} < 1.5): "
                            "the model misses persistent patterns; lag features would help.")
    if not findings:
        findings.append("No systematic problem in the residuals: no bias, no fanning, near-normal tails.")
    d["findings"] = findings
    return d
