"""
Preprocessing for regression.

Supports a single CSV (random, group-aware or time-aware split) or user-supplied
train / (val) / test files.

What changed in v3 and why
--------------------------
* **Row-counter detection instead of "sorted unique integers".** The old rule
  dropped ANY unique, integer, monotonic column, which deletes the key feature
  of every CSV that was exported sorted by a real integer (square footage, a
  year column in an annual series). Now only dense runs that start at 0 or 1
  (a real row counter) are dropped automatically; other suspicious columns are
  reported and kept.
* **Formatted numbers are parsed.** "$1,250", "12.5%", "(300)" and "1 234" used
  to become high-cardinality categoricals. They are converted to numbers.
* **Target encoding for high-cardinality categoricals.** Frequency encoding
  maps two categories with similar counts to the same value and throws away
  their identity. sklearn's cross-fitted TargetEncoder keeps the signal
  without leaking the row's own target.
* **Missing categories stay visible.** Categorical NaN becomes its own
  "__missing__" level instead of being silently replaced by the mode.
* **Numeric-coded categoricals** (zip / postcode / class / type codes) are
  treated as categories, and the user can name more.
* **Time-aware mode sorts every file**, including user-supplied training
  files, so TimeSeriesSplit folds really run forward in time. Cut points never
  split a timestamp between train and test.
* **Cross-file checks**: exact duplicate rows shared by train and test, and
  groups present in both, are counted and reported.
* **In-fold preprocessing**: the result carries the unfitted ColumnTransformer
  templates and the engineered raw frames, so modelling can fit the
  preprocessing INSIDE every CV fold (no imputation / scaling / encoding
  statistics from held-out rows).
* Interaction features are built only into the template used by linear models.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
    TargetEncoder,
)

HIGH_CARDINALITY_THRESHOLD = 20
LOW_VARIANCE_NUNIQUE = 1
NEAR_CONSTANT_FREQ = 0.999
#: library-level heuristic for log_transform_target="auto" (the orchestrator
#: additionally verifies the choice with cross-validation)
LOG_TRANSFORM_SKEW_THRESHOLD = 1.5
#: skew above which the orchestrator runs the CV comparison of raw vs log1p
LOG_CANDIDATE_SKEW = 1.0
MAX_INTERACTION_BASE_COLUMNS = 15
MISSING_TOKEN = "__missing__"
#: share of values that must parse for a text column to be treated as numeric
NUMERIC_STRING_MIN_SHARE = 0.95
#: row counter: unique integers starting at 0/1 with at most this much slack
ROW_COUNTER_DENSITY = 1.10

ID_COLUMN_NAME_TOKENS = frozenset({
    "id", "ids", "index", "idx", "rowid", "row", "rownum", "rownumber",
    "key", "pk", "uuid", "guid", "serial", "seq",
})
ID_COLUMN_NAME_PREFIXES = ("unnamed:",)
#: integer columns whose NAME says they are codes, not quantities
CATEGORICAL_CODE_TOKENS = frozenset({
    "zip", "zipcode", "postcode", "postal", "fips", "code", "category",
    "class", "subclass", "type",
})
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_RE_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NUM_NOISE_PATTERN = r"[\s,$€£¥₹%+']"
_ACCOUNTING_NEG_PATTERN = r"^\((.*)\)$"


# ---- small helpers ----------------------------------------------------------

def _name_tokens(col: str) -> set[str]:
    """Tokens of a column name, splitting on punctuation AND camelCase.

    'customerId' -> {customer, id};  'MSSubClass' -> {ms, sub, class}.
    """
    text = _CAMEL_RE_2.sub(" ", _CAMEL_RE_1.sub(" ", str(col).strip()))
    return {t for t in _TOKEN_SPLIT_RE.split(text.lower()) if t}


def _name_suggests_id(col: str) -> bool:
    """True only when a WHOLE TOKEN of the column name means 'identifier'."""
    lowered = str(col).lower().strip()
    if any(lowered.startswith(p) for p in ID_COLUMN_NAME_PREFIXES):
        return True
    return bool(_name_tokens(col) & ID_COLUMN_NAME_TOKENS)


def _is_low_variance(s: pd.Series) -> bool:
    """Constant, all-null, or near-constant columns carry no usable signal."""
    non_null = int(s.notna().sum())
    if non_null == 0:
        return True
    if int(s.nunique(dropna=True)) <= LOW_VARIANCE_NUNIQUE:
        return True
    try:
        top_share = float(s.value_counts(dropna=True).iloc[0]) / float(len(s))
    except Exception:
        return False
    return top_share >= NEAR_CONSTANT_FREQ


def _all_values_unique(s: pd.Series) -> bool:
    non_null = int(s.notna().sum())
    return non_null > 0 and int(s.nunique(dropna=True)) == non_null


def _unique_integers(s: pd.Series) -> np.ndarray | None:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return None
    if not _all_values_unique(s):
        return None
    arr = s.dropna().to_numpy(dtype="float64")
    if len(arr) < 3 or not np.all(np.isfinite(arr)) or not np.all(arr == np.round(arr)):
        return None
    return arr


def _is_row_counter(s: pd.Series) -> bool:
    """Unique integers forming a dense run that starts at 0 or 1: a row counter.

    Catches pandas' "Unnamed: 0" and R-style row names 1..n (sorted OR
    shuffled - a shuffled counter still leaks the original sort order).
    Does NOT catch a year column (1950..2023) or sorted square footage.
    """
    arr = _unique_integers(s)
    if arr is None:
        return False
    lo, hi = float(arr.min()), float(arr.max())
    return lo in (0.0, 1.0) and (hi - lo + 1) <= ROW_COUNTER_DENSITY * len(arr)


def _is_id_like_column(s: pd.Series) -> bool:
    """Backward-compatible name for the structural (name-free) ID rule."""
    return _is_row_counter(s)


def _is_suspicious_monotonic(s: pd.Series) -> bool:
    """Unique, integer, monotonic - but not a row counter. Reported, never dropped."""
    arr = _unique_integers(s)
    if arr is None or _is_row_counter(s):
        return False
    return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing)


def _looks_like_identifier(col: str, s: pd.Series) -> bool:
    """A row identifier that must not be a feature.

    1. Structural: a row counter (see _is_row_counter).
    2. Name token ('id', 'uuid', camelCase 'customerId', ...) AND all values
       unique. A name hint alone never deletes a column: 'store_id' with
       repeated values is a legitimate grouping key.
    """
    if _is_row_counter(s):
        return True
    return _name_suggests_id(col) and _all_values_unique(s)


def _clean_numeric_text(s: pd.Series) -> pd.Series:
    """'$1,250.50' -> 1250.5, '12%' -> 12, '(300)' -> -300; failures -> NaN."""
    txt = s.astype("string").str.strip()
    neg = txt.str.match(_ACCOUNTING_NEG_PATTERN).fillna(False).astype(bool).to_numpy()
    txt = txt.str.replace(_ACCOUNTING_NEG_PATTERN, r"\1", regex=True)
    txt = txt.str.replace(_NUM_NOISE_PATTERN, "", regex=True)
    num = pd.to_numeric(txt, errors="coerce").astype("float64")
    vals = num.to_numpy(copy=True)
    vals[neg] = -vals[neg]
    return pd.Series(vals, index=s.index, name=s.name)


def _numeric_string_column(col: str, s: pd.Series) -> bool:
    """Should this text column be parsed as numbers?

    Requires >=95% of non-null values to parse. Refuses when the name says it
    is a code (zip / postcode) or values carry leading zeros ('02139'), since
    those are categories that merely look numeric.
    """
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    nn = s.dropna()
    if len(nn) == 0:
        return False
    if _name_tokens(col) & CATEGORICAL_CODE_TOKENS:
        return False
    txt = nn.astype(str).str.strip()
    if txt.str.match(r"^0\d").mean() > 0.05:
        return False
    parsed = _clean_numeric_text(nn)
    return bool(parsed.notna().mean() >= NUMERIC_STRING_MIN_SHARE)


def _numeric_coded_categorical(col: str, s: pd.Series) -> bool:
    """An integer column whose name marks it as a code (zip, class, type...)."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    if not (_name_tokens(col) & CATEGORICAL_CODE_TOKENS):
        return False
    nn = s.dropna().to_numpy(dtype="float64")
    if len(nn) == 0 or not np.all(nn == np.round(nn)):
        return False
    n_unique = len(np.unique(nn))
    return 2 <= n_unique <= max(2, int(0.5 * len(nn)))


def _as_category_strings(s: pd.Series) -> pd.Series:
    return pd.Series([MISSING_TOKEN if pd.isna(v) else str(v) for v in s.to_numpy(dtype=object)],
                     index=s.index, name=s.name, dtype=object)


def _looks_like_datetime(s: pd.Series) -> bool:
    nn = s.dropna()
    if len(nn) == 0:
        return False
    sample = nn.astype(str).head(500)
    # a date needs digits AND a separator; bare words / codes never qualify
    if sample.str.contains(r"\d").mean() < 0.9 or sample.str.contains(r"[-/:.\s]").mean() < 0.9:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() > 0.85)


def _frame_hashes(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame.astype(str), index=False)


# ---- result container --------------------------------------------------------

@dataclass
class PreprocessingResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    summary: dict[str, Any] = field(default_factory=dict)
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    #: RawFeatureBuilder + fitted ColumnTransformer (accepts raw CSV rows)
    raw_preprocessor: Any = None
    target_transform: Literal["none", "log1p"] = "none"
    groups_train: np.ndarray | None = None
    # ---- v3: what in-fold preprocessing needs ----
    #: engineered frames (date parts, time features, parsed numbers) = the
    #: INPUT of the ColumnTransformer. Modelling fits the transformer per fold.
    X_train_raw: pd.DataFrame | None = None
    X_test_raw: pd.DataFrame | None = None
    X_val_raw: pd.DataFrame | None = None
    #: unfitted ColumnTransformers: {"default": ..., "linear": ...}
    preprocessor_templates: dict[str, Any] | None = None
    #: stateless raw-row -> engineered-frame step (front half of raw_preprocessor)
    raw_feature_builder: Any = None
    y_train_original: np.ndarray | None = None
    y_test_original: np.ndarray | None = None
    y_val_original: np.ndarray | None = None


# ---- profiling ----------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Shape, dtypes, missingness, target stats, datetime / ID / code hints."""
    tgt = df[target]
    if not pd.api.types.is_numeric_dtype(tgt):
        tgt = _clean_numeric_text(tgt)
    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "target": target,
        "target_dtype": str(df[target].dtype),
        "target_stats": {
            "mean": float(tgt.mean()) if tgt.notna().any() else float("nan"),
            "std": float(tgt.std()) if tgt.notna().sum() > 1 else float("nan"),
            "min": float(tgt.min()) if tgt.notna().any() else float("nan"),
            "max": float(tgt.max()) if tgt.notna().any() else float("nan"),
            "skew": float(tgt.skew()) if tgt.notna().sum() > 2 else 0.0,
            "missing": int(tgt.isna().sum()),
        },
        "columns": [],
        "missing_total": int(df.isna().sum().sum()),
        "datetime_candidates": [],
        "low_variance_columns": [],
        "high_cardinality_columns": [],
        "id_like_columns": [],
        "id_suspect_columns": [],
        "numeric_string_columns": [],
        "numeric_coded_categoricals": [],
    }
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        info: dict[str, Any] = {
            "name": col, "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if _looks_like_identifier(col, s):
            info["kind"] = "id_like"
            profile["id_like_columns"].append(col)
            profile["columns"].append(info)
            continue
        if _is_suspicious_monotonic(s):
            profile["id_suspect_columns"].append(col)
        if _numeric_string_column(col, s):
            info["kind"] = "numeric_text"
            profile["numeric_string_columns"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            info["kind"] = "numeric"
            info["mean"] = float(s.mean()) if s.notna().any() else None
            info["std"] = float(s.std()) if s.notna().sum() > 1 else None
            if _numeric_coded_categorical(col, s):
                info["kind"] = "numeric_code"
                profile["numeric_coded_categoricals"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["kind"] = "datetime"
            profile["datetime_candidates"].append(col)
        else:
            info["kind"] = "categorical"
            info["high_cardinality"] = info["unique"] > HIGH_CARDINALITY_THRESHOLD
            if _looks_like_datetime(s):
                info["kind"] = "datetime"
                profile["datetime_candidates"].append(col)
            elif info["high_cardinality"]:
                profile["high_cardinality_columns"].append(col)
        if _is_low_variance(s):
            profile["low_variance_columns"].append(col)
        profile["columns"].append(info)
    return profile


# ---- datetime features -----------------------------------------------------------

def _extract_datetime_features(df: pd.DataFrame, cols: list[str],
                               hour_cols: set[str] | None = None
                               ) -> tuple[pd.DataFrame, set[str]]:
    """Replace each datetime column with year/month/day/weekday(/hour).

    `hour_cols` is decided on the TRAINING frame and passed to every other
    frame so all splits get the same schema.
    """
    out = df.copy()
    used_hour_cols: set[str] = set()
    for col in cols:
        if col not in out.columns:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}_year"] = parsed.dt.year
        out[f"{col}_month"] = parsed.dt.month
        out[f"{col}_day"] = parsed.dt.day
        out[f"{col}_weekday"] = parsed.dt.weekday
        include_hour = (col in hour_cols) if hour_cols is not None \
            else (parsed.dt.hour.fillna(0).sum() > 0)
        if include_hour:
            out[f"{col}_hour"] = parsed.dt.hour
            used_hour_cols.add(col)
        out = out.drop(columns=[col])
    return out, used_hour_cols


def _time_split_features(frames: list[pd.DataFrame], col: str, parsed_train: pd.Series
                         ) -> tuple[list[pd.DataFrame], list[str], dict]:
    """Elapsed-days trend feature + calendar parts covering >= 2 cycles of training."""
    t0, t1 = parsed_train.min(), parsed_train.max()
    span_days = (t1 - t0).total_seconds() / 86400.0 if pd.notna(t0) and pd.notna(t1) else 0.0
    has_time_of_day = bool(parsed_train.dt.hour.fillna(0).sum() > 0)
    parts = {
        "weekday": span_days >= 14,
        "day": span_days >= 60,
        "month": span_days >= 730,
        "hour": has_time_of_day and span_days >= 2,
    }
    out = []
    for fr in frames:
        fr = fr.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(fr[col], errors="coerce")
        fr[f"{col}_elapsed_days"] = (parsed - t0).dt.total_seconds() / 86400.0
        for part, keep in parts.items():
            if keep:
                fr[f"{col}_{part}"] = getattr(parsed.dt, part)
        out.append(fr.drop(columns=[col]))
    names = [f"{col}_elapsed_days"] + [f"{col}_{p}" for p, k in parts.items() if k]
    return out, names, {"column": col, "t0": t0, "parts": [p for p, k in parts.items() if k]}


# ---- transformers (all picklable, all live inside the saved bundle) -----------

class RawFeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw CSV rows -> the engineered frame the ColumnTransformer expects.

    Parses formatted numbers, casts categorical codes to strings, derives
    date parts and time-split features. Stateless (all settings are params).
    """

    def __init__(self, datetime_cols=None, hour_cols=None, time_spec=None,
                 numeric_string_cols=None, categorical_cols=None):
        self.datetime_cols = datetime_cols
        self.hour_cols = hour_cols
        self.time_spec = time_spec
        self.numeric_string_cols = numeric_string_cols
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for c in (self.numeric_string_cols or []):
            if c in X.columns and not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = _clean_numeric_text(X[c])
        for c in (self.categorical_cols or []):
            if c in X.columns:
                X[c] = _as_category_strings(X[c])
        if self.datetime_cols:
            X, _ = _extract_datetime_features(X, [c for c in self.datetime_cols if c in X.columns],
                                              hour_cols=set(self.hour_cols or []))
        spec = self.time_spec
        if spec and spec["column"] in X.columns:
            col = spec["column"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_elapsed_days"] = (parsed - spec["t0"]).dt.total_seconds() / 86400.0
            for part in spec["parts"]:
                X[f"{col}_{part}"] = getattr(parsed.dt, part)
            X = X.drop(columns=[col])
        return X


class InfinityToNaN(BaseEstimator, TransformerMixin):
    """+/-inf -> NaN inside the pipeline (the imputer rejects non-finite input)."""

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1] if np.ndim(X) > 1 else 1
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype="float64")
        return np.where(np.isfinite(arr), arr, np.nan)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Everything to str; NaN becomes its own '__missing__' level.

    Mixed int/str columns used to crash OneHotEncoder, and mode imputation hid
    the fact that a value was missing (often informative).
    """

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame({c: _as_category_strings(X[c]) for c in X.columns}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Category -> relative frequency in the training data (optional encoding;
    kept so bundles saved by earlier versions still load)."""

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.lookups_ = {col: X[col].astype(object).value_counts(normalize=True)
                         for col in X.columns}
        return self

    def transform(self, X):
        X = self._as_frame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            out[col] = X[col].astype(object).map(self.lookups_[col]).astype("float64").fillna(0.0)
        return out.to_numpy(dtype="float64")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)

    @staticmethod
    def _as_frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _feature_names_from(preprocessor: ColumnTransformer, numeric_cols, low_card_cat,
                        high_card_cat) -> list[str]:
    try:
        cleaned = []
        for name in preprocessor.get_feature_names_out():
            text = str(name)
            for prefix in ("num__", "cat__", "freq__", "te__"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            cleaned.append(text)
        return cleaned
    except Exception:
        return list(numeric_cols) + list(low_card_cat) + list(high_card_cat)


def _build_column_transformer(numeric_cols, low_card_cat, high_card_cat, *,
                              add_missing_indicators: bool, interactions: bool,
                              high_cardinality_encoding: str, time_ordered: bool,
                              random_state: int) -> ColumnTransformer:
    numeric_steps = [
        ("finite", InfinityToNaN()),
        ("impute", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
        ("scale", StandardScaler()),
    ]
    if interactions:
        numeric_steps.append(("interact", PolynomialFeatures(degree=2, interaction_only=True,
                                                             include_bias=False)))
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), list(numeric_cols)))
    if low_card_cat:
        transformers.append(("cat", Pipeline([
            ("clean", CategoricalCleaner()),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(low_card_cat)))
    if high_card_cat:
        if high_cardinality_encoding == "frequency":
            transformers.append(("freq", Pipeline([
                ("clean", CategoricalCleaner()),
                ("freq", FrequencyEncoder()),
                ("scale", StandardScaler()),
            ]), list(high_card_cat)))
        else:
            # cross-fitted: fit_transform encodes each training row with
            # statistics from OTHER folds, so a row never sees its own target
            transformers.append(("te", Pipeline([
                ("clean", CategoricalCleaner()),
                ("encode", TargetEncoder(target_type="continuous", cv=5,
                                         shuffle=not time_ordered,
                                         random_state=None if time_ordered else random_state)),
                ("scale", StandardScaler()),
            ]), list(high_card_cat)))
    return ColumnTransformer(transformers, remainder="drop")


# ---- target transform ------------------------------------------------------------------

def _count_like(y: np.ndarray) -> bool:
    """Non-negative integers that behave like COUNTS, not integer-valued amounts.

    Prices in whole dollars are integers too; counts are distinguished by
    small magnitudes, many ties or many zeros.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0 or (y < 0).any() or not np.allclose(y, np.round(y)):
        return False
    zero_share = float(np.mean(y == 0))
    n_unique = len(np.unique(y))
    return bool(zero_share >= 0.05 or n_unique <= 100 or np.median(y) <= 50)


def _decide_log(y_train: np.ndarray, request) -> tuple[bool, str]:
    y = np.asarray(y_train, float)
    skew = float(pd.Series(y).skew()) if len(y) > 2 else 0.0
    if isinstance(request, str) and request.lower() == "auto":
        if (y < 0).any():
            return False, "target has negative values"
        if _count_like(y):
            return False, "target looks like counts (kept on its natural scale for Poisson)"
        if skew > LOG_TRANSFORM_SKEW_THRESHOLD:
            return True, f"training-target skew {skew:+.2f} > {LOG_TRANSFORM_SKEW_THRESHOLD}"
        return False, f"training-target skew {skew:+.2f} is mild"
    if request:
        if (y < 0).any():
            return False, "log1p requested but the training target has negative values"
        return True, "requested"
    return False, "not requested"


# ---- main API -----------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    target: str,
    *,
    df_val: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "random",          # "random" | "time"
    time_column: str | None = None,
    group_column: str | None = None,
    log_transform_target: bool | str = False,  # False | True | "auto"
    auto_datetime_features: bool = True,
    drop_low_variance: bool = True,
    auto_drop_id_columns: bool = True,
    drop_duplicate_rows: bool = True,
    add_missing_indicators: bool = True,
    add_interactions: bool = False,
    high_cardinality_encoding: str = "target",  # "target" | "frequency"
    categorical_columns: list[str] | None = None,
    coerce_numeric_strings: bool = True,
    drop_columns: list[str] | None = None,
) -> PreprocessingResult:
    """End-to-end preprocessing. See the module docstring for the design."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in training data.")
    if df_test is not None and target not in df_test.columns:
        raise ValueError(f"Target '{target}' missing from the test file.")
    if df_val is not None and target not in df_val.columns:
        raise ValueError(f"Target '{target}' missing from the validation file.")
    time_ordered = split_strategy == "time" and bool(time_column)

    frames = {"train": df.copy(), "val": None if df_val is None else df_val.copy(),
              "test": None if df_test is None else df_test.copy()}

    def _each(fn):
        for k, fr in frames.items():
            if fr is not None:
                frames[k] = fn(fr)

    # 0. user-requested column drops (e.g. a leakage suspect)
    user_dropped = [c for c in (drop_columns or []) if c in frames["train"].columns and c != target]
    if user_dropped:
        _each(lambda fr: fr.drop(columns=[c for c in user_dropped if c in fr.columns]))

    # 1. numeric target: parse "$250,000" style targets, reject true text
    if not pd.api.types.is_numeric_dtype(frames["train"][target]):
        parsed = _clean_numeric_text(frames["train"][target])
        if parsed.notna().mean() < NUMERIC_STRING_MIN_SHARE:
            raise ValueError(f"Target '{target}' is not numeric. This system handles regression only.")
        _each(lambda fr: fr.assign(**{target: _clean_numeric_text(fr[target])}))

    # 2. +/-inf -> NaN
    n_inf = 0

    def _sanitize(fr):
        nonlocal n_inf
        num = fr.select_dtypes(include=[np.number]).columns
        if len(num):
            mask = np.isinf(fr[num].to_numpy(dtype="float64", na_value=np.nan))
            if mask.any():
                n_inf += int(mask.sum())
                fr = fr.copy()
                fr[num] = fr[num].replace([np.inf, -np.inf], np.nan)
        return fr
    _each(_sanitize)

    # 3. rows without a target carry nothing to learn from
    _each(lambda fr: fr.dropna(subset=[target]).reset_index(drop=True))

    # 4. exact duplicates inside the training file (before any split)
    n_duplicates_dropped = 0
    if drop_duplicate_rows:
        before = len(frames["train"])
        frames["train"] = frames["train"].drop_duplicates().reset_index(drop=True)
        n_duplicates_dropped = before - len(frames["train"])

    # 5. label-free structural decisions, made on the training FILE
    profile = profile_dataframe(frames["train"], target)
    numeric_string_cols = []
    if coerce_numeric_strings:
        numeric_string_cols = [c for c in profile["numeric_string_columns"]
                               if c not in (time_column, group_column)]
        for c in numeric_string_cols:
            _each(lambda fr, c=c: fr.assign(**{c: _clean_numeric_text(fr[c])}) if c in fr.columns else fr)

    dropped_id_cols: list[str] = []
    if auto_drop_id_columns and profile["id_like_columns"]:
        keep = {time_column, group_column}
        dropped_id_cols = [c for c in profile["id_like_columns"] if c not in keep]
        _each(lambda fr: fr.drop(columns=[c for c in dropped_id_cols if c in fr.columns]))
    id_suspects = [c for c in profile["id_suspect_columns"]
                   if c not in dropped_id_cols and c not in (time_column, group_column)]

    categorical_overrides = [c for c in (categorical_columns or []) if c in frames["train"].columns
                             and c not in (target, time_column, group_column)]
    for c in profile["numeric_coded_categoricals"]:
        if c not in categorical_overrides and c not in (time_column, group_column) \
                and c in frames["train"].columns:
            categorical_overrides.append(c)
    for c in categorical_overrides:
        _each(lambda fr, c=c: fr.assign(**{c: _as_category_strings(fr[c])}) if c in fr.columns else fr)

    datetime_cols_used: list[str] = []
    hour_cols: set[str] = set()
    if auto_datetime_features and profile["datetime_candidates"]:
        protected = {time_column} if time_ordered else set()
        datetime_cols_used = [c for c in profile["datetime_candidates"]
                              if c not in protected and c not in categorical_overrides
                              and c in frames["train"].columns]
        if datetime_cols_used:
            frames["train"], hour_cols = _extract_datetime_features(frames["train"], datetime_cols_used)
            for k in ("val", "test"):
                if frames[k] is not None:
                    frames[k], _ = _extract_datetime_features(frames[k], datetime_cols_used,
                                                              hour_cols=hour_cols)

    # 6. split (or use the user's files)
    def _time_key(fr):
        s = fr[time_column]
        if pd.api.types.is_numeric_dtype(s):
            return s.reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(s, errors="coerce").reset_index(drop=True)

    def _sorted_by_time(fr):
        key = _time_key(fr)
        order = np.argsort(key.to_numpy(), kind="stable")
        return fr.iloc[order].reset_index(drop=True)

    time_order_ok = None
    if df_test is not None:
        train_df, test_df, val_df = frames["train"], frames["test"], frames["val"]
        split_used = "user-supplied"
        if time_ordered and time_column in train_df.columns:
            # TimeSeriesSplit assumes rows are in time order: sort every file
            train_df = _sorted_by_time(train_df)
            if time_column in test_df.columns:
                test_df = _sorted_by_time(test_df)
            if val_df is not None and time_column in val_df.columns:
                val_df = _sorted_by_time(val_df)
            tr_max = _time_key(train_df).max()
            later = [fr for fr in (val_df, test_df) if fr is not None and time_column in fr.columns]
            time_order_ok = bool(all(_time_key(fr).min() >= tr_max for fr in later)) if later else None
            split_used = f"user-supplied, each file sorted by '{time_column}'"
    else:
        base = frames["train"]
        if time_ordered:
            if time_column not in base.columns:
                raise ValueError(f"Time column '{time_column}' not found.")
            base = _sorted_by_time(base)
            key_sorted = _time_key(base)
            cut = int(len(base) * (1 - test_size))
            # never split one timestamp between train and test
            while 0 < cut < len(base) and key_sorted.iloc[cut] == key_sorted.iloc[cut - 1]:
                cut += 1
            if cut >= len(base):
                cut = int(len(base) * (1 - test_size))
            train_df, test_df = base.iloc[:cut].reset_index(drop=True), base.iloc[cut:].reset_index(drop=True)
            split_used = f"time-aware (sorted by '{time_column}')"
            time_order_ok = True
        elif group_column and group_column in base.columns:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(splitter.split(base, groups=base[group_column].to_numpy()))
            train_df = base.iloc[tr_idx].reset_index(drop=True)
            test_df = base.iloc[te_idx].reset_index(drop=True)
            split_used = f"group-aware (grouped by '{group_column}')"
        else:
            train_df, test_df = train_test_split(base, test_size=test_size, random_state=random_state)
            train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
            split_used = "random"
        val_df = None

    # 7. cross-file overlap diagnostics (reported, never silently "fixed")
    overlap = {"train_test_duplicate_rows": 0, "train_val_duplicate_rows": 0,
               "train_test_group_overlap": 0}
    try:
        cols = [c for c in train_df.columns]
        tr_h = set(_frame_hashes(train_df[cols]))
        if df_test is not None and set(cols) <= set(test_df.columns):
            overlap["train_test_duplicate_rows"] = int(_frame_hashes(test_df[cols]).isin(tr_h).sum())
        if val_df is not None and set(cols) <= set(val_df.columns):
            overlap["train_val_duplicate_rows"] = int(_frame_hashes(val_df[cols]).isin(tr_h).sum())
        if group_column and group_column in train_df.columns and group_column in test_df.columns:
            overlap["train_test_group_overlap"] = int(len(
                set(train_df[group_column].astype(str)) & set(test_df[group_column].astype(str))))
    except Exception:
        pass

    # 8. low-variance decision on the TRAINING split only
    dropped_lowvar: list[str] = []
    if drop_low_variance:
        dropped_lowvar = [c for c in train_df.columns
                          if c not in (target, time_column, group_column) and _is_low_variance(train_df[c])]
        if dropped_lowvar:
            train_df = train_df.drop(columns=dropped_lowvar)
            test_df = test_df.drop(columns=[c for c in dropped_lowvar if c in test_df.columns])
            if val_df is not None:
                val_df = val_df.drop(columns=[c for c in dropped_lowvar if c in val_df.columns])

    # 9. target + groups out
    def _pop(frame):
        y = frame[target].to_numpy(dtype=float)
        X = frame.drop(columns=[target])
        g = None
        if group_column and group_column in X.columns:
            g = X[group_column].to_numpy()
            X = X.drop(columns=[group_column])
        return X, y, g

    X_train_raw, y_train_orig, groups_train = _pop(train_df)
    X_test_raw, y_test_orig, _ = _pop(test_df)
    X_val_raw, y_val_orig = None, None
    if val_df is not None:
        X_val_raw, y_val_orig, _ = _pop(val_df)

    # 10. the split's time column becomes trend + calendar features
    time_features: list[str] = []
    time_spec: dict | None = None
    if time_ordered and time_column in X_train_raw.columns:
        parts = [X_train_raw, X_test_raw] + ([X_val_raw] if X_val_raw is not None else [])
        col = X_train_raw[time_column]
        if pd.api.types.is_numeric_dtype(col):
            time_features = [time_column]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed_train = pd.to_datetime(col, errors="coerce")
            if parsed_train.notna().mean() > 0.85:
                parts, time_features, time_spec = _time_split_features(parts, time_column, parsed_train)
            else:
                parts = [fr.drop(columns=[time_column]) for fr in parts]
        X_train_raw, X_test_raw = parts[0], parts[1]
        if X_val_raw is not None:
            X_val_raw = parts[2]

    # 11. column roles, decided on the training split
    numeric_cols, low_card_cat, high_card_cat = [], [], []
    for col in X_train_raw.columns:
        if pd.api.types.is_numeric_dtype(X_train_raw[col]) and col not in categorical_overrides:
            numeric_cols.append(col)
        elif X_train_raw[col].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD:
            high_card_cat.append(col)
        else:
            low_card_cat.append(col)
    model_cols = numeric_cols + low_card_cat + high_card_cat
    X_train_raw = X_train_raw[model_cols].reset_index(drop=True)
    X_test_raw = X_test_raw.reindex(columns=model_cols).reset_index(drop=True)
    if X_val_raw is not None:
        X_val_raw = X_val_raw.reindex(columns=model_cols).reset_index(drop=True)

    # 12. target transform - decided from TRAINING targets only
    want_log, log_reason = _decide_log(y_train_orig, log_transform_target)
    if want_log:
        held_out = [a for a in (y_test_orig, y_val_orig) if a is not None]
        if any((a <= -1).any() for a in held_out):
            want_log, log_reason = False, "a held-out target is <= -1, which log1p cannot represent"
    target_transform: Literal["none", "log1p"] = "log1p" if want_log else "none"

    def fwd(a):
        return np.log1p(a) if want_log else np.asarray(a, float)
    y_train, y_test = fwd(y_train_orig), fwd(y_test_orig)
    y_val = fwd(y_val_orig) if y_val_orig is not None else None

    # 13. ColumnTransformer templates + one fitted copy (display / legacy use)
    interactions_used = bool(add_interactions and 2 <= len(numeric_cols) <= MAX_INTERACTION_BASE_COLUMNS)
    common = dict(add_missing_indicators=add_missing_indicators,
                  high_cardinality_encoding=high_cardinality_encoding,
                  time_ordered=time_ordered, random_state=random_state)
    templates = {"default": _build_column_transformer(numeric_cols, low_card_cat, high_card_cat,
                                                      interactions=False, **common)}
    if interactions_used:
        templates["linear"] = _build_column_transformer(numeric_cols, low_card_cat, high_card_cat,
                                                        interactions=True, **common)
    preprocessor = clone(templates["linear" if interactions_used else "default"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train = preprocessor.fit_transform(X_train_raw, y_train)
        X_test = preprocessor.transform(X_test_raw)
        X_val = preprocessor.transform(X_val_raw) if X_val_raw is not None else None

    raw_builder = RawFeatureBuilder(datetime_cols=list(datetime_cols_used), hour_cols=sorted(hour_cols),
                                    time_spec=time_spec, numeric_string_cols=list(numeric_string_cols),
                                    categorical_cols=list(categorical_overrides))
    raw_preprocessor = Pipeline([("raw_features", raw_builder), ("columns", preprocessor)])

    feature_names = _feature_names_from(preprocessor, numeric_cols, low_card_cat, high_card_cat)
    if len(feature_names) != X_train.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    train_skew = float(pd.Series(y_train_orig).skew()) if len(y_train_orig) > 2 else 0.0
    summary = {
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]) if X_val is not None else 0,
        "n_test": int(X_test.shape[0]),
        "n_features_after": int(X_train.shape[1]),
        "numeric_cols": numeric_cols,
        "low_cardinality_categorical": low_card_cat,
        "high_cardinality_categorical": high_card_cat,
        # legacy key (these columns are target-encoded by default now)
        "high_cardinality_categorical_freq_encoded": high_card_cat,
        "high_cardinality_encoding": high_cardinality_encoding,
        "numeric_string_columns_parsed": numeric_string_cols,
        "categorical_overrides": categorical_overrides,
        "user_dropped_columns": user_dropped,
        "datetime_cols_extracted": datetime_cols_used,
        "low_variance_dropped": dropped_lowvar,
        "id_like_columns_dropped": dropped_id_cols,
        "id_suspect_columns_kept": id_suspects,
        "split_strategy_used": split_used,
        "time_order_ok": time_order_ok,
        "target_transform": target_transform,
        "target_transform_requested": ("auto" if isinstance(log_transform_target, str)
                                       and log_transform_target.lower() == "auto"
                                       else bool(log_transform_target)),
        "target_transform_reason": log_reason,
        "target_skew_train": train_skew,
        "target_count_like": _count_like(y_train_orig),
        "target_nonnegative": bool((y_train_orig >= 0).all()),
        "log_candidate": bool((y_train_orig >= 0).all() and train_skew > LOG_CANDIDATE_SKEW),
        "interaction_features_added": interactions_used,
        "time_features_from_split_column": time_features,
        "test_size": test_size,
        "random_state": random_state,
        "group_column": group_column if group_column and groups_train is not None else None,
        "duplicate_rows_dropped": n_duplicates_dropped,
        "infinite_values_sanitized": n_inf,
        "missing_indicators_added": bool(add_missing_indicators),
        **overlap,
    }
    return PreprocessingResult(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=feature_names, preprocessor=preprocessor, summary=summary,
        X_val=X_val, y_val=y_val, raw_preprocessor=raw_preprocessor,
        target_transform=target_transform, groups_train=groups_train,
        X_train_raw=X_train_raw, X_test_raw=X_test_raw, X_val_raw=X_val_raw,
        preprocessor_templates=templates, raw_feature_builder=raw_builder,
        y_train_original=y_train_orig, y_test_original=y_test_orig, y_val_original=y_val_orig,
    )
