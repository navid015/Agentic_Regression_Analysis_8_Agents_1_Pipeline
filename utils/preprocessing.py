"""
Preprocessing for regression — supports:

- Single CSV (random or time-aware split)
- Two CSVs (train + test, no split needed)
- Three CSVs (train + val + test)
- Auto datetime feature extraction (year/month/day/weekday/hour from datetime cols)
- Optional log-transform of skewed targets (y -> log1p(y))
- Optional group-aware CV (preserves the group column for downstream KFold)
- Drops near-constant / low-variance columns
- Median + mode imputation, one-hot for low-card cats, frequency encoding for high-card,
  StandardScaler at the end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

HIGH_CARDINALITY_THRESHOLD = 20
LOW_VARIANCE_NUNIQUE = 1  # column has only 1 unique value -> drop
NEAR_CONSTANT_FREQ = 0.999  # single value covers >=99.9% of rows -> effectively constant
LOG_TRANSFORM_SKEW_THRESHOLD = 1.5  # |skew| above this triggers auto log-suggestion
# Pairwise interaction features are only added when there are at most this many
# numeric columns: n columns produce n*(n-1)/2 extra features.
MAX_INTERACTION_BASE_COLUMNS = 15

# Whole-token names that mean "this is a row identifier".
#
# These are matched against TOKENS of the column name, never as substrings.
# The previous substring check ("id" in col.lower()) silently flagged - and
# therefore dropped - ordinary features such as fixed_acidity, volatile_acidity,
# citric_acid, residual_sugar, humidity, width, solidity and confidence. On the
# wine-quality dataset that removed four of eleven real features before training.
ID_COLUMN_NAME_TOKENS = frozenset({
    "id", "ids", "index", "idx", "rowid", "row", "rownum", "rownumber",
    "key", "pk", "uuid", "guid", "serial", "seq",
})
ID_COLUMN_NAME_PREFIXES = ("unnamed:",)
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _name_suggests_id(col: str) -> bool:
    """True only when a WHOLE TOKEN of the column name means 'identifier'.

    'customer_id' -> tokens {customer, id} -> True
    'fixed_acidity' -> tokens {fixed, acidity} -> False
    'Unnamed: 0' -> prefix match -> True
    """
    lowered = str(col).lower().strip()
    if any(lowered.startswith(p) for p in ID_COLUMN_NAME_PREFIXES):
        return True
    tokens = {t for t in _TOKEN_SPLIT_RE.split(lowered) if t}
    return bool(tokens & ID_COLUMN_NAME_TOKENS)


def _is_low_variance(s: pd.Series) -> bool:
    """Constant, all-null, or near-constant columns carry no usable signal.

    The original check only caught nunique <= 1. A column where 99.9% of rows
    share one value is statistically just as useless but survived that test,
    then went on to consume a one-hot slot or distort a scaler.
    """
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


def _looks_like_identifier(col: str, s: pd.Series) -> bool:
    """Decide whether a column is a row identifier that must not be a feature.

    Two independent routes, and BOTH require the values to actually behave like
    identifiers - a name hint alone is never enough to delete a column:

      1. Structural: numeric, all-unique, integer-valued, monotonic. This is the
         leftover-row-index case (pandas' "Unnamed: 0" on a target-sorted CSV).
      2. Name token + all values unique. Catches 'customer_id' holding random
         unique ints or strings, which is not monotonic and so fails route 1.

    A column named 'store_id' with repeated values is NOT an identifier here -
    it is a legitimate categorical / grouping key, and dropping it would throw
    away real signal.
    """
    if _is_id_like_column(s):
        return True
    return _name_suggests_id(col) and _all_values_unique(s)


def _is_id_like_column(s: pd.Series) -> bool:
    """A column looks like a row-index/ID if:
       - it's integer-like
       - every value is unique
       - it's monotonically increasing OR decreasing

    These columns are usually leftover row indices (e.g. pandas' "Unnamed: 0")
    that carry no real signal — but tree models can use them to leak the original
    sort order, which is the diamonds-dataset gotcha.
    """
    if not pd.api.types.is_numeric_dtype(s):
        return False
    if s.nunique(dropna=True) != s.notna().sum():
        return False  # not all unique
    # treat as ID only if values are integer-like
    arr = s.dropna().to_numpy()
    if not np.all(arr == arr.astype(np.int64)):
        return False
    return s.is_monotonic_increasing or s.is_monotonic_decreasing


@dataclass
class PreprocessingResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    preprocessor: ColumnTransformer
    summary: dict[str, Any] = field(default_factory=dict)
    # optional pieces
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    # RawFeatureBuilder + ColumnTransformer: accepts raw rows (used for the saved bundle)
    raw_preprocessor: Any = None
    target_transform: Literal["none", "log1p"] = "none"
    groups_train: np.ndarray | None = None  # for group-aware CV


def profile_dataframe(df: pd.DataFrame, target: str) -> dict[str, Any]:
    """Return shape, dtypes, missingness, target stats, datetime hints."""
    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "target": target,
        "target_dtype": str(df[target].dtype),
        "target_stats": {
            "mean": float(df[target].mean()),
            "std":  float(df[target].std()),
            "min":  float(df[target].min()),
            "max":  float(df[target].max()),
            "skew": float(df[target].skew()) if df[target].notna().sum() > 2 else 0.0,
            "missing": int(df[target].isna().sum()),
        },
        "columns": [],
        "missing_total": int(df.isna().sum().sum()),
        "datetime_candidates": [],
        "low_variance_columns": [],
        "high_cardinality_columns": [],
        "id_like_columns": [],
    }
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        info: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        # ID-like detection: structural pattern, or an identifier-style NAME
        # backed by all-unique values. A name hint alone never qualifies.
        if _looks_like_identifier(col, s):
            info["kind"] = "id_like"
            profile["id_like_columns"].append(col)
            profile["columns"].append(info)
            continue
        if pd.api.types.is_numeric_dtype(s):
            info["kind"] = "numeric"
            info["mean"] = float(s.mean()) if s.notna().any() else None
            info["std"]  = float(s.std())  if s.notna().any() else None
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["kind"] = "datetime"
            profile["datetime_candidates"].append(col)
        else:
            info["kind"] = "categorical"
            info["high_cardinality"] = info["unique"] > HIGH_CARDINALITY_THRESHOLD
            if info["high_cardinality"]:
                profile["high_cardinality_columns"].append(col)
            # try to parse as datetime — if most values parse, treat as datetime.
            # Use is_object_dtype OR is_string_dtype so we catch newer pandas
            # backends where string columns aren't `object`.
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().sum() / max(s.notna().sum(), 1) > 0.85:
                    info["kind"] = "datetime"
                    profile["datetime_candidates"].append(col)
                    # remove from high-cardinality list since it's actually a datetime
                    if col in profile["high_cardinality_columns"]:
                        profile["high_cardinality_columns"].remove(col)
        if _is_low_variance(s):
            profile["low_variance_columns"].append(col)
        profile["columns"].append(info)
    return profile


# ---- datetime feature extraction --------------------------------------------


def _extract_datetime_features(df: pd.DataFrame, cols: list[str],
                                hour_cols: set[str] | None = None
                                ) -> tuple[pd.DataFrame, set[str]]:
    """For each datetime column, replace it with year/month/day/weekday/hour.

    `hour_cols`, when given, forces hour-feature inclusion for exactly those
    columns instead of deciding per-frame. Deciding independently for train
    vs. val/test (e.g. "does this split have any non-midnight timestamps?")
    can add `_hour` to one split's schema but not another's, which then
    crashes `ColumnTransformer.transform` with a missing-column error. Always
    derive `hour_cols` from the training frame and pass it to every other
    frame so the extracted feature set matches across splits.
    """
    out = df.copy()
    used_hour_cols: set[str] = set()
    for col in cols:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}_year"]    = parsed.dt.year
        out[f"{col}_month"]   = parsed.dt.month
        out[f"{col}_day"]     = parsed.dt.day
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
    """Features from the column used for a time-aware split.

    * `<col>_elapsed_days`: days since the start of training. Carries the trend,
      so models (linear ones especially) can extrapolate into the future.
    * Calendar parts only when the TRAINING period covers at least two full
      cycles of them. With less than a year of data, "month" simply rises with
      time; a model learns it as a trend and then collapses when January comes
      round again in the test period. The span rule prevents that.
    * No `_year`: elapsed time already captures it, and future years are never
      seen in training.
    """
    t0, t1 = parsed_train.min(), parsed_train.max()
    span_days = (t1 - t0).total_seconds() / 86400.0 if pd.notna(t0) and pd.notna(t1) else 0.0
    has_time_of_day = bool(parsed_train.dt.hour.fillna(0).sum() > 0)
    parts = {
        "weekday": span_days >= 14,
        "day":     span_days >= 60,
        "month":   span_days >= 730,
        "hour":    has_time_of_day and span_days >= 2,
    }
    out = []
    for fr in frames:
        fr = fr.copy()
        parsed = pd.to_datetime(fr[col], errors="coerce")
        fr[f"{col}_elapsed_days"] = (parsed - t0).dt.total_seconds() / 86400.0
        for part, keep in parts.items():
            if keep:
                fr[f"{col}_{part}"] = getattr(parsed.dt, part)
        out.append(fr.drop(columns=[col]))
    names = [f"{col}_elapsed_days"] + [f"{col}_{p}" for p, k in parts.items() if k]
    return out, names, {"column": col, "t0": t0, "parts": [p for p, k in parts.items() if k]}


class RawFeatureBuilder(BaseEstimator, TransformerMixin):
    """Re-creates, on RAW rows, the features that preprocess() derives before the
    ColumnTransformer: date-part columns and the time-split features.

    Placed in front of the ColumnTransformer in the saved bundle so that
    best_model.joblib really does accept raw CSV rows (previously it failed
    with "columns are missing" whenever the data had date columns).
    """

    def __init__(self, datetime_cols=None, hour_cols=None, time_spec=None):
        self.datetime_cols = datetime_cols
        self.hour_cols = hour_cols
        self.time_spec = time_spec

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.datetime_cols:
            X, _ = _extract_datetime_features(X, [c for c in self.datetime_cols if c in X.columns],
                                              hour_cols=set(self.hour_cols or []))
        spec = self.time_spec
        if spec and spec["column"] in X.columns:
            col = spec["column"]
            parsed = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_elapsed_days"] = (parsed - spec["t0"]).dt.total_seconds() / 86400.0
            for part in spec["parts"]:
                X[f"{col}_{part}"] = getattr(parsed.dt, part)
            X = X.drop(columns=[col])
        return X


class InfinityToNaN(BaseEstimator, TransformerMixin):
    """Convert +/-inf to NaN so the downstream imputer can handle it.

    This lives INSIDE the pipeline rather than only in preprocess(), because
    otherwise the persisted artifact would still crash on inference data
    containing infinities - sklearn's imputer rejects non-finite input.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1] if np.ndim(X) > 1 else 1
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype="float64")
        return np.where(np.isfinite(arr), arr, np.nan)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Map each category to its relative frequency in the TRAINING data.

    Why this is a transformer rather than a loose helper: previously the
    frequency lookups lived in a local dict and were never stored on the fitted
    ColumnTransformer. That meant the persisted artifact (preprocessor + model)
    could not transform new raw data containing high-cardinality categoricals -
    the mapping simply wasn't saved. Implementing fit/transform puts the lookups
    inside the pipeline, so they are pickled with everything else.

    Unseen categories map to 0.0 (they were never observed in training).
    """

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.lookups_ = {
            col: X[col].astype(object).value_counts(normalize=True)
            for col in X.columns
        }
        return self

    def transform(self, X):
        X = self._as_frame(X)
        out = pd.DataFrame(index=X.index)
        for col in self.feature_names_in_:
            lookup = self.lookups_[col]
            out[col] = (
                X[col].astype(object).map(lookup).astype("float64").fillna(0.0)
            )
        return out.to_numpy(dtype="float64")

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)

    @staticmethod
    def _as_frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _feature_names_from(preprocessor: ColumnTransformer,
                        numeric_cols: list[str],
                        low_card_cat: list[str],
                        high_card_cat: list[str]) -> list[str]:
    """Read output feature names off the fitted ColumnTransformer.

    Falls back to manual reconstruction only if the sklearn version in use
    cannot report them.
    """
    try:
        raw = list(preprocessor.get_feature_names_out())
        cleaned = []
        for name in raw:
            text = str(name)
            for prefix in ("num__", "cat__", "freq__"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            cleaned.append(text)
        return cleaned
    except Exception:
        names = list(numeric_cols)
        if low_card_cat:
            try:
                ohe = preprocessor.named_transformers_["cat"].named_steps["encode"]
                names.extend(ohe.get_feature_names_out(low_card_cat).tolist())
            except Exception:
                names.extend(low_card_cat)
        names.extend(high_card_cat)
        return names


# ---- main preprocessing API -------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    target: str,
    *,
    df_val: pd.DataFrame | None = None,
    df_test: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "random",        # "random" | "time"
    time_column: str | None = None,
    group_column: str | None = None,
    log_transform_target: bool | str = False,   # False | True | "auto"
    auto_datetime_features: bool = True,
    drop_low_variance: bool = True,
    auto_drop_id_columns: bool = True,
    drop_duplicate_rows: bool = True,
    add_missing_indicators: bool = False,
    add_interactions: bool = False,
) -> PreprocessingResult:
    """End-to-end preprocessing.

    The two main paths:
      * `df_test` is None -> we split `df` ourselves (random or time-aware)
      * `df_test` is provided -> no split, use it directly. `df_val` optional.

    `log_transform_target="auto"` applies log1p only when the TRAINING target
    is right-skewed (skew > 1.5) and non-negative. `add_interactions=True`
    adds pairwise products of numeric features, which lets linear models
    capture effects such as "size matters more downtown" (a common cause of
    underfitting).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in training data.")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(
            f"Target '{target}' is not numeric. This system handles regression only."
        )
    if df_test is not None and target not in df_test.columns:
        raise ValueError(f"Target '{target}' missing from the test file.")
    if df_val is not None and target not in df_val.columns:
        raise ValueError(f"Target '{target}' missing from the validation file.")

    # step 0b: replace +/-inf with NaN so the imputer can actually handle them.
    # SimpleImputer does NOT treat inf as missing, so an infinity survived
    # imputation and then turned an entire scaled column into NaN downstream.
    def _sanitize_inf(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        numeric = frame.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            return frame, 0
        mask = np.isinf(frame[numeric].to_numpy(dtype="float64", na_value=np.nan))
        count = int(mask.sum())
        if count:
            frame = frame.copy()
            frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        return frame, count

    df, n_inf = _sanitize_inf(df)
    if df_val is not None:
        df_val, extra = _sanitize_inf(df_val);  n_inf += extra
    if df_test is not None:
        df_test, extra = _sanitize_inf(df_test); n_inf += extra

    # step 1: drop rows where target is NaN (in every provided frame)
    df = df.dropna(subset=[target]).copy()
    if df_val  is not None: df_val  = df_val.dropna(subset=[target]).copy()
    if df_test is not None: df_test = df_test.dropna(subset=[target]).copy()

    # step 1b: drop exact duplicate rows BEFORE splitting.
    # Identical rows landing on both sides of the split mean the model has
    # literally already seen the answer, which inflates every test metric.
    n_duplicates_dropped = 0
    if drop_duplicate_rows:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_duplicates_dropped = before - len(df)

    # step 2: optional datetime feature extraction (apply identically to all frames).
    # IMPORTANT: do not extract from `time_column` if a time-aware split is requested —
    # that column needs to stay intact for sorting.
    profile = profile_dataframe(df, target)

    # step 2a: drop ID-like columns BEFORE anything else — these almost always leak
    # information when the dataset is sorted (e.g. pandas' default "Unnamed: 0"
    # column when the data was indexed by price).
    dropped_id_cols: list[str] = []
    if auto_drop_id_columns and profile["id_like_columns"]:
        # do not drop the time_column even if it looks ID-like — we need it for sorting
        keepers = {time_column} if time_column else set()
        dropped_id_cols = [c for c in profile["id_like_columns"] if c not in keepers]
        if dropped_id_cols:
            df = df.drop(columns=[c for c in dropped_id_cols if c in df.columns])
            if df_val  is not None: df_val  = df_val.drop( columns=[c for c in dropped_id_cols if c in df_val.columns])
            if df_test is not None: df_test = df_test.drop(columns=[c for c in dropped_id_cols if c in df_test.columns])
            # re-profile after dropping
            profile = profile_dataframe(df, target)

    datetime_cols_used: list[str] = []
    hour_cols: set[str] = set()
    time_spec: dict | None = None
    if auto_datetime_features and profile["datetime_candidates"]:
        protected = {time_column} if (split_strategy == "time" and time_column) else set()
        datetime_cols_used = [c for c in profile["datetime_candidates"] if c not in protected]
        if datetime_cols_used:
            df, hour_cols = _extract_datetime_features(df, datetime_cols_used)
            if df_val  is not None: df_val,  _ = _extract_datetime_features(df_val,  datetime_cols_used, hour_cols=hour_cols)
            if df_test is not None: df_test, _ = _extract_datetime_features(df_test, datetime_cols_used, hour_cols=hour_cols)

    # step 3: optional drop of zero-variance columns
    dropped_lowvar: list[str] = []
    if drop_low_variance:
        for col in list(df.columns):
            if col == target:
                continue
            if col in (time_column, group_column):
                continue  # needed for splitting / CV, not used as a feature
            if _is_low_variance(df[col]):
                dropped_lowvar.append(col)
        if dropped_lowvar:
            df = df.drop(columns=dropped_lowvar)
            if df_val  is not None: df_val  = df_val.drop( columns=[c for c in dropped_lowvar if c in df_val.columns])
            if df_test is not None: df_test = df_test.drop(columns=[c for c in dropped_lowvar if c in df_test.columns])

    # step 4: split (or use the user-provided splits)
    groups_train = None
    if df_test is not None:
        train_df = df.copy()
        test_df  = df_test.copy()
        val_df   = df_val.copy() if df_val is not None else None
        split_used = "user-supplied"
    else:
        if split_strategy == "time":
            if time_column is None or time_column not in df.columns:
                # fall back to random
                split_used = "random (time column missing)"
                train_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=random_state
                )
            else:
                df_sorted = df.sort_values(time_column).reset_index(drop=True)
                cut = int(len(df_sorted) * (1 - test_size))
                train_df, test_df = df_sorted.iloc[:cut], df_sorted.iloc[cut:]
                split_used = f"time-aware (sorted by '{time_column}')"
        elif group_column and group_column in df.columns:
            # Group-aware split. Previously the group column only protected the
            # CV folds, while the train/test split was still plain random - so
            # the same patient / store / user could appear on both sides and the
            # test score measured memorisation rather than generalisation.
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_idx, test_idx = next(
                splitter.split(df, groups=df[group_column].to_numpy())
            )
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df  = df.iloc[test_idx].reset_index(drop=True)
            split_used = f"group-aware (grouped by '{group_column}')"
        else:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            split_used = "random"
        val_df = None

    # step 5: pull out target + groups, prepare X frames
    def _pop(frame, with_groups=False):
        y = frame[target].values.astype(float)
        X = frame.drop(columns=[target])
        g = None
        if with_groups and group_column and group_column in X.columns:
            g = X[group_column].values
            X = X.drop(columns=[group_column])
        elif group_column and group_column in X.columns:
            X = X.drop(columns=[group_column])
        return X, y, g

    X_train_raw, y_train, groups_train = _pop(train_df, with_groups=True)
    X_test_raw,  y_test,  _            = _pop(test_df)
    X_val_raw, y_val = None, None
    if val_df is not None:
        X_val_raw, y_val, _ = _pop(val_df)

    # After a time-aware split the time column has done its sorting job, but
    # dropping it outright (as before) hid any TREND from every model: future
    # test rows then sit above/below anything seen in training and all models
    # underfit. Instead, turn it into features: elapsed time since the start of
    # training (lets linear models extrapolate a trend) plus calendar parts.
    time_features: list[str] = []
    if time_column and time_column in X_train_raw.columns and split_strategy == "time":
        frames = [X_train_raw, X_test_raw] + ([X_val_raw] if X_val_raw is not None else [])
        col = X_train_raw[time_column]
        if pd.api.types.is_numeric_dtype(col):
            time_features = [time_column]          # already a usable trend feature
        else:
            parsed_train = pd.to_datetime(col, errors="coerce")
            if parsed_train.notna().mean() > 0.85:
                frames, time_features, time_spec = _time_split_features(frames, time_column,
                                                                         parsed_train)
            else:
                frames = [fr.drop(columns=[time_column]) for fr in frames]
        X_train_raw, X_test_raw = frames[0], frames[1]
        if X_val_raw is not None:
            X_val_raw = frames[2]

    # step 6: classify columns now that all extra cols are removed
    numeric_cols, low_card_cat, high_card_cat = [], [], []
    for col in X_train_raw.columns:
        if pd.api.types.is_numeric_dtype(X_train_raw[col]):
            numeric_cols.append(col)
        elif X_train_raw[col].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD:
            high_card_cat.append(col)
        else:
            low_card_cat.append(col)

    # step 7 + 8: build ONE ColumnTransformer that owns every learned mapping.
    #
    # Frequency encoding used to happen here as a loose loop, with the lookups
    # kept in a local dict that was never persisted. The saved artifact was
    # therefore unable to transform new raw data. FrequencyEncoder now lives
    # inside the pipeline, so fitting it stores the lookups and pickling the
    # preprocessor carries them along.
    numeric_steps = [
        ("finite", InfinityToNaN()),
        ("impute", SimpleImputer(strategy="median",
                                 add_indicator=add_missing_indicators)),
        ("scale",  StandardScaler()),
    ]
    interactions_used = bool(add_interactions and
                             2 <= len(numeric_cols) <= MAX_INTERACTION_BASE_COLUMNS)
    if interactions_used:
        numeric_steps.append(("interact", PolynomialFeatures(degree=2, interaction_only=True,
                                                             include_bias=False)))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    freq_pipe = Pipeline([
        ("freq",  FrequencyEncoder()),
        ("scale", StandardScaler()),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if low_card_cat:
        transformers.append(("cat", cat_pipe, low_card_cat))
    if high_card_cat:
        transformers.append(("freq", freq_pipe, high_card_cat))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    X_train = preprocessor.fit_transform(X_train_raw)
    raw_preprocessor = Pipeline([
        ("raw_features", RawFeatureBuilder(datetime_cols=list(datetime_cols_used),
                                           hour_cols=sorted(hour_cols), time_spec=time_spec)),
        ("columns", preprocessor),
    ])
    X_test  = preprocessor.transform(X_test_raw)
    X_val   = preprocessor.transform(X_val_raw) if X_val_raw is not None else None

    # step 9: optional target log-transform (applied AFTER the split)
    target_transform: Literal["none", "log1p"] = "none"
    auto_log = isinstance(log_transform_target, str) and log_transform_target.lower() == "auto"
    train_skew = float(pd.Series(y_train).skew()) if len(y_train) > 2 else 0.0
    count_like = bool(len(y_train) and (y_train >= 0).all()
                      and np.allclose(y_train, np.round(y_train)))
    if auto_log:
        # decided from TRAINING data only; left skew is not helped by log1p.
        # Count targets (non-negative integers) stay on their natural scale so
        # PoissonRegressor, which is built for counts, can be used.
        want_log = train_skew > LOG_TRANSFORM_SKEW_THRESHOLD and not count_like
    else:
        want_log = bool(log_transform_target)
    if want_log:
        if (y_train < 0).any() or (y_test < 0).any() or (y_val is not None and (y_val < 0).any()):
            # log1p needs non-negative values
            target_transform = "none"
        else:
            y_train = np.log1p(y_train)
            y_test  = np.log1p(y_test)
            if y_val is not None:
                y_val = np.log1p(y_val)
            target_transform = "log1p"

    # step 10: feature names, taken FROM THE FITTED TRANSFORMER.
    # Rebuilding this list by hand meant it could silently drift out of sync
    # with the actual column order, mislabelling every feature-importance chart.
    # Asking the fitted object is the only way to stay correct when the
    # transformer grows extra outputs (e.g. missing-value indicators).
    feature_names = _feature_names_from(preprocessor, numeric_cols,
                                        low_card_cat, high_card_cat)
    if len(feature_names) != X_train.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    summary = {
        "n_train": int(X_train.shape[0]),
        "n_val":   int(X_val.shape[0]) if X_val is not None else 0,
        "n_test":  int(X_test.shape[0]),
        "n_features_after": int(X_train.shape[1]),
        "numeric_cols": numeric_cols,
        "low_cardinality_categorical": low_card_cat,
        "high_cardinality_categorical_freq_encoded": high_card_cat,
        "datetime_cols_extracted": datetime_cols_used,
        "low_variance_dropped": dropped_lowvar,
        "id_like_columns_dropped": dropped_id_cols,
        "split_strategy_used": split_used,
        "target_transform": target_transform,
        "target_transform_requested": ("auto" if auto_log else bool(log_transform_target)),
        "target_skew_train": train_skew,
        "target_count_like": count_like,
        "interaction_features_added": interactions_used,
        "time_features_from_split_column": time_features,
        "test_size": test_size,
        "random_state": random_state,
        "group_column": group_column if group_column and groups_train is not None else None,
        "duplicate_rows_dropped": n_duplicates_dropped,
        "infinite_values_sanitized": n_inf,
        "missing_indicators_added": bool(add_missing_indicators),
    }

    return PreprocessingResult(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=feature_names, preprocessor=preprocessor, summary=summary,
        X_val=X_val, y_val=y_val,
        target_transform=target_transform, groups_train=groups_train,
        raw_preprocessor=raw_preprocessor,
    )
