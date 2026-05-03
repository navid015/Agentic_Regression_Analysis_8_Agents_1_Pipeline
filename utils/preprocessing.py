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

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HIGH_CARDINALITY_THRESHOLD = 20
LOW_VARIANCE_NUNIQUE = 1  # column has only 1 unique value -> drop
LOG_TRANSFORM_SKEW_THRESHOLD = 1.5  # |skew| above this triggers auto log-suggestion
ID_COLUMN_NAME_HINTS = ("unnamed:", "id", "index", "rowid", "row_id", "row_num", "_id")


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
        # ID-like detection: name hint OR sequential integer pattern
        name_hint = any(h in col.lower() for h in ID_COLUMN_NAME_HINTS)
        if name_hint or _is_id_like_column(s):
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
        if info["unique"] <= LOW_VARIANCE_NUNIQUE:
            profile["low_variance_columns"].append(col)
        profile["columns"].append(info)
    return profile


# ---- datetime feature extraction --------------------------------------------


def _extract_datetime_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """For each datetime column, replace it with year/month/day/weekday/hour."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}_year"]    = parsed.dt.year
        out[f"{col}_month"]   = parsed.dt.month
        out[f"{col}_day"]     = parsed.dt.day
        out[f"{col}_weekday"] = parsed.dt.weekday
        # only add hour if at least some values have non-zero hours
        if parsed.dt.hour.fillna(0).sum() > 0:
            out[f"{col}_hour"] = parsed.dt.hour
        out = out.drop(columns=[col])
    return out


def _frequency_encode_inplace(df: pd.DataFrame, col: str, lookup: pd.Series | None = None
                              ) -> pd.Series:
    """Frequency-encode a column. If `lookup` given, use it (for val/test)."""
    if lookup is None:
        counts = df[col].value_counts(normalize=True)
    else:
        counts = lookup
    df[col] = df[col].map(counts).fillna(0.0)
    return counts


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
    log_transform_target: bool = False,
    auto_datetime_features: bool = True,
    drop_low_variance: bool = True,
    auto_drop_id_columns: bool = True,
) -> PreprocessingResult:
    """End-to-end preprocessing.

    The two main paths:
      * `df_test` is None -> we split `df` ourselves (random or time-aware)
      * `df_test` is provided -> no split, use it directly. `df_val` optional.
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

    # step 1: drop rows where target is NaN (in every provided frame)
    df = df.dropna(subset=[target]).copy()
    if df_val  is not None: df_val  = df_val.dropna(subset=[target]).copy()
    if df_test is not None: df_test = df_test.dropna(subset=[target]).copy()

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
    if auto_datetime_features and profile["datetime_candidates"]:
        protected = {time_column} if (split_strategy == "time" and time_column) else set()
        datetime_cols_used = [c for c in profile["datetime_candidates"] if c not in protected]
        if datetime_cols_used:
            df = _extract_datetime_features(df, datetime_cols_used)
            if df_val  is not None: df_val  = _extract_datetime_features(df_val,  datetime_cols_used)
            if df_test is not None: df_test = _extract_datetime_features(df_test, datetime_cols_used)

    # step 3: optional drop of zero-variance columns
    dropped_lowvar: list[str] = []
    if drop_low_variance:
        for col in list(df.columns):
            if col == target:
                continue
            if df[col].nunique(dropna=True) <= LOW_VARIANCE_NUNIQUE:
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

    # also drop the time column from X if it was only used for sorting
    if time_column and time_column in X_train_raw.columns and split_strategy == "time":
        X_train_raw = X_train_raw.drop(columns=[time_column])
        X_test_raw  = X_test_raw.drop( columns=[time_column])
        if X_val_raw is not None:
            X_val_raw = X_val_raw.drop(columns=[time_column])

    # step 6: classify columns now that all extra cols are removed
    numeric_cols, low_card_cat, high_card_cat = [], [], []
    for col in X_train_raw.columns:
        if pd.api.types.is_numeric_dtype(X_train_raw[col]):
            numeric_cols.append(col)
        elif X_train_raw[col].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD:
            high_card_cat.append(col)
        else:
            low_card_cat.append(col)

    # step 7: frequency-encode high-card categoricals using TRAIN frequencies
    freq_lookups: dict[str, pd.Series] = {}
    for col in high_card_cat:
        lookup = X_train_raw[col].astype(object).value_counts(normalize=True)
        freq_lookups[col] = lookup
        X_train_raw[col] = X_train_raw[col].map(lookup).fillna(0.0)
        X_test_raw[col]  = X_test_raw[col].map(lookup).fillna(0.0)
        if X_val_raw is not None:
            X_val_raw[col] = X_val_raw[col].map(lookup).fillna(0.0)
        numeric_cols.append(col)

    # step 8: build the ColumnTransformer
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if low_card_cat:
        transformers.append(("cat", cat_pipe, low_card_cat))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)
    X_val   = preprocessor.transform(X_val_raw) if X_val_raw is not None else None

    # step 9: optional target log-transform (applied AFTER the split)
    target_transform: Literal["none", "log1p"] = "none"
    if log_transform_target:
        if (y_train < 0).any() or (y_test < 0).any() or (y_val is not None and (y_val < 0).any()):
            # log1p needs non-negative values
            target_transform = "none"
        else:
            y_train = np.log1p(y_train)
            y_test  = np.log1p(y_test)
            if y_val is not None:
                y_val = np.log1p(y_val)
            target_transform = "log1p"

    # step 10: build feature-name list
    feature_names: list[str] = list(numeric_cols)
    if low_card_cat:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encode"]
        feature_names.extend(ohe.get_feature_names_out(low_card_cat).tolist())

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
        "test_size": test_size,
        "random_state": random_state,
        "group_column": group_column if group_column and groups_train is not None else None,
    }

    return PreprocessingResult(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=feature_names, preprocessor=preprocessor, summary=summary,
        X_val=X_val, y_val=y_val,
        target_transform=target_transform, groups_train=groups_train,
    )
