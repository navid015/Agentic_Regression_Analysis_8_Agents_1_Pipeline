"""
Robust table loading.

`pd.read_csv(path)` with defaults silently mis-reads two very common kinds of
file: European exports that use `;` as the delimiter (everything lands in ONE
column) and files saved in Windows-1252 / Latin-1 (UnicodeDecodeError). Both
are handled here without asking the user.
"""

from __future__ import annotations

import csv
import os

import pandas as pd

_CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
_SNIFF_BYTES = 64 * 1024


def _read_head(path: str, encoding: str) -> str:
    with open(path, "r", encoding=encoding, errors="strict", newline="") as fh:
        return fh.read(_SNIFF_BYTES)


def detect_encoding(path: str) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            _read_head(path, enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_delimiter(sample: str) -> str:
    """csv.Sniffer first; fall back to the delimiter that splits lines most consistently."""
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(_CANDIDATE_DELIMITERS)).delimiter
    except csv.Error:
        pass
    lines = [ln for ln in sample.splitlines()[:50] if ln.strip()]
    best, best_score = ",", -1.0
    for d in _CANDIDATE_DELIMITERS:
        counts = [ln.count(d) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        # consistent, non-zero field counts score highest
        score = min(counts) + (1.0 if len(set(counts)) == 1 else 0.0)
        if score > best_score:
            best, best_score = d, score
    return best


def read_table(path: str | None) -> pd.DataFrame | None:
    """Read a CSV/TSV with automatic encoding and delimiter detection."""
    if path is None or not os.path.exists(path):
        return None
    encoding = detect_encoding(path)
    sample = _read_head(path, encoding) if encoding != "latin-1" else \
        open(path, "r", encoding="latin-1", newline="").read(_SNIFF_BYTES)
    sep = detect_delimiter(sample)
    df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    # a 1-column result from a multi-field file means the delimiter guess failed
    if df.shape[1] == 1 and sep != ";" and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";", encoding=encoding, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df
