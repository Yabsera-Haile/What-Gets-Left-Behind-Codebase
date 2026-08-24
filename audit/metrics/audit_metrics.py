from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _counts(df: pd.DataFrame, selected_ids: Iterable[str], group_col: str):
    selected_set = set(selected_ids)
    pool_counts = df[group_col].value_counts()
    kept_mask = df["id"].isin(selected_set)
    kept_counts = df.loc[kept_mask, group_col].value_counts()
    kept_counts = kept_counts.reindex(pool_counts.index, fill_value=0)
    return pool_counts, kept_counts


def _shannon_entropy(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def representation_ratio(df, selected_ids, group_col) -> dict:
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    total_pool = float(pool_counts.sum())
    total_kept = float(kept_counts.sum())
    out = {}
    for g in pool_counts.index:
        pool_share = pool_counts[g] / total_pool if total_pool else 0.0
        if total_kept == 0 or pool_share == 0:
            out[g] = 0.0
        else:
            kept_share = kept_counts[g] / total_kept
            out[g] = kept_share / pool_share
    return out


def retention_rate(df, selected_ids, group_col) -> dict:
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    return {
        g: (kept_counts[g] / pool_counts[g]) if pool_counts[g] else 0.0
        for g in pool_counts.index
    }


def coverage(df, selected_ids, group_col) -> float:
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    n_groups = len(pool_counts)
    if n_groups == 0:
        return 0.0
    covered = int((kept_counts > 0).sum())
    return covered / n_groups


def group_entropy(df, selected_ids, group_col) -> tuple:
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    return _shannon_entropy(kept_counts), _shannon_entropy(pool_counts)


def gini(df, selected_ids, group_col) -> float:
    _, kept_counts = _counts(df, selected_ids, group_col)
    values = sorted(float(v) for v in kept_counts.values)
    n = len(values)
    total = sum(values)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * x for i, x in enumerate(values))
    return (2.0 * weighted) / (n * total) - (n + 1) / n


def summarise(df, selected_ids, group_col) -> pd.DataFrame:
    pool_counts, kept_counts = _counts(df, selected_ids, group_col)
    total_pool = float(pool_counts.sum())
    total_kept = float(kept_counts.sum())
    rep = representation_ratio(df, selected_ids, group_col)
    ret = retention_rate(df, selected_ids, group_col)

    rows = []
    for g in pool_counts.index:
        rows.append({
            "group": g,
            "pool_count": int(pool_counts[g]),
            "pool_share": (pool_counts[g] / total_pool) if total_pool else 0.0,
            "kept_count": int(kept_counts[g]),
            "kept_share": (kept_counts[g] / total_kept) if total_kept else 0.0,
            "retention_rate": ret[g],
            "representation_ratio": rep[g],
        })
    out = pd.DataFrame(rows, columns=[
        "group", "pool_count", "pool_share", "kept_count", "kept_share",
        "retention_rate", "representation_ratio",
    ])
    return out.sort_values("representation_ratio", ascending=True).reset_index(drop=True)
