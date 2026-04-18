from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


def _stable_float(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def assign_group_split(
    df: pd.DataFrame,
    group_col: str,
    split_cfg: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    groups = sorted(df[group_col].dropna().astype(str).unique())
    scored_groups = sorted(groups, key=lambda g: (_stable_float(g, split_cfg.seed), g))

    n_groups = len(scored_groups)
    n_train = int(round(n_groups * split_cfg.train_ratio))
    n_val = int(round(n_groups * split_cfg.val_ratio))
    n_train = min(n_train, n_groups)
    n_val = min(n_val, max(0, n_groups - n_train))
    train_groups = set(scored_groups[:n_train])
    val_groups = set(scored_groups[n_train : n_train + n_val])
    test_groups = set(scored_groups[n_train + n_val :])

    def _split_for_group(value: str) -> str:
        if value in train_groups:
            return "train"
        if value in val_groups:
            return "val"
        if value in test_groups:
            return "test"
        return "train"

    out = df.copy()
    out["split"] = out[group_col].astype(str).map(_split_for_group)
    return out


def assign_holdout_split(
    df: pd.DataFrame,
    group_col: str,
    train_flag_col: str,
    holdout_val_ratio: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    out = df.copy()
    train_mask = out[train_flag_col].fillna(0).astype(int) == 1
    out.loc[train_mask, "split"] = "train"

    holdout_groups = sorted(out.loc[~train_mask, group_col].dropna().astype(str).unique())
    scored_groups = sorted(
        holdout_groups, key=lambda g: (_stable_float(g, seed), g)
    )
    n_holdout = len(scored_groups)
    n_val = int(round(n_holdout * holdout_val_ratio))
    val_groups = set(scored_groups[:n_val])
    test_groups = set(scored_groups[n_val:])

    def _split_for_group(value: str) -> str:
        if value in val_groups:
            return "val"
        if value in test_groups:
            return "test"
        return "train"

    holdout_mask = ~train_mask
    out.loc[holdout_mask, "split"] = out.loc[holdout_mask, group_col].astype(str).map(
        _split_for_group
    )
    return out

