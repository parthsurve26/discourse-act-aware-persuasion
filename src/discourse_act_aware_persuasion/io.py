from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_parent


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    ensure_parent(path)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

