"""Winning Arguments dataset loading and batching."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data.winning_arguments import DELTA_TOKEN_RE
from ..paths import PROCESSED_DIR


def _parse_json_col(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return json.loads(value)


def _strip_delta_tokens(text: str) -> str:
    return DELTA_TOKEN_RE.sub(" ", text).strip()


@dataclass
class ThreadRecord:
    """A single argument thread, post-masking, ready for the model."""

    comment_texts: List[str]
    comment_is_op: List[bool]
    label: int
    thread_id: str

    def __len__(self) -> int:
        return len(self.comment_texts)


def load_winning_arguments_df(parquet_path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(parquet_path) if parquet_path else PROCESSED_DIR / "winning_arguments.parquet"
    return pd.read_parquet(path)


def df_to_records(
    df: pd.DataFrame,
    *,
    drop_delta_ack: bool = True,
    max_comments: int = 32,
) -> List[ThreadRecord]:
    """Convert the wide parquet rows into ThreadRecord objects.

    Mirrors the masking semantics in
    src/discourse_act_aware_persuasion/data/winning_arguments.py:99-103 —
    delta-ack comments are dropped, and any residual ∆/!delta tokens are
    stripped from the surviving comment texts as a defence-in-depth check.
    """
    records: List[ThreadRecord] = []
    for _, row in df.iterrows():
        comment_texts = _parse_json_col(row["comment_texts"])
        comment_is_op = _parse_json_col(row["comment_is_op"])
        comment_is_delta_ack = _parse_json_col(row["comment_is_delta_ack"])

        kept_texts: List[str] = []
        kept_is_op: List[bool] = []
        for txt, is_op, is_ack in zip(comment_texts, comment_is_op, comment_is_delta_ack):
            if drop_delta_ack and is_ack:
                continue
            cleaned = _strip_delta_tokens(str(txt))
            if not cleaned:
                continue
            kept_texts.append(cleaned)
            kept_is_op.append(bool(is_op))

        if not kept_texts:
            continue

        if len(kept_texts) > max_comments:
            kept_texts = kept_texts[-max_comments:]
            kept_is_op = kept_is_op[-max_comments:]

        records.append(
            ThreadRecord(
                comment_texts=kept_texts,
                comment_is_op=kept_is_op,
                label=int(row["label"]),
                thread_id=str(row["argument_thread_id"]),
            )
        )
    return records


class WinningArgumentsDataset(Dataset):
    """Indexable view over a list of ThreadRecord objects."""

    def __init__(self, records: Sequence[ThreadRecord]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> ThreadRecord:
        return self.records[idx]


def collate_threads(batch: List[ThreadRecord]) -> Dict[str, object]:
    """Variable-length collation. Returns Python lists for text fields and
    a tensor for `is_op` padded with 0 plus a `comment_mask` (1 = real, 0 = pad).
    Tokenization is the encoder's responsibility.
    """
    max_n = max(len(r) for r in batch)
    is_op = torch.zeros(len(batch), max_n, dtype=torch.long)
    mask = torch.zeros(len(batch), max_n, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)
    texts: List[List[str]] = []

    for i, r in enumerate(batch):
        n = len(r)
        is_op[i, :n] = torch.tensor([int(x) for x in r.comment_is_op], dtype=torch.long)
        mask[i, :n] = True
        labels[i] = r.label
        texts.append(list(r.comment_texts))

    return {
        "comment_texts": texts,        # List[List[str]] — ragged, length B; inner length ≤ max_n
        "is_op": is_op,                # (B, max_n) long
        "comment_mask": mask,          # (B, max_n) bool
        "labels": labels,              # (B,)
    }


def split_records(
    df: pd.DataFrame,
    *,
    drop_delta_ack: bool = True,
    max_comments: int = 32,
) -> Dict[str, List[ThreadRecord]]:
    """Split parquet rows by the precomputed `split` column."""
    out: Dict[str, List[ThreadRecord]] = {"train": [], "val": [], "test": []}
    for split_name in out:
        sub = df[df["split"] == split_name]
        out[split_name] = df_to_records(
            sub, drop_delta_ack=drop_delta_ack, max_comments=max_comments
        )
    return out
