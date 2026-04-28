from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split


DEFAULT_HF_DATASET_ID = "Vijayrathank/reddit_discourse_cleaned"
DEFAULT_HF_FILENAME = "data/train-00000-of-00001.parquet"


def _clean_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if text.lower() in {"", "none", "[deleted]", "[removed]"}:
        return ""
    return text


def load_huggingface_discourse_dataframe(
    dataset_id: str = DEFAULT_HF_DATASET_ID,
    filename: str = DEFAULT_HF_FILENAME,
    local_path: str | Path | None = None,
) -> pd.DataFrame:
    if local_path is not None:
        source_path = Path(local_path)
    else:
        source_path = Path(
            hf_hub_download(
                repo_id=dataset_id,
                filename=filename,
                repo_type="dataset",
            )
        )

    df = pd.read_parquet(source_path).copy()
    if "comment_depth" in df.columns and "post_depth" not in df.columns:
        df = df.rename(columns={"comment_depth": "post_depth"})
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected Hugging Face discourse data to contain `text` and `label` columns.")

    df["text"] = df["text"].map(_clean_text)
    df = df[df["text"] != ""].reset_index(drop=True)
    if "input_text" not in df.columns:
        df["input_text"] = df["text"]
    if "num_chars" not in df.columns:
        df["num_chars"] = df["text"].str.len()
    if "num_words" not in df.columns:
        df["num_words"] = df["text"].str.split().str.len()
    return df


def ensure_discourse_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if "split" in df.columns:
        observed = set(df["split"].dropna().astype(str).unique().tolist())
        if {"train", "val", "test"}.issubset(observed):
            return df.reset_index(drop=True)

    train_idx, holdout_idx = train_test_split(
        df.index.to_numpy(),
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    holdout = df.loc[holdout_idx]
    val_idx, test_idx = train_test_split(
        holdout.index.to_numpy(),
        test_size=0.5,
        random_state=seed,
        stratify=holdout["label"],
    )

    split_df = df.copy()
    split_df["split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.loc[test_idx, "split"] = "test"
    return split_df.reset_index(drop=True)
