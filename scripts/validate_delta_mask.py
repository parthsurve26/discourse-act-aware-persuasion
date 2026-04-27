from _bootstrap import *  # noqa: F401,F403

import json
import re
from pathlib import Path

import pandas as pd

from discourse_act_aware_persuasion.paths import PROCESSED_DIR


DELTA_RE = re.compile(r"(?:∆|Δ|!delta|&#8710;|&#916;)")


def main() -> None:
    parquet_path = PROCESSED_DIR / "winning_arguments.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"Columns: {list(df.columns)}\n")

    list_cols = [
        "comment_ids",
        "comment_texts",
        "comment_speakers",
        "comment_is_op",
        "comment_is_delta_ack",
    ]
    parsed = {c: df[c].apply(json.loads) for c in list_cols}
    lengths = pd.DataFrame({c: parsed[c].str.len() for c in list_cols})
    aligned = lengths.eq(df["num_kept_comments"], axis=0).all(axis=1)
    print(f"Alignment OK for all rows: {bool(aligned.all())}")
    if not aligned.all():
        bad = df.loc[~aligned, ["pair_id", "argument_thread_id", "num_kept_comments"]]
        print("Misaligned rows (first 5):")
        print(bad.head())
    print()

    by_label = (
        df.assign(any_ack=df["num_delta_ack_comments"] > 0)
        .groupby("label")["any_ack"]
        .agg(["sum", "mean", "count"])
    )
    print("rows_with_any_delta_ack by label:")
    print(by_label)
    print()

    text_hits = df["text"].str.contains(DELTA_RE, regex=True).sum()
    masked_hits = df["text_ack_masked"].str.contains(DELTA_RE, regex=True).sum()
    print(f"Rows with delta token in `text`            : {int(text_hits)}")
    print(f"Rows with delta token in `text_ack_masked` : {int(masked_hits)}")
    print()

    sample_pairs = ["p_1"]
    for pid in sample_pairs:
        sub = df[df["pair_id"] == pid]
        if sub.empty:
            print(f"[skip] pair_id={pid} not found")
            continue
        print(f"--- pair_id = {pid} ---")
        for _, row in sub.iterrows():
            ack_flags = json.loads(row["comment_is_delta_ack"])
            n_ack = sum(ack_flags)
            print(
                f"  label={row['label']} kept={row['num_kept_comments']} "
                f"ack={row['num_delta_ack_comments']} "
                f"chars: {row['num_chars']} -> {row['num_chars_ack_masked']}"
            )
            if n_ack:
                texts = json.loads(row["comment_texts"])
                speakers = json.loads(row["comment_speakers"])
                for txt, spk, is_ack in zip(texts, speakers, ack_flags):
                    if is_ack:
                        snippet = txt.replace("\n", " ")[:200]
                        print(f"    [MASKED by {spk}] {snippet}")
        print()

    winning = df[df["label"] == 1]
    if not winning.empty:
        sample = winning.sample(min(3, len(winning)), random_state=0)
        print("--- Random winning-thread spot checks ---")
        for _, row in sample.iterrows():
            ack_flags = json.loads(row["comment_is_delta_ack"])
            if not any(ack_flags):
                print(f"  pair={row['pair_id']} (no delta-ack in this thread)")
                continue
            texts = json.loads(row["comment_texts"])
            speakers = json.loads(row["comment_speakers"])
            print(f"  pair={row['pair_id']} op={row['op_user_id']}")
            for txt, spk, is_ack in zip(texts, speakers, ack_flags):
                if is_ack:
                    snippet = txt.replace("\n", " ")[:240]
                    print(f"    [MASKED] speaker={spk}: {snippet}")
            print()


if __name__ == "__main__":
    main()
