from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..io import write_parquet
from ..paths import PROCESSED_DIR, REPORTS_DIR
from ..utils import seed_everything, setup_logging, write_json
from .coarse_discourse import build_coarse_discourse_dataframe
from .convokit_loaders import (
    COARSE_DISCOURSE_DOWNLOAD_NAME,
    WINNING_ARGS_DOWNLOAD_NAME,
    download_corpus,
)
from .winning_arguments import build_winning_arguments_dataframe


LOGGER = logging.getLogger(__name__)


def load_corpus(download_name: str):
    from convokit import Corpus

    return Corpus(filename=str(download_corpus(download_name)))


def preprocess_winning_arguments(output_path: Path, seed: int = 42) -> Path:
    corpus = load_corpus(WINNING_ARGS_DOWNLOAD_NAME)
    df = build_winning_arguments_dataframe(corpus, seed=seed)
    write_parquet(df, output_path)
    write_json(
        REPORTS_DIR / "winning_arguments_preprocess_summary.json",
        {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "split_counts": df["split"].value_counts(dropna=False).to_dict(),
            "label_counts": df["label"].value_counts(dropna=False).to_dict(),
            "num_delta_ack_comments_total": int(df["num_delta_ack_comments"].sum()),
            "rows_with_any_delta_ack": int((df["num_delta_ack_comments"] > 0).sum()),
            "rows_with_any_delta_ack_by_label": (
                df.assign(any_ack=df["num_delta_ack_comments"] > 0)
                .groupby("label")["any_ack"]
                .sum()
                .to_dict()
            ),
        },
    )
    return output_path


def preprocess_coarse_discourse(output_path: Path, seed: int = 42) -> Path:
    corpus = load_corpus(COARSE_DISCOURSE_DOWNLOAD_NAME)
    df = build_coarse_discourse_dataframe(corpus, seed=seed)
    if df.empty or "split" not in df.columns:
        raise ValueError(
            "Coarse Discourse preprocessing produced no split column. "
            "Check that the corpus metadata contains majority_type labels."
        )
    write_parquet(df, output_path)
    write_json(
        REPORTS_DIR / "coarse_discourse_preprocess_summary.json",
        {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "split_counts": df["split"].value_counts(dropna=False).to_dict(),
            "label_counts": df["label"].value_counts(dropna=False).to_dict(),
        },
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ConvoKit corpora to parquet.")
    parser.add_argument(
        "--dataset",
        choices=["winning-arguments", "coarse-discourse", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("winning-arguments", "all"):
        out = args.output_dir / "winning_arguments.parquet"
        LOGGER.info("Preprocessing Winning Arguments -> %s", out)
        preprocess_winning_arguments(out, seed=args.seed)

    if args.dataset in ("coarse-discourse", "all"):
        out = args.output_dir / "coarse_discourse.parquet"
        LOGGER.info("Preprocessing Coarse Discourse -> %s", out)
        preprocess_coarse_discourse(out, seed=args.seed)


if __name__ == "__main__":
    main()
