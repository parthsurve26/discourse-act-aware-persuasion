from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..paths import REPORTS_DIR
from ..utils import seed_everything, setup_logging, write_json
from .convokit_loaders import WINNING_ARGS_DOWNLOAD_NAME, download_corpus
from .inspection import inspect_winning_arguments_corpus


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the live Winning Arguments corpus structure and save a report."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "winning_arguments_inspection.json")
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)

    from convokit import Corpus

    corpus = Corpus(filename=str(download_corpus(WINNING_ARGS_DOWNLOAD_NAME)))
    report = inspect_winning_arguments_corpus(corpus)
    write_json(args.output, report)
    LOGGER.info("Wrote inspection report to %s", args.output)
    LOGGER.info("Observed conversation meta keys: %s", report["conversation_meta_keys"])
    LOGGER.info("Observed utterance meta keys: %s", report["utterance_meta_keys"])


if __name__ == "__main__":
    main()
