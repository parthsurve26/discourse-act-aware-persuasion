from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..paths import RAW_DIR
from ..utils import setup_logging, write_json
from .convokit_loaders import (
    COARSE_DISCOURSE_DOWNLOAD_NAME,
    WINNING_ARGS_DOWNLOAD_NAME,
    download_corpus,
)


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the core ConvoKit corpora.")
    parser.add_argument("--output", type=Path, default=RAW_DIR / "convokit_downloads.json")
    args = parser.parse_args()

    setup_logging()
    raw_paths = {
        "winning-arguments": str(download_corpus(WINNING_ARGS_DOWNLOAD_NAME)),
        "coarse-discourse": str(download_corpus(COARSE_DISCOURSE_DOWNLOAD_NAME)),
    }
    write_json(args.output, raw_paths)
    LOGGER.info("Downloaded corpora and wrote paths to %s", args.output)
    for name, path in raw_paths.items():
        LOGGER.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
