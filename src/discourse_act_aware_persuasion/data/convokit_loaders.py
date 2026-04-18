from __future__ import annotations

from pathlib import Path


WINNING_ARGS_DOWNLOAD_NAME = "winning-args-corpus"
COARSE_DISCOURSE_DOWNLOAD_NAME = "reddit-coarse-discourse-corpus"


def download_corpus(download_name: str) -> Path:
    from convokit import download

    return Path(download(download_name))
