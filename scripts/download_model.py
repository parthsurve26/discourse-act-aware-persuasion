"""
Download the trained discourse act classifier from Hugging Face Hub.

Usage:
    uv run python scripts/download_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from huggingface_hub import hf_hub_download

from discourse_act_aware_persuasion.paths import MODELS_DIR

REPO_ID   = "Vijayrathank/discourse_act_classifier"
FILENAME  = "model.pt"
DEST_DIR  = MODELS_DIR / "discourse_act_classifier"

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = DEST_DIR / FILENAME

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Model already downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    print(f"Downloading {REPO_ID}/{FILENAME} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(DEST_DIR),
    )
    size_mb = Path(path).stat().st_size / 1e6
    print(f"Downloaded to: {path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
