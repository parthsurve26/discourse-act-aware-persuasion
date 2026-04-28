#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/content/cmv-persuasion}"
DRIVE_MODEL_PATH="${2:-/content/drive/MyDrive/cmv-persuasion-assets/model.pt}"

mkdir -p "$REPO_DIR/data/models/external"
cp "$DRIVE_MODEL_PATH" "$REPO_DIR/data/models/external/model.pt"

cd "$REPO_DIR"
python scripts/preprocess_datasets.py --dataset winning-arguments

echo "Colab bootstrap complete."
echo "Checkpoint copied to: $REPO_DIR/data/models/external/model.pt"
echo "Winning Arguments parquet refreshed."
