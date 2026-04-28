# Colab Run Guide

This guide runs the CMV persuasion ladder on Google Colab with GPU using:

- the repo code from GitHub
- the local Winning Arguments preprocessing logic already in this repo
- the teammate discourse checkpoint `model.pt`

## What You Need Before Starting

1. Push the latest repo code to GitHub.
2. Upload the teammate checkpoint to Google Drive:

`MyDrive/cmv-persuasion-assets/model.pt`

3. Open a new Colab notebook and switch runtime to GPU:

`Runtime -> Change runtime type -> T4 GPU` (or any GPU)

## Recommended Colab Workflow

### Cell 1: Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Cell 2: Clone Repo

Replace `<YOUR_GITHUB_REPO_URL>` if needed.

```bash
!git clone <YOUR_GITHUB_REPO_URL> /content/cmv-persuasion
%cd /content/cmv-persuasion
```

If the repo is already cloned:

```bash
%cd /content/cmv-persuasion
!git pull
```

### Cell 3: Install Dependencies

```bash
!pip install -q -r requirements.txt
```

Optional but useful:

```bash
!python -m nltk.downloader punkt
```

### Cell 4: Put the Teammate Checkpoint in the Expected Place

```bash
!mkdir -p /content/cmv-persuasion/data/models/external
!cp /content/drive/MyDrive/cmv-persuasion-assets/model.pt /content/cmv-persuasion/data/models/external/model.pt
!ls -lh /content/cmv-persuasion/data/models/external
```

### Cell 5: Preprocess Winning Arguments

This regenerates the parquet with the comment-sequence and delta-ack-masked fields used by the BiLSTM models.

```bash
!python scripts/preprocess_datasets.py --dataset winning-arguments
```

### Cell 6: Smoke Test One Model First

```bash
!python scripts/train_persuasion.py \
  --architecture fresh-bilstm-attn \
  --epochs 1 \
  --batch-size 8 \
  --max-comments 16 \
  --max-comment-length 128 \
  --output-dir data/models/colab_smoke
```

If that works, move to the full ladder.

### Cell 7: Full Ladder

```bash
!./scripts/run_persuasion_ladder.sh \
  --epochs 3 \
  --batch-size 8 \
  --max-comments 16 \
  --max-comment-length 128 \
  --output-dir data/models/colab_full
```

### Cell 8: Inspect Results

```bash
!find data/models/colab_full -maxdepth 1 -type f | sort
```

```python
import json
from pathlib import Path

root = Path("/content/cmv-persuasion/data/models/colab_full")
for path in sorted(root.glob("*_metrics.json")):
    payload = json.loads(path.read_text())
    print(path.name)
    print("  accuracy :", round(payload["test"]["accuracy"], 4))
    print("  macro_f1 :", round(payload["test"]["macro_f1"], 4))
    print("  loss     :", round(payload["test"]["loss"], 4))
```

### Cell 9: Copy Outputs Back to Drive

```bash
!mkdir -p /content/drive/MyDrive/cmv-persuasion-runs
!cp -r /content/cmv-persuasion/data/models/colab_full /content/drive/MyDrive/cmv-persuasion-runs/
```

## If You Want A Faster First Pass

Use shorter runs:

```bash
!./scripts/run_persuasion_ladder.sh \
  --epochs 1 \
  --batch-size 8 \
  --max-comments 12 \
  --max-comment-length 96 \
  --output-dir data/models/colab_quick
```

## Notes

- On Colab GPU, do not use `--local-files-only`.
- The discourse-weighted models automatically use:

`data/models/external/model.pt`

- The fresh models use `bert-base-uncased`.
- The Winning Arguments preprocessing now defaults to the pair/thread-level definition already verified in this repo.
