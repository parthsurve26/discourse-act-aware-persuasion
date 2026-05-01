# Discourse-Act-Aware Persuasion Modeling

Research on whether discourse act structure improves persuasion prediction on r/ChangeMyView. The project trains a transformer over Reddit threads and tests whether encoding *how* people argue (discourse acts) — not just *what* they say — predicts argument success.

## How it works

1. A discourse act classifier encodes each comment into an act-aware embedding.
2. A thread-level transformer with transition-aware attention (act-pair, speaker, distance biases) predicts whether an argument will change the OP's view.
3. Ablations isolate the contribution of each signal.

## HuggingFace

| Resource | Hub ID |
|---|---|
| Discourse act classifier weights | [`Vijayrathank/discourse_act_classifier`](https://huggingface.co/Vijayrathank/discourse_act_classifier) |
| Persuasion model weights | [`Vijayrathank/persuasion_model_with_discoarse_cls`](https://huggingface.co/Vijayrathank/persuasion_model_with_discoarse_cls) |
| Reddit discourse dataset | [`Vijayrathank/persuasion_model_without_discoarse_cls`](https://huggingface.co/Vijayrathank/persuasion_model_without_discoarse_cls) |

Weights are downloaded automatically on first run — no manual setup required.

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

## First run

```bash
python scripts/download_datasets.py
python scripts/preprocess_datasets.py --dataset winning-arguments
python scripts/preprocess_datasets.py --dataset coarse-discourse
python scripts/train_baselines.py --dataset winning-arguments
python scripts/train_baselines.py --dataset coarse-discourse
```

## Transformer variants

Three variants of the winning-arguments transformer, each isolating a different set of signals:

| Variant | Encoder | Act-pair bias | Speaker bias | Distance bias |
|---|---|---|---|---|
| Standalone | BERT | — | — | — |
| With discourse | Discourse act | ✓ | ✓ | ✓ |
| No-bias ablation | Discourse act | ✓ | — | — |

**Standalone** — text-only baseline:

```bash
python scripts/train_winning_args_standalone.py
python scripts/train_winning_args_standalone.py --mode eval \
    --ckpt-path data/models/winning_args_standalone.pt
```

**With discourse** — full transition-aware model:

```bash
python scripts/train_winning_args_with_discourse.py
python scripts/train_winning_args_with_discourse.py --mode eval \
    --ckpt-path data/models/winning_args_with_discourse.pt
```

**No-bias ablation** — act-pair signal only:

```bash
python scripts/train_winning_args_no_bias_ablation.py
python scripts/train_winning_args_with_discourse.py --mode eval \
    --no-speaker-bias --no-distance-bias \
    --ckpt-path data/models/winning_args_with_discourse_no_bias.pt
```

## Repository layout

```
scripts/    runnable entry points
src/        reusable library code
data/       processed parquet, reports, model checkpoints
notebooks/  exploratory notebooks
```
