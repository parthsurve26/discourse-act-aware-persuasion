# Discourse-Act-Aware Persuasion Modeling on r/ChangeMyView

Clean starter repo for research on persuasion and discourse act modeling with ConvoKit, PyTorch, Hugging Face Transformers, scikit-learn, pandas, pyarrow/parquet, matplotlib, and tqdm.

This first pass focuses on the boring but necessary pieces:

- dataset download
- corpus inspection
- preprocessing
- train/val/test splits
- baseline models
- evaluation utilities

The core datasets in this repo are:

- Winning Arguments corpus
- Cornell Coarse Discourse corpus

Awry CMV is intentionally left out of the core pipeline for now and should be treated as an optional future extension.

## Repository layout

- `scripts/` runnable entry points
- `src/` reusable library code
- `data/` raw-cache pointers, processed parquet, reports, and model outputs
- `notebooks/` exploratory notebooks

## Winning Arguments structure check

Before preprocessing, run the inspection script to verify the live corpus structure in your local ConvoKit install/cache:

```bash
python scripts/inspect_winning_arguments.py
```

Observed with ConvoKit 4.1.0 on the live `winning-args-corpus`:

- each `Conversation` to correspond to one full r/ChangeMyView thread
- conversation metadata to include `op-userID`, `op-text-body`, `op-title`, `pair_ids`, and `train`
- utterance metadata to include `success`, `pair_ids`, and Reddit API fields
- `success` is stored on comments that belong to a labeled argument thread, not on a single standalone prediction item
- `pair_ids` is stored at both conversation and utterance level; some utterances belong to multiple pairs
- each `pair_id` maps to exactly two top-level reply threads in the loaded corpus: one successful and one unsuccessful
- all other comments to have `success = None`
- ConvoKit casts some missing Reddit comment text to the string `"None"` during load; preprocessing drops empty / `"None"` thread text
- the corpus to be loaded with `Corpus(filename=download("winning-args-corpus"))`

The inspection script writes a JSON summary into `data/reports/` so the actual observed fields are documented alongside these assumptions.

## Preprocessing assumptions

The initial preprocessing pipeline makes a few simple, explicit choices:

- the Winning Arguments prediction unit is one row per `(pair_id, top-level reply thread)`, not one row per comment
- the label is whether that paired argument thread changed the OP's view: `1` for successful and `0` for unsuccessful
- text for a Winning Arguments row is the OP title/body plus the labeled comments in that pair member's reply thread
- this produces 8,526 rows in the verified corpus: 4,263 successful and 4,263 unsuccessful pair members
- split by conversation, not by individual utterance, to avoid leakage
- use the dataset-provided `train` flag for Winning Arguments where available
- derive val/test from the remaining holdout conversations deterministically
- keep the first baseline tasks text-centered and readable
- store processed tables as parquet

The important mismatch we found during verification: treating every `success != None` utterance as a separate training example is not faithful to the corpus task. It overweights longer argument threads and breaks the original matched-pair framing. The cleaned preprocessing now keeps the paired argument thread as the prediction unit while preserving `pair_id` for paired analyses.

These assumptions are conservative on purpose. They are meant to get us to a trustworthy baseline before we add richer discourse-aware modeling.

## Setup

Create a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

## First run

1. Download the corpora.
2. Inspect Winning Arguments.
3. Preprocess each dataset to parquet.
4. Train the baselines.

The exact commands are listed at the end of this README.

## Notes on outputs

- raw corpus downloads are kept in ConvoKit’s cache by default
- processed datasets are written to `data/processed/`
- inspection reports are written to `data/reports/`
- baseline artifacts are written to `data/models/`

## Exact commands to run first

```bash
python scripts/download_datasets.py
python scripts/inspect_winning_arguments.py
python scripts/preprocess_datasets.py --dataset winning-arguments
python scripts/preprocess_datasets.py --dataset coarse-discourse
python scripts/train_baselines.py --dataset winning-arguments
python scripts/train_baselines.py --dataset coarse-discourse
```
