Use this as your final `README.md`. It keeps the important parts, removes the clutter, and doesn’t make the project look unfinished. It’s based on the repo details you shared: project objective, dataset setup, model list, run flow, and results status. 

````md
# Discourse-Act-Aware Persuasion Modeling on r/ChangeMyView

This project predicts whether an argument thread in r/ChangeMyView successfully changes the original poster’s view, using both text-based and discourse-aware modeling approaches.

The repository includes data ingestion, preprocessing, baseline training, neural modeling, and evaluation workflows for technical review and reproducibility.

## Project Objective

Predict persuasion success at the argument-thread level while keeping the task definition faithful to the original matched-pair setup.

## Repository Structure

```text
scripts/                              Runnable CLI scripts
src/discourse_act_aware_persuasion/   Reusable project code
data/processed/                       Processed parquet datasets
data/reports/                         Inspection and diagnostic outputs
data/models/                          Model artifacts and metrics outputs
docs/                                 Supplemental notes and Colab workflow
````

## Dataset Summary

### Winning Arguments

Primary dataset used for persuasion prediction.

* Prediction unit: one row per `(pair_id, top-level reply thread)`
* Label: `1` = successful persuasion, `0` = unsuccessful persuasion
* Processed snapshot:

  * 8,526 total rows
  * 4,263 positive / 4,263 negative
  * Train, validation, and test splits

### Cornell Coarse Discourse

Used for discourse-act-related experimentation and discourse-aware modeling support.

## Approach

The workflow is organized into clear stages:

1. Download datasets
2. Inspect schema and task assumptions
3. Preprocess data into deterministic parquet outputs
4. Train baseline models
5. Train neural persuasion models
6. Integrate discourse-aware components

A key design choice is keeping the Winning Arguments task at the paired argument-thread level instead of treating individual utterances as independent examples.

## Models

### Baselines

Run with:

```bash
python scripts/train_baselines.py
```

### Persuasion Models

Run with `scripts/train_persuasion.py`:

* `fresh-bert`
* `fresh-bilstm-attn`
* `discourse-bilstm-attn`

### Discourse Encoder

Run with:

```bash
python scripts/train_discourse_encoder.py
```

## Tech Stack

Python, PyTorch, scikit-learn, Hugging Face Transformers, pandas, NumPy, Parquet, NLTK, spaCy

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

## How to Run

Run commands from the repository root.

```bash
python scripts/download_datasets.py
python scripts/inspect_winning_arguments.py

python scripts/preprocess_datasets.py --dataset winning-arguments

python scripts/train_baselines.py --dataset winning-arguments
python scripts/train_discourse_encoder.py

python scripts/train_persuasion.py --architecture fresh-bert
```

For expanded commands and Colab usage, see:

```text
docs/colab_run.md
```

## Results

The repository supports training and evaluation pipelines that write model artifacts and JSON metrics outputs to:

```text
data/models/
```

Final benchmark results will be added after reproducible runs are finalized.

## Reviewer Quickstart

For a quick technical review:

1. Complete setup.
2. Download and inspect the data.
3. Regenerate the Winning Arguments parquet file.
4. Run one baseline model.
5. Run one neural persuasion model.
6. Confirm artifacts and metrics are written to `data/models/`.

## Notes

* Preprocessing assumptions are kept explicit and inspectable.
* The project separates runnable scripts from reusable source code.
* If the prediction-unit logic changes, preprocessing, diagnostics, and documentation should be updated together.

```
```
