# Discourse-Act-Aware Persuasion Modeling on r/ChangeMyView

This project explores whether discourse structure helps predict persuasion success in Reddit discussions from r/ChangeMyView.

The main task is to predict whether a reply thread successfully changes the original poster's view. The repository includes data download, preprocessing, baseline models, neural models, and evaluation utilities.

## What This Repo Contains

- data download and inspection scripts
- preprocessing pipelines for the core datasets
- baseline machine learning models
- neural persuasion models, including BiGRU and transformer-based variants
- discourse-aware modeling components
- analysis and reporting scripts

## Datasets

### Winning Arguments

This is the main dataset used for persuasion prediction.

- task: predict persuasion success
- label: `1` for successful persuasion, `0` for unsuccessful persuasion
- prediction unit: one argument thread

### Cornell Coarse Discourse

This dataset is used for discourse-act related experiments and discourse-aware modeling.

## Models

The repository includes several model types:

- baseline text models
- a BiGRU persuasion model
- transformer-based thread models
- discourse-aware variants that incorporate discourse-act information

## Repository Structure

```text
scripts/                               runnable training and preprocessing scripts
src/discourse_act_aware_persuasion/    reusable project code
coarse_discourse_cls/                  discourse classifier code
data/                                  processed data, reports, and model outputs
notebooks/                             exploratory notebooks
```

## Setup

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

## Quick Start

Download the datasets:

```bash
python scripts/download_datasets.py
```

Inspect the main persuasion dataset:

```bash
python scripts/inspect_winning_arguments.py
```

Preprocess the data:

```bash
python scripts/preprocess_datasets.py --dataset winning-arguments
python scripts/preprocess_datasets.py --dataset coarse-discourse
```

Train a baseline model:

```bash
python scripts/train_baselines.py --dataset winning-arguments
```

Train the BiGRU persuasion model:

```bash
python scripts/download_model.py
python scripts/train_persuasion.py
```

Train a transformer-based persuasion model:

```bash
python scripts/train_winning_args_with_discourse.py
```

## Outputs

The project writes outputs to:

- `data/processed/` for processed datasets
- `data/reports/` for inspection and analysis outputs
- `data/models/` for trained models, checkpoints, and metrics

## Notes

- The project keeps runnable scripts separate from reusable source code.
- The persuasion task is modeled at the argument-thread level.
- Additional analysis scripts are available under `scripts/` for diagnostics and model inspection.
