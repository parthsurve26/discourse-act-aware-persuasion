from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from .evaluation import classification_metrics, plot_confusion_matrix, save_metrics
from .io import read_parquet
from .models import text_baseline_pipeline, text_plus_numeric_pipeline
from .paths import MODELS_DIR, PROCESSED_DIR
from .utils import seed_everything, setup_logging


LOGGER = logging.getLogger(__name__)


def _task_config(dataset: str):
    if dataset == "winning-arguments":
        return {
            "path": PROCESSED_DIR / "winning_arguments.parquet",
            "label_col": "label",
            "text_col": "input_text",
            "numeric": True,
            "labels": [0, 1],
        }
    if dataset == "coarse-discourse":
        return {
            "path": PROCESSED_DIR / "coarse_discourse.parquet",
            "label_col": "label",
            "text_col": "input_text",
            "numeric": False,
            "labels": None,
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def train(dataset: str, seed: int = 42, output_dir: Path = MODELS_DIR) -> dict:
    cfg = _task_config(dataset)
    df = read_parquet(cfg["path"])

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Expected non-empty train/val/test splits in {cfg['path']}. "
            "Run preprocessing first and verify the split columns."
        )

    if cfg["numeric"] and dataset == "winning-arguments":
        model = text_plus_numeric_pipeline(text_col=cfg["text_col"])
    else:
        model = text_baseline_pipeline()

    if cfg["numeric"]:
        train_x = train_df
        val_x = val_df
        test_x = test_df
    else:
        train_x = train_df[cfg["text_col"]]
        val_x = val_df[cfg["text_col"]]
        test_x = test_df[cfg["text_col"]]

    model.fit(train_x, train_df[cfg["label_col"]])
    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    val_metrics = classification_metrics(val_df[cfg["label_col"]], val_pred, labels=cfg["labels"])
    test_metrics = classification_metrics(test_df[cfg["label_col"]], test_pred, labels=cfg["labels"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{dataset}_baseline.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    save_metrics(val_metrics, output_dir / f"{dataset}_val_metrics.json")
    save_metrics(test_metrics, output_dir / f"{dataset}_test_metrics.json")

    labels = cfg["labels"] if cfg["labels"] is not None else sorted(pd.unique(df[cfg["label_col"]]).tolist())
    plot_confusion_matrix(
        test_df[cfg["label_col"]],
        test_pred,
        labels=labels,
        title=f"{dataset} baseline confusion matrix",
        path=output_dir / f"{dataset}_test_confusion_matrix.png",
    )

    LOGGER.info("Saved model to %s", model_path)
    LOGGER.info("Validation metrics: %s", val_metrics)
    LOGGER.info("Test metrics: %s", test_metrics)
    return {"val": val_metrics, "test": test_metrics, "model_path": str(model_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simple baseline models.")
    parser.add_argument(
        "--dataset",
        choices=["winning-arguments", "coarse-discourse"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)
    train(args.dataset, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
