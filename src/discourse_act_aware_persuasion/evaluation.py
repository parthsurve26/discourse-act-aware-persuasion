from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import ensure_parent, write_json


def classification_metrics(y_true, y_pred, labels=None) -> Dict[str, Any]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
    }
    return metrics


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    write_json(path, metrics)


def plot_confusion_matrix(y_true, y_pred, labels, title: str, path: Path) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    ensure_parent(path)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

