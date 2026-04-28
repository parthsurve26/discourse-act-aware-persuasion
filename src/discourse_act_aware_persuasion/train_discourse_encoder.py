from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from .data.huggingface_discourse import (
    DEFAULT_HF_DATASET_ID,
    DEFAULT_HF_FILENAME,
    ensure_discourse_splits,
    load_huggingface_discourse_dataframe,
)
from .evaluation import save_metrics
from .neural import (
    DiscourseEncoderModel,
    TextClassificationDataset,
    build_label_vocab,
    evaluate_model,
    save_checkpoint,
    train_model,
)
from .paths import MODELS_DIR, REPORTS_DIR
from .utils import seed_everything, setup_logging, write_json


def _make_loader(df: pd.DataFrame, tokenizer, label_to_id, max_length: int, batch_size: int, shuffle: bool):
    dataset = TextClassificationDataset(
        texts=df["text"].tolist(),
        labels=[label_to_id[str(label)] for label in df["label"].tolist()],
        tokenizer=tokenizer,
        max_length=max_length,
    )
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_discourse_encoder(
    output_dir: Path,
    base_model_name: str,
    dataset_id: str,
    dataset_filename: str,
    local_path: str | None,
    epochs: int,
    batch_size: int,
    max_length: int,
    lr: float,
    seed: int,
) -> dict:
    df = load_huggingface_discourse_dataframe(
        dataset_id=dataset_id,
        filename=dataset_filename,
        local_path=local_path,
    )
    df = ensure_discourse_splits(df, seed=seed)
    label_to_id, id_to_label = build_label_vocab(df["label"].tolist())

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_loader = _make_loader(
        df[df["split"] == "train"],
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _make_loader(
        df[df["split"] == "val"],
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = _make_loader(
        df[df["split"] == "test"],
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=False,
    )

    model = DiscourseEncoderModel(
        base_model_name=base_model_name,
        num_labels=len(label_to_id),
    )
    model, history, selection = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
    )
    test_metrics = evaluate_model(model, test_loader)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "discourse_encoder.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "class_embedding_dim": model.class_embedding_dim,
            "base_model_name": base_model_name,
            "tokenizer_name": base_model_name,
            "max_length": max_length,
            "dataset_id": dataset_id,
            "dataset_filename": dataset_filename,
            "local_path": local_path,
        },
    )

    metrics_payload = {
        "history": history,
        "selection": selection,
        "test": test_metrics,
        "label_distribution": df["label"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "checkpoint_path": str(checkpoint_path),
    }
    save_metrics(metrics_payload, output_dir / "discourse_encoder_metrics.json")
    write_json(REPORTS_DIR / "hf_discourse_summary.json", metrics_payload)
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the discourse encoder on Hugging Face coarse discourse data.")
    parser.add_argument("--dataset-id", default=DEFAULT_HF_DATASET_ID)
    parser.add_argument("--dataset-filename", default=DEFAULT_HF_FILENAME)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "discourse_encoder")
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)
    metrics = train_discourse_encoder(
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        dataset_id=args.dataset_id,
        dataset_filename=args.dataset_filename,
        local_path=args.local_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        seed=args.seed,
    )
    print(metrics["test"])


if __name__ == "__main__":
    main()
