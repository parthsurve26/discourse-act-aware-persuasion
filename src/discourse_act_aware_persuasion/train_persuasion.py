from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import save_metrics
from .neural import (
    CommentAttentionClassifier,
    TransformerThreadClassifier,
    build_fresh_encoder_bundle,
    evaluate_model,
    load_discourse_encoder_bundle,
    load_winning_arguments_splits,
    make_comment_dataloader,
    make_text_dataloader,
    save_checkpoint,
    train_model,
)
from .paths import MODELS_DIR, PROCESSED_DIR
from .utils import seed_everything, setup_logging

DEFAULT_DISCOURSE_CHECKPOINT = MODELS_DIR / "external" / "model.pt"


def train_persuasion_model(
    architecture: str,
    winning_arguments_path: Path,
    output_dir: Path,
    base_model_name: str,
    discourse_checkpoint: str | None,
    batch_size: int,
    max_length: int,
    max_comments: int,
    max_comment_length: int,
    epochs: int,
    lr: float,
    use_ack_masked: bool,
    local_files_only: bool,
    train_limit: int | None,
    val_limit: int | None,
    test_limit: int | None,
) -> dict:
    splits = load_winning_arguments_splits(winning_arguments_path)
    if train_limit is not None:
        splits["train"] = splits["train"].head(train_limit).reset_index(drop=True)
    if val_limit is not None:
        splits["val"] = splits["val"].head(val_limit).reset_index(drop=True)
    if test_limit is not None:
        splits["test"] = splits["test"].head(test_limit).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if architecture == "fresh-bert":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=local_files_only)
        text_col = "input_text_ack_masked" if use_ack_masked and "input_text_ack_masked" in splits["train"].columns else "input_text"
        train_loader = make_text_dataloader(splits["train"], tokenizer, text_col, max_length, batch_size, shuffle=True)
        val_loader = make_text_dataloader(splits["val"], tokenizer, text_col, max_length, batch_size, shuffle=False)
        test_loader = make_text_dataloader(splits["test"], tokenizer, text_col, max_length, batch_size, shuffle=False)
        model = TransformerThreadClassifier(base_model_name=base_model_name, local_files_only=local_files_only)
    elif architecture in {"fresh-attn", "fresh-bilstm-attn", "discourse-attn", "discourse-bilstm-attn"}:
        if architecture in {"fresh-attn", "fresh-bilstm-attn"}:
            encoder_bundle = build_fresh_encoder_bundle(base_model_name, local_files_only=local_files_only)
            use_attention_bias = False
            use_latent_features = False
        else:
            if discourse_checkpoint is None:
                raise ValueError("`discourse-bilstm-attn` requires --discourse-checkpoint.")
            encoder_bundle = load_discourse_encoder_bundle(
                checkpoint_path=discourse_checkpoint,
                base_model_name=base_model_name,
                local_files_only=local_files_only,
            )
            use_attention_bias = True
            use_latent_features = True

        train_loader = make_comment_dataloader(
            splits["train"],
            encoder_bundle.tokenizer,
            max_comments=max_comments,
            max_comment_length=max_comment_length,
            batch_size=batch_size,
            shuffle=True,
            use_ack_masked=use_ack_masked,
        )
        val_loader = make_comment_dataloader(
            splits["val"],
            encoder_bundle.tokenizer,
            max_comments=max_comments,
            max_comment_length=max_comment_length,
            batch_size=batch_size,
            shuffle=False,
            use_ack_masked=use_ack_masked,
        )
        test_loader = make_comment_dataloader(
            splits["test"],
            encoder_bundle.tokenizer,
            max_comments=max_comments,
            max_comment_length=max_comment_length,
            batch_size=batch_size,
            shuffle=False,
            use_ack_masked=use_ack_masked,
        )
        model = CommentAttentionClassifier(
            encoder_bundle=encoder_bundle,
            use_bilstm=architecture in {"fresh-bilstm-attn", "discourse-bilstm-attn"},
            use_attention_bias=use_attention_bias,
            use_latent_features=use_latent_features,
            freeze_encoder=architecture in {"discourse-attn", "discourse-bilstm-attn"},
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
    )
    test_metrics = evaluate_model(model, test_loader)

    checkpoint_path = output_dir / f"{architecture}.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "base_model_name": base_model_name,
            "history": history,
            "test_metrics": test_metrics,
        },
    )
    metrics_payload = {
        "architecture": architecture,
        "history": history,
        "test": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    save_metrics(metrics_payload, output_dir / f"{architecture}_metrics.json")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train persuasion models on Winning Arguments.")
    parser.add_argument(
        "--architecture",
        choices=[
            "fresh-bert",
            "fresh-attn",
            "fresh-bilstm-attn",
            "discourse-attn",
            "discourse-bilstm-attn",
        ],
        required=True,
    )
    parser.add_argument(
        "--winning-arguments-path",
        type=Path,
        default=PROCESSED_DIR / "winning_arguments.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "persuasion")
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument("--discourse-checkpoint", default=str(DEFAULT_DISCOURSE_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-comments", type=int, default=16)
    parser.add_argument("--max-comment-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--use-ack-masked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use delta-acknowledgement-masked Winning Arguments text/comments when available.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load base Transformer/tokenizer files from the local Hugging Face cache.",
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Optional limit for train rows.")
    parser.add_argument("--val-limit", type=int, default=None, help="Optional limit for validation rows.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional limit for test rows.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)
    metrics = train_persuasion_model(
        architecture=args.architecture,
        winning_arguments_path=args.winning_arguments_path,
        output_dir=args.output_dir,
        base_model_name=args.base_model,
        discourse_checkpoint=args.discourse_checkpoint,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_comments=args.max_comments,
        max_comment_length=args.max_comment_length,
        epochs=args.epochs,
        lr=args.lr,
        use_ack_masked=args.use_ack_masked,
        local_files_only=args.local_files_only,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )
    print(metrics["test"])


if __name__ == "__main__":
    main()
