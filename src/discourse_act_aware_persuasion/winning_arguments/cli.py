"""Shared CLI runner for both winning-arguments transformer variants."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..paths import MODELS_DIR
from .data import (
    WinningArgumentsDataset,
    collate_threads,
    load_winning_arguments_df,
    split_records,
)
from .encoders import FrozenBertCommentEncoder, FrozenDiscourseEncoder
from .infer import WinningArgumentsPredictor
from .model import TransitionAwareThreadTransformer
from .trainer import ThreadTrainer


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_encoder(variant: str, args):
    if variant == "standalone":
        return FrozenBertCommentEncoder(
            bert_model_name=args.bert_model_name,
            max_length=args.max_tokens,
        )
    if variant == "with_discourse":
        return FrozenDiscourseEncoder(
            hf_repo_id=args.discourse_repo_id,
            checkpoint_filename=args.discourse_filename,
            local_checkpoint=args.discourse_local_path,
            max_length=args.max_tokens,
        )
    raise ValueError(f"Unknown variant: {variant}")


def _build_model(variant: str, args, encoder) -> TransitionAwareThreadTransformer:
    comment_dim = encoder.OUTPUT_DIM if variant == "standalone" else encoder.output_dim
    return TransitionAwareThreadTransformer(
        comment_dim=comment_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_comments=args.max_comments,
        use_discourse_acts=(variant == "with_discourse"),
        use_speaker_bias=not args.no_speaker_bias,
        use_distance_bias=not args.no_distance_bias,
    )


def _make_loader(records, batch_size: int, shuffle: bool) -> DataLoader:
    ds = WinningArgumentsDataset(records)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_threads)


def _checkpoint_path(variant: str, args) -> Path:
    if args.ckpt_path:
        return Path(args.ckpt_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"winning_args_{variant}.pt"


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", choices=["train", "eval", "predict", "sweep"], default="train")
    p.add_argument("--parquet-path", default=None)
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Cap records per split (0 = use all). Smoke-test knob.")

    # Data
    p.add_argument("--max-comments", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--early-stop-patience", type=int, default=3)

    # Encoder-specific
    p.add_argument("--bert-model-name", default="bert-base-uncased")
    p.add_argument("--discourse-repo-id", default="Vijayrathank/discourse_act_classifier")
    p.add_argument("--discourse-filename", default="model.pt")
    p.add_argument("--discourse-local-path", default=None)

    # Bias ablation flags (proposal ablation: model without transition-aware attention)
    p.add_argument("--no-speaker-bias", action="store_true",
                   help="Disable the (speaker_i, speaker_j) attention bias table.")
    p.add_argument("--no-distance-bias", action="store_true",
                   help="Disable the bucketed turn-distance attention bias table.")

    # Predict mode
    p.add_argument("--thread-json", default=None,
                   help="Path to a JSON file with one thread dict or a list of them.")


def build_parser(variant: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Winning Arguments transformer ({variant})")
    _add_common_args(p)
    return p


def run(variant: str, argv: Optional[list] = None) -> None:
    args = build_parser(variant).parse_args(argv)
    _seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{variant}] device={device} mode={args.mode}")

    encoder = _build_encoder(variant, args)
    model = _build_model(variant, args, encoder)
    ckpt_path = _checkpoint_path(variant, args)

    if args.mode == "predict":
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        predictor = WinningArgumentsPredictor(encoder, model, device)

        if not args.thread_json:
            raise ValueError("--thread-json is required for predict mode")
        with open(args.thread_json) as f:
            payload = json.load(f)
        threads = payload if isinstance(payload, list) else [payload]
        for r in predictor.predict(threads):
            print(json.dumps(r))
        return

    df = load_winning_arguments_df(args.parquet_path)
    splits = split_records(df, max_comments=args.max_comments)
    if args.limit > 0:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    print({k: len(v) for k, v in splits.items()})

    if args.mode == "sweep":
        from .sweep import run_sweep
        run_sweep(
            variant=variant,
            args=args,
            encoder=encoder,
            device=device,
            splits=splits,
            make_loader=_make_loader,
            ckpt_path=ckpt_path,
        )
        return

    if args.mode == "eval":
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        trainer = ThreadTrainer(encoder=encoder, model=model, device=device)
        test_loader = _make_loader(splits["test"], args.batch_size, shuffle=False)
        metrics = trainer.evaluate(test_loader)
        print("test metrics:", json.dumps(metrics, indent=2))
        return

    # mode == "train"
    train_loader = _make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader = _make_loader(splits["val"], args.batch_size, shuffle=False)
    test_loader = _make_loader(splits["test"], args.batch_size, shuffle=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_frac),
        num_training_steps=total_steps,
    )
    trainer = ThreadTrainer(encoder=encoder, model=model, device=device,
                            optimizer=optimizer, scheduler=scheduler)
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        early_stop_metric="auc",
        early_stop_patience=args.early_stop_patience,
    )

    test_metrics = trainer.evaluate(test_loader)
    print("test metrics:", json.dumps(test_metrics, indent=2))

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "test_metrics": test_metrics,
        },
        ckpt_path,
    )
    print(f"saved checkpoint to {ckpt_path}")
