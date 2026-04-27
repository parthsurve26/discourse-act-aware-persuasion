"""
Train the BiGRU persuasion model on the Winning Arguments dataset.

Usage:
    uv run python scripts/train_persuasion.py
    uv run python scripts/train_persuasion.py --ablation text_only
    uv run python scripts/train_persuasion.py --hidden_dim 512 --epochs 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from discourse_act_aware_persuasion.bigru_model import BiGRUPersuasionModel
from discourse_act_aware_persuasion.paths import DATA_DIR, MODELS_DIR
from discourse_act_aware_persuasion.persuasion_dataset import build_dataloaders
from discourse_act_aware_persuasion.utils import seed_everything, setup_logging

# ── Paths ────────────────────────────────────────────────────────────────────
CLASSIFIER_CKPT = MODELS_DIR / "discourse_act_classifier" / "model.pt"
PARQUET_PATH    = DATA_DIR / "processed" / "winning_arguments.parquet"
CACHE_ROOT      = DATA_DIR / "processed" / "persuasion_latent_cache"


# ── Load discourse act classifier (frozen) ───────────────────────────────────

def load_predictor(ckpt_path: Path, device: torch.device, ablation: str):
    from transformers import BertTokenizer
    from coarse_discourse_cls.model import BertDiscourseClassifier, DiscoursePredictor

    ckpt = torch.load(ckpt_path, map_location=device)
    model = BertDiscourseClassifier(
        bert_model_name=ckpt["bert_model_name"],
        class_embedding_dim=ckpt["class_embedding_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = BertTokenizer.from_pretrained(ckpt["bert_model_name"])
    predictor = DiscoursePredictor(model, tokenizer, device)

    latent_dim = ckpt["latent_dim"]     # 832 (768 + 64)
    cls_dim    = ckpt["hidden_size"]    # 768

    # For text-only ablation, we monkey-patch encode() to zero out the
    # class_embedding so the GRU only sees the BERT [CLS] representation.
    if ablation == "text_only":
        original_encode = predictor.encode

        def encode_text_only(texts, batch_size=32):
            out = original_encode(texts, batch_size=batch_size)
            # Zero the discourse-act slice (last 64 dims)
            out["latent"][:, cls_dim:] = 0.0
            return out

        predictor.encode = encode_text_only
        print("Ablation: text_only — discourse-act embedding zeroed out.")

    return predictor, latent_dim


# ── Training loop ────────────────────────────────────────────────────────────

def run_epoch(
    model: BiGRUPersuasionModel,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    train: bool,
) -> dict:
    model.train() if train else model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            latents = batch["latents"].to(device)
            lengths = batch["lengths"].to(device)
            labels  = batch["labels"].to(device)

            out    = model(latents, lengths)
            loss   = loss_fn(out["logits"], labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    f1      = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probs)
    return {
        "loss":    total_loss / max(len(loader), 1),
        "f1":      f1,
        "auc_roc": auc_roc,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation",    default="none",
                        choices=["none", "text_only"],
                        help="Ablation variant to run.")
    parser.add_argument("--hidden_dim",  type=int, default=256)
    parser.add_argument("--num_layers",  type=int, default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int, default=15)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen discourse classifier
    predictor, latent_dim = load_predictor(CLASSIFIER_CKPT, device, args.ablation)

    # Build dataloaders (encodes + caches on first run, ~5-10 min)
    cache_suffix = f"_{args.ablation}" if args.ablation != "none" else ""
    loaders = build_dataloaders(
        parquet_path=PARQUET_PATH,
        predictor=predictor,
        cache_root=CACHE_ROOT / f"cache{cache_suffix}",
        batch_size=args.batch_size,
        encode_batch_size=64,
    )

    # Build BiGRU model
    model = BiGRUPersuasionModel(
        input_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    run_name   = f"bigru_{args.ablation}_h{args.hidden_dim}_l{args.num_layers}"
    output_dir = MODELS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    best_val_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, device, train=True)
        val_metrics   = run_epoch(model, loaders["val"],   optimizer, device, train=False)
        scheduler.step()

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                               **{f"val_{k}":   v for k, v in val_metrics.items()}}
        history.append(row)

        print(
            f"Epoch {epoch:>2} | "
            f"train loss {train_metrics['loss']:.4f}  f1 {train_metrics['f1']:.4f}  auc {train_metrics['auc_roc']:.4f} | "
            f"val   loss {val_metrics['loss']:.4f}  f1 {val_metrics['f1']:.4f}  auc {val_metrics['auc_roc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ saved best model (val F1={best_val_f1:.4f})")

    # Test evaluation on best checkpoint
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = run_epoch(model, loaders["test"], optimizer, device, train=False)
    print(f"\nTest | f1 {test_metrics['f1']:.4f}  auc_roc {test_metrics['auc_roc']:.4f}")

    # Save results
    results = {
        "args":    vars(args),
        "history": history,
        "test":    test_metrics,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
