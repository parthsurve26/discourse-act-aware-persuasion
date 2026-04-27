"""
Train persuasion models on the Winning Arguments dataset.

Models:
    bigru       — Bidirectional GRU (Thameem's model 1)
    transformer — Transition-Aware Transformer (Thameem's model 2)

Usage:
    uv run python scripts/train_persuasion.py --model bigru
    uv run python scripts/train_persuasion.py --model transformer
    uv run python scripts/train_persuasion.py --model bigru --ablation text_only
    uv run python scripts/train_persuasion.py --model transformer --ablation text_only
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
from discourse_act_aware_persuasion.transformer_model import TransitionAwareTransformer
from discourse_act_aware_persuasion.paths import DATA_DIR, MODELS_DIR
from discourse_act_aware_persuasion.persuasion_dataset import build_dataloaders
from discourse_act_aware_persuasion.utils import seed_everything, setup_logging

# ── Paths ─────────────────────────────────────────────────────────────────────
CLASSIFIER_CKPT = MODELS_DIR / "discourse_act_classifier" / "model.pt"
PARQUET_PATH    = DATA_DIR / "processed" / "winning_arguments.parquet"
CACHE_ROOT      = DATA_DIR / "processed" / "persuasion_latent_cache"


# ── Load discourse act classifier (frozen) ────────────────────────────────────

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

    tokenizer  = BertTokenizer.from_pretrained(ckpt["bert_model_name"])
    predictor  = DiscoursePredictor(model, tokenizer, device)
    latent_dim = ckpt["latent_dim"]   # 832
    cls_dim    = ckpt["hidden_size"]  # 768

    if ablation == "text_only":
        original_encode = predictor.encode

        def encode_text_only(texts, batch_size=32):
            out = original_encode(texts, batch_size=batch_size)
            out["latent"][:, cls_dim:] = 0.0  # zero the 64-dim discourse slice
            return out

        predictor.encode = encode_text_only
        print("Ablation: text_only — discourse-act embedding zeroed out.")

    return predictor, latent_dim


# ── Build model ───────────────────────────────────────────────────────────────

def build_model(args, latent_dim: int) -> nn.Module:
    if args.model == "bigru":
        return BiGRUPersuasionModel(
            input_dim=latent_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:  # transformer
        return TransitionAwareTransformer(
            input_dim=latent_dim,
            d_model=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.hidden_dim * 2,
            dropout=args.dropout,
        )


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    model:    nn.Module,
    loader:   DataLoader,
    optimizer,
    device:   torch.device,
    train:    bool,
    is_transformer: bool = False,
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
            act_ids = batch["act_ids"].to(device)

            # Transformer needs act_ids; BiGRU ignores them
            if is_transformer:
                out = model(latents, act_ids, lengths)
            else:
                out = model(latents, lengths)

            loss = loss_fn(out["logits"], labels)

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

    return {
        "loss":    total_loss / max(len(loader), 1),
        "f1":      f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="bigru",
                        choices=["bigru", "transformer"],
                        help="Which model to train.")
    parser.add_argument("--ablation",    default="none",
                        choices=["none", "text_only"],
                        help="Ablation variant.")
    # Shared hyperparameters
    parser.add_argument("--hidden_dim",  type=int,   default=128)
    parser.add_argument("--num_layers",  type=int,   default=1)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--seed",        type=int,   default=42)
    # Transformer-specific
    parser.add_argument("--num_heads",   type=int,   default=4,
                        help="Number of attention heads (transformer only).")
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}  |  Ablation: {args.ablation}")

    predictor, latent_dim = load_predictor(CLASSIFIER_CKPT, device, args.ablation)

    cache_suffix = f"_{args.ablation}" if args.ablation != "none" else ""
    loaders = build_dataloaders(
        parquet_path=PARQUET_PATH,
        predictor=predictor,
        cache_root=CACHE_ROOT / f"cache{cache_suffix}",
        batch_size=args.batch_size,
        encode_batch_size=64,
    )

    model = build_model(args, latent_dim).to(device)
    is_transformer = (args.model == "transformer")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name   = f"{args.model}_{args.ablation}_h{args.hidden_dim}_l{args.num_layers}"
    output_dir = MODELS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1       = 0.0
    epochs_no_improve = 0
    history           = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, loaders["train"], optimizer, device, True,  is_transformer)
        val_m   = run_epoch(model, loaders["val"],   optimizer, device, False, is_transformer)
        scheduler.step()

        row = {"epoch": epoch,
               **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}":   v for k, v in val_m.items()}}
        history.append(row)

        print(
            f"Epoch {epoch:>2} | "
            f"train loss {train_m['loss']:.4f}  f1 {train_m['f1']:.4f}  auc {train_m['auc_roc']:.4f} | "
            f"val   loss {val_m['loss']:.4f}  f1 {val_m['f1']:.4f}  auc {val_m['auc_roc']:.4f}"
        )

        if val_m["f1"] > best_val_f1:
            best_val_f1       = val_m["f1"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ saved best model (val F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no val F1 improvement for {args.patience} epochs)")
                break

    # ── Test on best checkpoint ───────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_m = run_epoch(model, loaders["test"], optimizer, device, False, is_transformer)
    print(f"\nTest | f1 {test_m['f1']:.4f}  auc_roc {test_m['auc_roc']:.4f}")

    # ── Save transition bias matrix (transformer only) ────────────────────────
    if is_transformer and args.ablation == "none":
        bias_info = model.get_transition_bias()
        bias_dict = {
            "labels": bias_info["labels"],
            "matrix": bias_info["matrix"].tolist(),
        }
        with open(output_dir / "transition_bias.json", "w") as f:
            json.dump(bias_dict, f, indent=2)
        print("Transition bias matrix saved to transition_bias.json")

    results = {"args": vars(args), "history": history, "test": test_m}
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
