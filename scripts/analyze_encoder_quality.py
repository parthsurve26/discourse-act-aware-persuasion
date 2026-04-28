"""Diagnose the discourse encoder's quality on its own held-out data and on CMV.

Two passes:

  A. In-domain (Cornell Coarse Discourse held-out, last 15% under seed 42).
     Per-class precision/recall/F1, full 10x10 confusion matrix, mean
     softmax confidence on the gold class, silhouette score on the 832-d
     latent grouped by gold label, kNN accuracy on the latent, and a t-SNE
     plot colored by gold label.

  B. Out-of-domain (ChangeMyView test comments).
     Predicted-id class counts, silhouette + kNN on the latent grouped by
     predicted id (internal-consistency reading, not a labeled accuracy),
     and a t-SNE plot colored by predicted id.

A summary rollup is written to data/reports/encoder_quality_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import *  # noqa: F401,F403  (puts repo src on sys.path)

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from transformers import BertTokenizer

# coarse_discourse_cls lives at the repo root as a namespace package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coarse_discourse_cls.model import (  # noqa: E402
    BertDiscourseClassifier,
    DiscoursePredictor,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    NUM_LABELS,
)
from discourse_act_aware_persuasion.paths import REPORTS_DIR  # noqa: E402
from discourse_act_aware_persuasion.winning_arguments.data import (  # noqa: E402
    load_winning_arguments_df,
    split_records,
)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_classifier(repo_id, filename, local_path, device):
    if local_path is not None:
        ckpt_path = local_path
    else:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bert_name = ckpt.get("bert_model_name", "bert-base-uncased")
    class_dim = ckpt.get("class_embedding_dim", 64)

    tokenizer = BertTokenizer.from_pretrained(bert_name)
    classifier = BertDiscourseClassifier(
        bert_model_name=bert_name,
        num_labels=NUM_LABELS,
        class_embedding_dim=class_dim,
    )
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.to(device).eval()
    return classifier, tokenizer


def _load_in_domain_val(data_repo_id, filename, seed=42, train_frac=0.85):
    """Replicate coarse_discourse_cls.model.main()'s shuffle + split exactly."""
    import pandas as pd

    parquet_path = hf_hub_download(
        repo_id=data_repo_id, filename=filename, repo_type="dataset"
    )
    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(df))
    df = df.iloc[shuffled].reset_index(drop=True)
    split_at = int(len(df) * train_frac)
    val_df = df.iloc[split_at:]
    return val_df["text"].tolist(), val_df["label"].tolist()


def _load_cmv_test_comments(parquet_path, max_comments):
    df = load_winning_arguments_df(parquet_path)
    splits = split_records(df, max_comments=max_comments)
    test_records = splits["test"]
    flat = []
    for r in test_records:
        for t in r.comment_texts:
            t = t.strip()
            if t:
                flat.append(t)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _per_class_prf(cm):
    """Precision / recall / F1 from a confusion matrix (rows = gold, cols = pred)."""
    out = []
    for c in range(cm.shape[0]):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        support = int(cm[c, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({
            "label": LABELS[c],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        })
    macro_f1 = float(np.mean([r["f1"] for r in out])) if out else 0.0
    micro_acc = float(np.diag(cm).sum() / cm.sum()) if cm.sum() else 0.0
    return out, macro_f1, micro_acc


def _knn_accuracy(latent, labels, *, k=5, n_splits=5, sample_size=5000, seed=42):
    """Mean kNN accuracy across n_splits random 80/20 splits."""
    n = len(latent)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
    X = latent[idx]
    y = np.asarray(labels)[idx]
    if len(np.unique(y)) < 2:
        return {"mean": 0.0, "std": 0.0, "n_samples": int(len(y)), "note": "fewer than 2 classes"}
    accs = []
    for fold in range(n_splits):
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=int(seed + fold), stratify=y
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=int(seed + fold)
            )
        clf = KNeighborsClassifier(n_neighbors=min(k, len(X_tr)))
        clf.fit(X_tr, y_tr)
        accs.append(float(clf.score(X_te, y_te)))
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "n_samples": int(len(y)),
        "k": k,
        "n_splits": n_splits,
    }


def _silhouette(latent, labels, *, sample_size=5000, seed=42):
    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        return {"score": 0.0, "n_samples": int(len(y)), "note": "fewer than 2 classes"}
    s = float(silhouette_score(
        latent, y,
        sample_size=min(sample_size, len(y)),
        random_state=seed,
    ))
    return {"score": s, "n_samples": int(min(sample_size, len(y)))}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _save_confusion(cm, out_path, title="Confusion matrix (rows = gold, cols = pred)"):
    fig, ax = plt.subplots(figsize=(9, 8))
    norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(NUM_LABELS))
    ax.set_yticks(range(NUM_LABELS))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    for i in range(NUM_LABELS):
        for j in range(NUM_LABELS):
            v = cm[i, j]
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_tsne(latent, labels, out_path, title, *, sample=3000, seed=42):
    n = len(latent)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    X = latent[idx]
    y = np.asarray(labels)[idx]

    tsne = TSNE(
        n_components=2, perplexity=min(30, max(5, len(X) // 20)),
        init="pca", random_state=seed, learning_rate="auto",
    )
    X2 = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    classes = sorted(np.unique(y).tolist())
    for i, c in enumerate(classes):
        m = y == c
        label_str = LABELS[c] if 0 <= c < NUM_LABELS else str(c)
        ax.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.6,
                   color=cmap(i % 10), label=label_str)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Eval passes
# ─────────────────────────────────────────────────────────────────────────────

def _encode(predictor, texts, batch_size, label="texts"):
    print(f"[encode] {label}: {len(texts)} examples (batch={batch_size})")
    return predictor.encode(texts, batch_size=batch_size)


def _eval_in_domain(predictor, texts, gold_labels, args, reports_dir):
    print("[A] in-domain encode...")
    enc = _encode(predictor, texts, args.batch_size, label="cornell-val")
    latent = enc["latent"]
    pred_ids = enc["predicted_ids"]
    probs = enc["probabilities"]
    gold_ids = np.array([LABEL2ID[g] for g in gold_labels], dtype=np.int64)

    # Confusion + per-class P/R/F1
    cm = confusion_matrix(gold_ids, pred_ids, labels=list(range(NUM_LABELS)))
    per_class, macro_f1, micro_acc = _per_class_prf(cm)

    # Mean confidence on the gold class
    conf_on_gold = probs[np.arange(len(gold_ids)), gold_ids]
    mean_conf_per_class = []
    for c in range(NUM_LABELS):
        mask = gold_ids == c
        mean_conf_per_class.append({
            "label": LABELS[c],
            "mean_conf_on_gold": float(conf_on_gold[mask].mean()) if mask.any() else 0.0,
            "support": int(mask.sum()),
        })

    # Embedding separability
    sil = _silhouette(latent, gold_ids,
                      sample_size=args.cluster_sample, seed=args.seed)
    knn = _knn_accuracy(latent, gold_ids,
                        sample_size=args.cluster_sample, seed=args.seed)

    # Plots
    _save_confusion(cm, reports_dir / "encoder_quality_in_domain_confusion.png",
                    title="In-domain confusion (Cornell held-out)")
    _save_tsne(latent, gold_ids,
               reports_dir / "encoder_quality_in_domain_tsne.png",
               title="In-domain t-SNE (832-d latent, colored by gold label)",
               sample=args.tsne_sample, seed=args.seed)

    summary = {
        "domain": "cornell_coarse_discourse_held_out",
        "n_examples": int(len(texts)),
        "macro_f1": macro_f1,
        "micro_accuracy": micro_acc,
        "per_class": per_class,
        "mean_confidence_on_gold_per_class": mean_conf_per_class,
        "confusion_matrix": cm.tolist(),
        "labels": list(LABELS),
        "silhouette": sil,
        "knn_accuracy": knn,
    }
    out_path = reports_dir / "encoder_quality_in_domain.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[A] wrote {out_path}")
    print(f"[A] macro_f1={macro_f1:.4f}  micro_acc={micro_acc:.4f}  "
          f"silhouette={sil['score']:.4f}  knn={knn['mean']:.4f}±{knn['std']:.4f}")
    return summary


def _eval_ood(predictor, texts, args, reports_dir):
    print("[B] out-of-domain encode...")
    enc = _encode(predictor, texts, args.batch_size, label="cmv-test")
    latent = enc["latent"]
    pred_ids = enc["predicted_ids"]

    # Class counts
    counts = {LABELS[c]: int((pred_ids == c).sum()) for c in range(NUM_LABELS)}

    sil = _silhouette(latent, pred_ids,
                      sample_size=args.cluster_sample, seed=args.seed)
    knn = _knn_accuracy(latent, pred_ids,
                        sample_size=args.cluster_sample, seed=args.seed)

    _save_tsne(latent, pred_ids,
               reports_dir / "encoder_quality_cmv_tsne.png",
               title="CMV t-SNE (832-d latent, colored by predicted id)",
               sample=args.tsne_sample, seed=args.seed)

    summary = {
        "domain": "cmv_test_comments",
        "n_examples": int(len(texts)),
        "predicted_id_counts": counts,
        "labels": list(LABELS),
        "silhouette": sil,
        "knn_accuracy": knn,
        "note": (
            "kNN target is the encoder's own predicted_id (no gold labels on CMV) — "
            "high accuracy means the latent is internally consistent with the argmax; "
            "low accuracy means the argmax is unstable in the latent geometry."
        ),
    }
    out_path = reports_dir / "encoder_quality_cmv.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[B] wrote {out_path}")
    print(f"[B] n_examples={summary['n_examples']}  "
          f"silhouette={sil['score']:.4f}  knn={knn['mean']:.4f}±{knn['std']:.4f}")
    return summary


def _dump_summary(in_domain, ood, reports_dir):
    summary = {
        "in_domain": {
            "domain": in_domain["domain"],
            "n_examples": in_domain["n_examples"],
            "macro_f1": in_domain["macro_f1"],
            "micro_accuracy": in_domain["micro_accuracy"],
            "silhouette": in_domain["silhouette"],
            "knn_accuracy": in_domain["knn_accuracy"],
        },
        "out_of_domain": {
            "domain": ood["domain"],
            "n_examples": ood["n_examples"],
            "predicted_id_counts": ood["predicted_id_counts"],
            "silhouette": ood["silhouette"],
            "knn_accuracy": ood["knn_accuracy"],
        },
        "interpretation_matrix": {
            "in_domain_healthy + ood_healthy":
                "encoder is fine; downstream issues are in the persuasion transformer",
            "in_domain_healthy + ood_weak":
                "domain shift; predicted ids on CMV are noisy",
            "in_domain_weak + ood_weak":
                "classifier itself is the bottleneck",
            "in_domain_weak + ood_healthy":
                "anomalous; investigate",
        },
    }
    out_path = reports_dir / "encoder_quality_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--discourse-repo-id", default="Vijayrathank/discourse_act_classifier")
    p.add_argument("--discourse-filename", default="model.pt")
    p.add_argument("--discourse-local-path", default=None)
    p.add_argument("--discourse-data-repo-id", default="Vijayrathank/reddit_discourse_cleaned")
    p.add_argument("--discourse-data-filename", default="data/train-00000-of-00001.parquet")
    p.add_argument("--parquet-path", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-in-domain", type=int, default=0,
                   help="0 = use full Cornell val split.")
    p.add_argument("--max-ood", type=int, default=30000,
                   help="Cap on number of CMV comments to encode.")
    p.add_argument("--cluster-sample", type=int, default=5000)
    p.add_argument("--tsne-sample", type=int, default=3000)
    p.add_argument("--max-comments", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reports-dir", default=str(REPORTS_DIR))
    return p.parse_args()


def _subsample(items, max_n, seed, *paired):
    if max_n <= 0 or len(items) <= max_n:
        return (items, *paired) if paired else items
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=max_n, replace=False)
    sub = [items[i] for i in idx]
    if not paired:
        return sub
    out = [sub]
    for arr in paired:
        out.append([arr[i] for i in idx])
    return tuple(out)


def main():
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("[load] discourse classifier...")
    classifier, tokenizer = _load_classifier(
        args.discourse_repo_id, args.discourse_filename,
        args.discourse_local_path, device,
    )
    predictor = DiscoursePredictor(
        classifier, tokenizer, device, max_length=args.max_tokens
    )

    # ── A. In-domain ────────────────────────────────────────────
    print("[load] Cornell held-out val split...")
    val_texts, val_labels = _load_in_domain_val(
        args.discourse_data_repo_id, args.discourse_data_filename,
        seed=42, train_frac=0.85,
    )
    if args.max_in_domain > 0:
        val_texts, val_labels = _subsample(
            val_texts, args.max_in_domain, args.seed, val_labels
        )
    print(f"[load] cornell val: {len(val_texts)} examples")
    in_domain = _eval_in_domain(predictor, val_texts, val_labels, args, reports_dir)

    # ── B. Out-of-domain ────────────────────────────────────────
    print("[load] CMV test comments...")
    cmv_texts = _load_cmv_test_comments(args.parquet_path, args.max_comments)
    if args.max_ood > 0:
        cmv_texts = _subsample(cmv_texts, args.max_ood, args.seed)
    print(f"[load] cmv: {len(cmv_texts)} comments")
    ood = _eval_ood(predictor, cmv_texts, args, reports_dir)

    # ── C. Summary rollup ───────────────────────────────────────
    _dump_summary(in_domain, ood, reports_dir)


if __name__ == "__main__":
    main()
