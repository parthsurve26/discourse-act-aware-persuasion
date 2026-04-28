"""Analyze the learned act_pair_bias and compare against empirical transition frequencies.

Produces, in data/reports/:

  - act_pair_bias.json                      raw 11x11 learned bias table
  - act_pair_bias_heatmap.png               labeled heatmap of the bias
  - act_transition_freq.json                empirical (winning, losing) counts/freqs/delta
  - act_transition_delta_heatmap.png        winning - losing frequency
  - act_transition_log_odds_heatmap.png     log-odds heatmap
  - act_transition_top_k.json               top-K winning/losing transitions
  - act_transition_alignment.json           Spearman(learned, empirical) correlation

Default checkpoint: Vijayrathank/persuasion_model_with_discoarse_cls.
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
from torch.utils.data import DataLoader

# coarse_discourse_cls lives at the repo root as a namespace package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from coarse_discourse_cls.model import LABELS as ACT_LABELS  # noqa: E402
from discourse_act_aware_persuasion.paths import REPORTS_DIR  # noqa: E402
from discourse_act_aware_persuasion.winning_arguments.data import (  # noqa: E402
    WinningArgumentsDataset,
    collate_threads,
    load_winning_arguments_df,
    split_records,
)
from discourse_act_aware_persuasion.winning_arguments.encoders import (  # noqa: E402
    FrozenDiscourseEncoder,
)


NUM_ACTS = 10
AXIS_WITH_CLS = list(ACT_LABELS) + ["[CLS]"]


def _load_act_pair_bias(repo_id: str, filename: str, local_path):
    """Pull the persuasion checkpoint and extract the (num_acts+1)^2 bias table."""
    if local_path is not None:
        ckpt_path = local_path
    else:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    if "act_pair_bias.weight" not in state:
        raise KeyError(
            "Checkpoint missing act_pair_bias.weight — was this trained with "
            "use_discourse_acts=True?"
        )
    weight = state["act_pair_bias.weight"].detach().cpu().numpy()
    n = NUM_ACTS + 1
    table = weight.reshape(n, n)
    return table, ckpt


def _save_heatmap(matrix, labels, title, out_path, *, symmetric=True, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(9, 8))
    if symmetric:
        m = max(abs(matrix.min()), abs(matrix.max())) or 1.0
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-m, vmax=m)
        thresh = m / 2.0
    else:
        im = ax.imshow(matrix, cmap="viridis")
        thresh = (float(matrix.min()) + float(matrix.max())) / 2.0

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("act_j (attended-to)")
    ax.set_ylabel("act_i (attending)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if abs(v) > thresh else "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _empirical_transitions(records, encoder, device, batch_size):
    """Run the discourse encoder over test threads and tally adjacent (act_t, act_{t+1}) pairs."""
    counts = {
        0: np.zeros((NUM_ACTS, NUM_ACTS), dtype=np.int64),
        1: np.zeros((NUM_ACTS, NUM_ACTS), dtype=np.int64),
    }
    ds = WinningArgumentsDataset(records)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_threads)
    n_threads = 0
    for batch in loader:
        comment_mask = batch["comment_mask"].to(device)
        labels = batch["labels"].numpy()
        with torch.no_grad():
            _, acts = encoder(batch["comment_texts"], comment_mask)
        acts = acts.cpu().numpy()
        mask = comment_mask.cpu().numpy()
        for b in range(acts.shape[0]):
            n = int(mask[b].sum())
            n_threads += 1
            if n < 2:
                continue
            seq = acts[b, :n]
            label = int(labels[b])
            for t in range(n - 1):
                a_i, a_j = int(seq[t]), int(seq[t + 1])
                if 0 <= a_i < NUM_ACTS and 0 <= a_j < NUM_ACTS:
                    counts[label][a_i, a_j] += 1
        print(f"  processed {n_threads} threads", end="\r")
    print()
    return counts[1], counts[0]


def _rankdata(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(arr)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x, y):
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else 0.0


def _top_k(matrix, k, labels):
    flat = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            flat.append({
                "act_i": labels[i],
                "act_j": labels[j],
                "value": float(matrix[i, j]),
            })
    flat.sort(key=lambda r: r["value"], reverse=True)
    return {
        "top_winning": flat[:k],
        "top_losing": flat[-k:][::-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persuasion-repo-id",
                        default="Vijayrathank/persuasion_model_with_discoarse_cls")
    parser.add_argument("--persuasion-filename", default="model.pt")
    parser.add_argument("--persuasion-local-path", default=None)
    parser.add_argument("--discourse-repo-id",
                        default="Vijayrathank/discourse_act_classifier")
    parser.add_argument("--discourse-filename", default="model.pt")
    parser.add_argument("--discourse-local-path", default=None)
    parser.add_argument("--parquet-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-comments", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    # ── A. learned act_pair_bias ─────────────────────────────────────────
    print("[A] loading persuasion checkpoint...")
    bias_table, ckpt = _load_act_pair_bias(
        args.persuasion_repo_id, args.persuasion_filename, args.persuasion_local_path
    )
    bias_path = reports_dir / "act_pair_bias.json"
    with bias_path.open("w") as f:
        json.dump({
            "labels": AXIS_WITH_CLS,
            "bias_table": bias_table.tolist(),
            "shape": list(bias_table.shape),
            "checkpoint_args": ckpt.get("args"),
        }, f, indent=2, default=str)
    print(f"[A] wrote {bias_path}")

    _save_heatmap(
        bias_table, AXIS_WITH_CLS,
        title="Learned act_pair_bias (transition-aware attention)",
        out_path=reports_dir / "act_pair_bias_heatmap.png",
    )
    print("[A] wrote act_pair_bias_heatmap.png")

    # ── B. empirical transitions on test split ───────────────────────────
    print("[B] loading test records...")
    df = load_winning_arguments_df(args.parquet_path)
    splits = split_records(df, max_comments=args.max_comments)
    test_records = splits["test"]
    print(f"[B] test records = {len(test_records)}")

    print("[B] building discourse encoder...")
    encoder = FrozenDiscourseEncoder(
        hf_repo_id=args.discourse_repo_id,
        checkpoint_filename=args.discourse_filename,
        local_checkpoint=args.discourse_local_path,
        max_length=args.max_tokens,
    ).to(device)

    print("[B] running encoder on test threads...")
    count_winning, count_losing = _empirical_transitions(
        test_records, encoder, device, args.batch_size
    )

    eps = 1e-9
    total_w = count_winning.sum() + eps
    total_l = count_losing.sum() + eps
    freq_winning = count_winning / total_w
    freq_losing = count_losing / total_l
    delta = freq_winning - freq_losing
    log_odds = np.log((freq_winning + eps) / (freq_losing + eps))

    freq_path = reports_dir / "act_transition_freq.json"
    with freq_path.open("w") as f:
        json.dump({
            "labels": list(ACT_LABELS),
            "count_winning": count_winning.tolist(),
            "count_losing": count_losing.tolist(),
            "freq_winning": freq_winning.tolist(),
            "freq_losing": freq_losing.tolist(),
            "delta": delta.tolist(),
            "log_odds": log_odds.tolist(),
            "total_transitions_winning": int(count_winning.sum()),
            "total_transitions_losing": int(count_losing.sum()),
        }, f, indent=2)
    print(f"[B] wrote {freq_path}")

    _save_heatmap(
        delta, list(ACT_LABELS),
        title="Empirical transition frequency: winning - losing",
        out_path=reports_dir / "act_transition_delta_heatmap.png",
        fmt="{:+.3f}",
    )
    _save_heatmap(
        log_odds, list(ACT_LABELS),
        title="Log-odds of (act_i -> act_j) given winning vs losing",
        out_path=reports_dir / "act_transition_log_odds_heatmap.png",
        fmt="{:+.2f}",
    )
    print("[B] wrote transition heatmaps")

    top_k_path = reports_dir / "act_transition_top_k.json"
    with top_k_path.open("w") as f:
        json.dump({
            "labels": list(ACT_LABELS),
            "k": args.top_k,
            "by_delta": _top_k(delta, args.top_k, list(ACT_LABELS)),
            "by_log_odds": _top_k(log_odds, args.top_k, list(ACT_LABELS)),
        }, f, indent=2)
    print(f"[B] wrote {top_k_path}")

    # ── C. alignment between learned bias and empirical signal ───────────
    learned_inner = bias_table[:NUM_ACTS, :NUM_ACTS]
    sp_delta = _spearman(learned_inner, delta)
    sp_log_odds = _spearman(learned_inner, log_odds)
    alignment_path = reports_dir / "act_transition_alignment.json"
    with alignment_path.open("w") as f:
        json.dump({
            "spearman_learned_vs_delta": sp_delta,
            "spearman_learned_vs_log_odds": sp_log_odds,
            "n_cells": int(NUM_ACTS * NUM_ACTS),
            "interpretation": (
                "Spearman rank correlation between the learned 10x10 act_pair_bias "
                "(excluding [CLS] sentinel) and the empirical winning-vs-losing "
                "transition signal on the test split. Positive => model agrees "
                "with the data on which transitions favor persuasion."
            ),
        }, f, indent=2)
    print(f"[C] spearman(learned, delta)    = {sp_delta:+.4f}")
    print(f"[C] spearman(learned, log_odds) = {sp_log_odds:+.4f}")
    print(f"[C] wrote {alignment_path}")


if __name__ == "__main__":
    main()
