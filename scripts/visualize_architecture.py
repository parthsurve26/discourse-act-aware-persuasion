"""
Architecture visualization for the Discourse-Act-Aware Persuasion Model.
Generates two figures:
  1. Full pipeline (CMV chain → prediction)
  2. BiGRU internals (one timestep expanded)

Usage:
    uv run python scripts/visualize_architecture.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT_DIR = Path("data/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(
    bert      = "#4A90D9",
    discourse = "#E8A838",
    concat    = "#7B68EE",
    gru       = "#50C878",
    mlp       = "#E05C5C",
    input     = "#B0C4DE",
    output    = "#FFD700",
    arrow     = "#555555",
    bg        = "#F8F9FA",
    white     = "#FFFFFF",
    text_dark = "#1A1A2E",
)

def box(ax, x, y, w, h, label, sublabel="", color=C["bert"], fontsize=10, sublabelsize=8):
    """Draw a rounded rectangle with a label."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor="#333333",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.06 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=C["text_dark"], zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel,
                ha="center", va="center", fontsize=sublabelsize,
                color="#444444", zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", color=C["arrow"]):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, fontsize=7.5, color="#555555",
                ha="left", va="center", style="italic", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fig_full_pipeline():
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.axis("off")
    ax.set_title("Discourse-Act-Aware Persuasion Model — Full Pipeline",
                 fontsize=14, fontweight="bold", pad=14, color=C["text_dark"])

    # ── 3 example comments ───────────────────────────────────────────────────
    comments = [
        ("Comment 1", '"I disagree because…"'),
        ("Comment 2", '"Great point, but…"'),
        ("Comment 3", '"Could you explain…"'),
    ]
    ys = [7.2, 5.0, 2.8]
    for (title, quote), y in zip(comments, ys):
        box(ax, 1.5, y, 2.6, 1.0, title, quote, color=C["input"], fontsize=9)

    # ── BERT encoder ─────────────────────────────────────────────────────────
    for y in ys:
        arrow(ax, 2.85, y, 3.55, y)
    box(ax, 4.1, 5.0, 2.2, 5.2, "BERT Encoder\n(frozen)",
        "bert-base-uncased", color=C["bert"], fontsize=10)
    ax.text(4.1, 2.1, "Shared weights\nacross all comments",
            ha="center", va="center", fontsize=7.5, color="#555555", style="italic")

    # CLS outputs
    cls_ys = [7.2, 5.0, 2.8]
    for y in cls_ys:
        arrow(ax, 5.25, y, 5.95, y, "768-dim\n[CLS]")

    # ── Discourse classifier head ─────────────────────────────────────────────
    box(ax, 6.8, 5.0, 1.5, 5.2, "Discourse\nAct Head\n(frozen)",
        "Linear→GELU\n→Linear(10)", color=C["discourse"], fontsize=9)

    # logits → label embeddings
    for y in cls_ys:
        arrow(ax, 7.6, y, 8.2, y)

    # ── Label embedding lookup ────────────────────────────────────────────────
    box(ax, 8.85, 7.2, 1.6, 0.85, "Label Embedding",  "64-dim", color=C["discourse"], fontsize=9)
    box(ax, 8.85, 5.0, 1.6, 0.85, "Label Embedding",  "64-dim", color=C["discourse"], fontsize=9)
    box(ax, 8.85, 2.8, 1.6, 0.85, "Label Embedding",  "64-dim", color=C["discourse"], fontsize=9)

    # ── Concat ───────────────────────────────────────────────────────────────
    for y in cls_ys:
        # CLS from BERT goes right across discourse head to concat
        arrow(ax, 9.7, y, 10.3, y)

    box(ax, 10.95, 7.2, 1.2, 0.85, "Concat", "832-dim", color=C["concat"], fontsize=9)
    box(ax, 10.95, 5.0, 1.2, 0.85, "Concat", "832-dim", color=C["concat"], fontsize=9)
    box(ax, 10.95, 2.8, 1.2, 0.85, "Concat", "832-dim", color=C["concat"], fontsize=9)

    ax.text(10.95, 1.5, "768 (BERT) + 64 (discourse act)\n= 832-dim per comment",
            ha="center", va="center", fontsize=8, color="#555555", style="italic")

    # bracket showing "sequence"
    ax.annotate("", xy=(11.6, 7.65), xytext=(11.6, 2.35),
                arrowprops=dict(arrowstyle="-", color="#999", lw=1, linestyle="dashed"))
    ax.text(11.75, 5.0, "sequence\n(T=3)", ha="left", va="center",
            fontsize=8, color="#999", rotation=90)

    # ── BiGRU ────────────────────────────────────────────────────────────────
    for y in cls_ys:
        arrow(ax, 11.6, y, 12.3, y)
    box(ax, 13.1, 5.0, 1.6, 5.2, "Bidirectional\nGRU",
        "hidden=128\n→ 256-dim out", color=C["gru"], fontsize=10)

    arrow(ax, 13.95, 5.0, 14.6, 5.0, "mean\npool →\n256-dim")

    # ── MLP head ─────────────────────────────────────────────────────────────
    box(ax, 15.4, 5.0, 1.5, 1.5, "MLP Head",
        "Linear(256→128)\nReLU  Dropout\nLinear(128→1)", color=C["mlp"], fontsize=8.5)

    arrow(ax, 16.2, 5.0, 16.9, 5.0)

    # ── Output ───────────────────────────────────────────────────────────────
    box(ax, 17.4, 5.0, 0.9, 0.85, "Δ?", "sigmoid", color=C["output"], fontsize=11)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        (C["input"],     "Raw comment text"),
        (C["bert"],      "BERT encoder (frozen)"),
        (C["discourse"], "Discourse act classifier (frozen)"),
        (C["concat"],    "Concatenated latent (832-dim)"),
        (C["gru"],       "Bidirectional GRU (trained)"),
        (C["mlp"],       "MLP head (trained)"),
        (C["output"],    "Delta prediction"),
    ]
    for i, (color, label) in enumerate(legend_items):
        px, py = 0.3, 1.4 - i * 0.22
        rect = mpatches.Patch(facecolor=color, edgecolor="#333", label=label)
        ax.add_patch(FancyBboxPatch((px, py), 0.25, 0.15,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="#333", linewidth=1, zorder=3))
        ax.text(px + 0.35, py + 0.075, label, fontsize=7.5, va="center", color=C["text_dark"])

    plt.tight_layout()
    out = OUT_DIR / "architecture_full_pipeline.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — BiGRU internals
# ══════════════════════════════════════════════════════════════════════════════

def fig_bigru_internals():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.axis("off")
    ax.set_title("BiGRU Internals — Processing an Argument Chain (T=3 comments)",
                 fontsize=13, fontweight="bold", pad=12, color=C["text_dark"])

    # ── Input vectors ─────────────────────────────────────────────────────────
    xs = [3.0, 7.5, 12.0]
    labels = ["Comment 1\n832-dim latent\n(disagreement)",
              "Comment 2\n832-dim latent\n(elaboration)",
              "Comment 3\n832-dim latent\n(question)"]
    for x, lbl in zip(xs, labels):
        box(ax, x, 1.2, 2.4, 1.0, lbl, color=C["concat"], fontsize=8.5)

    # ── Forward GRU cells ─────────────────────────────────────────────────────
    ax.text(0.6, 3.5, "Forward\nGRU →", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C["gru"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["gru"], alpha=0.2))
    fwd_cells = []
    for x in xs:
        b = FancyBboxPatch((x - 1.0, 3.0), 2.0, 1.0,
                           boxstyle="round,pad=0.08",
                           facecolor=C["gru"], edgecolor="#333", linewidth=1.5, zorder=3)
        ax.add_patch(b)
        ax.text(x, 3.5, "GRU cell\nh=128", ha="center", va="center",
                fontsize=9, fontweight="bold", color=C["text_dark"], zorder=4)
        fwd_cells.append(x)

    # Forward hidden state arrows between cells
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 1.05, 3.5, xs[i+1] - 1.05, 3.5, "→ h_fwd")

    # Input → forward cell
    for x in xs:
        arrow(ax, x, 1.75, x, 2.95)

    # ── Backward GRU cells ────────────────────────────────────────────────────
    ax.text(15.4, 5.5, "← Backward\nGRU", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#3A8A5A",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["gru"], alpha=0.2))
    bwd_color = "#3DBE71"
    for x in xs:
        b = FancyBboxPatch((x - 1.0, 5.0), 2.0, 1.0,
                           boxstyle="round,pad=0.08",
                           facecolor=bwd_color, edgecolor="#333", linewidth=1.5, zorder=3)
        ax.add_patch(b)
        ax.text(x, 5.5, "GRU cell\nh=128", ha="center", va="center",
                fontsize=9, fontweight="bold", color=C["text_dark"], zorder=4)

    # Backward hidden state arrows (right to left)
    for i in range(len(xs) - 1, 0, -1):
        arrow(ax, xs[i] - 1.05, 5.5, xs[i-1] + 1.05, 5.5, "← h_bwd")

    # Input → backward cell
    for x in xs:
        arrow(ax, x, 1.75, x, 4.95)

    # ── Concat fwd + bwd outputs ──────────────────────────────────────────────
    for x in xs:
        box(ax, x, 6.9, 2.2, 0.7, "[h_fwd ; h_bwd]  256-dim",
            color=C["concat"], fontsize=8)
        arrow(ax, x, 4.05, x, 5.0)   # fwd cell top → concat (via bwd)
        arrow(ax, x, 6.05, x, 6.52)  # bwd cell top → concat box

    # ── Mean pool ────────────────────────────────────────────────────────────
    # Arrow from all concat boxes to pool
    ax.annotate("", xy=(7.5, 7.55), xytext=(3.0, 7.28),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=1.3), zorder=2)
    ax.annotate("", xy=(7.5, 7.55), xytext=(12.0, 7.28),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=1.3), zorder=2)
    box(ax, 7.5, 7.75, 3.2, 0.6, "Mean Pool over T steps  →  256-dim",
        color=C["gru"], fontsize=9)

    # ── Dimension annotations ─────────────────────────────────────────────────
    ax.text(1.1, 1.2, "input_dim\n= 832", ha="center", va="center",
            fontsize=8, color="#555", style="italic")
    ax.text(1.1, 3.5, "hidden\n= 128", ha="center", va="center",
            fontsize=8, color="#555", style="italic")
    ax.text(1.1, 5.5, "hidden\n= 128", ha="center", va="center",
            fontsize=8, color="#555", style="italic")
    ax.text(1.1, 6.9, "128+128\n= 256", ha="center", va="center",
            fontsize=8, color="#555", style="italic")

    # ── pack_padded note ──────────────────────────────────────────────────────
    ax.text(7.5, 0.35,
            "pack_padded_sequence ensures padding tokens are never passed through GRU cells",
            ha="center", va="center", fontsize=8.5, color="#666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEE", edgecolor="#CCC"))

    plt.tight_layout()
    out = OUT_DIR / "architecture_bigru_internals.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    fig_full_pipeline()
    fig_bigru_internals()
