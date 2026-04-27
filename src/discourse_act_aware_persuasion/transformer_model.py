"""
Transition-Aware Transformer for persuasion prediction.

The key idea: standard self-attention treats all comment-to-comment
relationships equally. We add a *learned transition bias* T[h, act_i, act_j]
to each attention score, so the model can learn that certain discourse-act
transitions (e.g. disagreement → elaboration) are more persuasive than others.

Architecture:
    (B, T, input_dim=832) + (B, T) act_ids
        ↓  Linear projection
    (B, T, d_model=256)
        ↓  Positional encoding (over comment positions)
        ↓  N × TransitionAwareEncoderLayer
            ├─ TransitionAwareMultiHeadAttention  ← the novel part
            ├─ Add & Norm
            ├─ Feed-Forward (d_model → d_ff → d_model)
            └─ Add & Norm
        ↓  Mean pool over valid positions
    (B, d_model)
        ↓  MLP head
    (B, 1) logit  →  BCEWithLogitsLoss
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_LABELS = 10   # discourse act vocabulary size


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Transition-Aware Multi-Head Attention
# ─────────────────────────────────────────────────────────────────────────────

class TransitionAwareMultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with one addition:

        score(i→j) = (Q_i · K_j) / √d_k  +  T[head, act_i, act_j]

    T is a learned (num_heads × num_labels × num_labels) parameter.
    Each head can independently learn which discourse-act pairs are important,
    e.g. one head might up-weight "question → answer" transitions while
    another focuses on "disagreement → elaboration".
    """

    def __init__(
        self,
        d_model:    int,
        num_heads:  int,
        num_labels: int = NUM_LABELS,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.num_labels = num_labels
        self.d_k        = d_model // num_heads

        # Standard QKV projections
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        # ── Transition bias ───────────────────────────────────────────────
        # Shape: (num_heads, num_labels, num_labels)
        # transition_bias[h, a, b] = bias added when head h attends from
        # a comment with discourse act 'a' to one with discourse act 'b'.
        # Initialised to zero so it starts as standard attention.
        self.transition_bias = nn.Parameter(
            torch.zeros(num_heads, num_labels, num_labels)
        )

    def _build_transition_bias(self, act_ids: torch.Tensor) -> torch.Tensor:
        """
        Build the (B, H, T, T) additive bias from (B, T) act_ids.

        For each batch b, head h, and position pair (i, j):
            bias[b, h, i, j] = transition_bias[h, act_ids[b,i], act_ids[b,j]]
        """
        B, T = act_ids.shape

        # Expand act_ids to (B, T, T) for row (attending-from) and col (attending-to)
        act_i = act_ids.unsqueeze(2).expand(-1, -1, T)   # (B, T, T)  "from" act
        act_j = act_ids.unsqueeze(1).expand(-1, T, -1)   # (B, T, T)  "to"   act

        # Flatten the (L, L) table into L*L so we can use integer indexing
        flat_idx = act_i * self.num_labels + act_j         # (B, T, T)

        # transition_bias: (H, L, L) → (H, L*L)
        bias_flat = self.transition_bias.view(self.num_heads, -1)

        # Gather: bias_flat[:, flat_idx] → (H, B, T, T) → (B, H, T, T)
        bias = bias_flat[:, flat_idx]                      # (H, B, T, T)
        return bias.permute(1, 0, 2, 3)                    # (B, H, T, T)

    def forward(
        self,
        x:           torch.Tensor,            # (B, T, d_model)
        act_ids:     torch.Tensor,            # (B, T)  int  discourse act per position
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) bool, True = PAD
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, d_k  = self.num_heads, self.d_k

        # ── Project Q, K, V ───────────────────────────────────────────────
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, T, d_model) → (B, H, T, d_k)
            return t.view(B, T, H, d_k).transpose(1, 2)

        Q = split_heads(self.q_proj(x))   # (B, H, T, d_k)
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # ── Attention scores ──────────────────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T, T)

        # ── Add transition bias ───────────────────────────────────────────
        scores = scores + self._build_transition_bias(act_ids)           # (B, H, T, T)

        # ── Mask padding positions ────────────────────────────────────────
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) True where position is a PAD token
            # Broadcast to (B, 1, 1, T) so every query is masked to the same keys
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)       # (B, H, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Weighted sum of values ────────────────────────────────────────
        out = torch.matmul(attn_weights, V)            # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous()         # (B, T, H, d_k)
        out = out.view(B, T, self.d_model)             # (B, T, d_model)
        return self.out_proj(out)                      # (B, T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Encoder Layer  (Attention + FFN + LayerNorm)
# ─────────────────────────────────────────────────────────────────────────────

class TransitionAwareEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model:    int,
        num_heads:  int,
        d_ff:       int,
        num_labels: int = NUM_LABELS,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.attn     = TransitionAwareMultiHeadAttention(d_model, num_heads, num_labels, dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(
        self,
        x:                torch.Tensor,            # (B, T, d_model)
        act_ids:          torch.Tensor,            # (B, T)
        key_padding_mask: Optional[torch.Tensor],  # (B, T)
    ) -> torch.Tensor:
        # ── Self-attention (pre-norm style) ───────────────────────────────
        residual = x
        x = self.norm1(x)
        x = self.attn(x, act_ids, key_padding_mask)
        x = self.dropout(x) + residual

        # ── Feed-forward ──────────────────────────────────────────────────
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual

        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding over comment positions (not sub-word tokens).
    Max sequence = 50 comments (well above the observed max of 10).
    """

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full Transformer Model
# ─────────────────────────────────────────────────────────────────────────────

class TransitionAwareTransformer(nn.Module):
    """
    Full transition-aware transformer for CMV persuasion prediction.

    Args:
        input_dim:  Dimensionality of input comment latents (832).
        d_model:    Internal transformer dimension (256).
        num_heads:  Number of attention heads (4).
        num_layers: Number of encoder layers (2).
        d_ff:       Feed-forward hidden size (512).
        num_labels: Discourse-act vocabulary size (10).
        dropout:    Dropout probability (0.1).
    """

    def __init__(
        self,
        input_dim:  int = 832,
        d_model:    int = 256,
        num_heads:  int = 4,
        num_layers: int = 2,
        d_ff:       int = 512,
        num_labels: int = NUM_LABELS,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # Project 832-dim latents into transformer's d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            TransitionAwareEncoderLayer(d_model, num_heads, d_ff, num_labels, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        latents:  torch.Tensor,   # (B, T, input_dim)
        act_ids:  torch.Tensor,   # (B, T)  discourse act label IDs
        lengths:  torch.Tensor,   # (B,)    actual sequence lengths
    ) -> Dict[str, torch.Tensor]:

        B, T, _ = latents.shape

        # ── Key padding mask: True where position is padding ──────────────
        idx = torch.arange(T, device=lengths.device).unsqueeze(0)   # (1, T)
        key_padding_mask = idx >= lengths.unsqueeze(1)               # (B, T)

        # ── Project input → d_model, add positional encoding ─────────────
        x = self.input_proj(latents)   # (B, T, d_model)
        x = self.pos_enc(x)

        # ── Transformer layers ────────────────────────────────────────────
        for layer in self.layers:
            x = layer(x, act_ids, key_padding_mask)

        x = self.norm(x)

        # ── Mean-pool over valid (non-padding) positions ──────────────────
        mask = (~key_padding_mask).unsqueeze(-1).float()   # (B, T, 1) — 1 = valid
        pooled = (x * mask).sum(dim=1) / lengths.unsqueeze(1).float()  # (B, d_model)

        logits = self.classifier(pooled).squeeze(-1)   # (B,)
        return {"logits": logits}

    def get_transition_bias(self) -> torch.Tensor:
        """
        Return the averaged (across heads) transition bias matrix.
        Shape: (num_labels, num_labels).
        Useful for analysis: which act-pair transitions did the model learn
        are most persuasive?
        """
        from coarse_discourse_cls.model import LABELS
        return {
            "matrix": self.layers[0].attn.transition_bias.mean(dim=0).detach().cpu(),
            "labels": LABELS,
        }
