"""Transition-aware self-attention transformer over comment sequences."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bucket_distance(diff: torch.Tensor, num_buckets: int = 8) -> torch.Tensor:
    """Map signed turn distance to a bucket id in [0, 2*num_buckets-1].

    Buckets are symmetric around 0 with logarithmic spacing for |diff| ≥ 1.
    diff < 0  → buckets [0, num_buckets-1]   (j is earlier than i)
    diff = 0  → bucket  num_buckets-1
    diff > 0  → buckets [num_buckets, 2*num_buckets-1]
    """
    sign = torch.sign(diff)
    abs_d = diff.abs().clamp(min=0)
    log_d = torch.log1p(abs_d.float()).floor().long()
    log_d = log_d.clamp(max=num_buckets - 1)
    bucket = torch.where(
        sign >= 0,
        num_buckets - 1 + log_d * (sign.long().clamp(min=0)),
        num_buckets - 1 - log_d,
    )
    return bucket.clamp(min=0, max=2 * num_buckets - 1)


class TransitionAwareAttention(nn.Module):
    """Multi-head self-attention with an additive (B, N, N) bias term."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # (B, N, D)
        bias: torch.Tensor,              # (B, N, N) additive
        key_padding_mask: torch.Tensor,  # (B, N) True = pad
    ) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                           # each (B, N, H, dh)
        q = q.transpose(1, 2)                                 # (B, H, N, dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        scores = scores + bias.unsqueeze(1)                                     # broadcast over heads
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)            # (B, N, D)
        return self.out(ctx)


class TransitionTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = TransitionAwareAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_padding_mask):
        x = x + self.drop(self.attn(self.ln1(x), bias, key_padding_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class TransitionAwareThreadTransformer(nn.Module):
    """Comment-level transformer with transition-aware additive attention bias.

    Args:
        comment_dim:        input dim of per-comment vectors (768 or 832)
        d_model:            internal transformer width
        n_heads:            attention heads
        n_layers:           transformer depth
        ffn_dim:            FFN hidden size
        dropout:            dropout prob
        max_comments:       max thread length used for the position embedding
        num_distance_buckets:   half-window of signed turn-distance buckets
        use_discourse_acts: enable the (act_i, act_j) transition bias table
        num_acts:           vocabulary size for discourse acts (default 10)
        num_labels:         output classes (2 — winning vs losing)
    """

    def __init__(
        self,
        comment_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_comments: int = 32,
        num_distance_buckets: int = 8,
        use_discourse_acts: bool = False,
        num_acts: int = 10,
        num_labels: int = 2,
    ):
        super().__init__()
        self.use_discourse_acts = use_discourse_acts
        self.num_distance_buckets = num_distance_buckets

        # +1 length budget for the prepended [CLS_thread] token.
        self.max_seq = max_comments + 1

        self.input_proj = nn.Linear(comment_dim, d_model)
        self.pos_emb = nn.Embedding(self.max_seq, d_model)
        self.speaker_emb = nn.Embedding(3, d_model)  # 0 = challenger, 1 = OP, 2 = [CLS_thread]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transition bias tables. All produce a scalar that is added to the
        # raw attention score for the (i, j) pair, broadcast across heads.
        self.speaker_pair_bias = nn.Embedding(3 * 3, 1)        # 3 speaker types, pair index = a*3+b
        self.distance_bias = nn.Embedding(2 * num_distance_buckets, 1)
        if use_discourse_acts:
            # +1 for the [CLS_thread] "act" sentinel
            self.act_pair_bias = nn.Embedding((num_acts + 1) ** 2, 1)
            self.num_acts = num_acts
        nn.init.zeros_(self.speaker_pair_bias.weight)
        nn.init.zeros_(self.distance_bias.weight)
        if use_discourse_acts:
            nn.init.zeros_(self.act_pair_bias.weight)

        self.layers = nn.ModuleList(
            [TransitionTransformerLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )

        self.input_drop = nn.Dropout(dropout)

    def _build_bias(
        self,
        is_op: torch.Tensor,         # (B, N) long, with [CLS]=2 prepended
        acts: Optional[torch.Tensor],  # (B, N) long or None, with [CLS]=num_acts
    ) -> torch.Tensor:
        B, N = is_op.shape
        device = is_op.device
        idx = torch.arange(N, device=device)
        diff = idx.unsqueeze(0) - idx.unsqueeze(1)             # (N, N), j - i
        dist_bucket = _bucket_distance(diff, self.num_distance_buckets)  # (N, N)
        dist_b = self.distance_bias(dist_bucket).squeeze(-1)             # (N, N)

        sp_pair = is_op.unsqueeze(2) * 3 + is_op.unsqueeze(1)            # (B, N, N)
        sp_b = self.speaker_pair_bias(sp_pair).squeeze(-1)               # (B, N, N)

        bias = sp_b + dist_b.unsqueeze(0)
        if self.use_discourse_acts and acts is not None:
            act_pair = acts.unsqueeze(2) * (self.num_acts + 1) + acts.unsqueeze(1)
            bias = bias + self.act_pair_bias(act_pair).squeeze(-1)
        return bias

    def forward(
        self,
        comment_embs: torch.Tensor,        # (B, N, comment_dim)
        is_op: torch.Tensor,               # (B, N) long
        comment_mask: torch.Tensor,        # (B, N) bool, True = real
        labels: Optional[torch.Tensor] = None,
        comment_acts: Optional[torch.Tensor] = None,  # (B, N) long
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = comment_embs.shape

        x = self.input_proj(comment_embs)                       # (B, N, D)
        x = x + self.speaker_emb(is_op)

        cls = self.cls_token.expand(B, 1, -1)                   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                          # (B, N+1, D)

        seq_idx = torch.arange(N + 1, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(seq_idx)
        x = self.input_drop(x)

        # Augment per-token features for bias computation: prepend [CLS] sentinels.
        cls_speaker = torch.full((B, 1), 2, dtype=is_op.dtype, device=is_op.device)
        is_op_full = torch.cat([cls_speaker, is_op], dim=1)     # (B, N+1)

        if self.use_discourse_acts and comment_acts is not None:
            cls_act = torch.full(
                (B, 1), self.num_acts, dtype=comment_acts.dtype, device=comment_acts.device
            )
            acts_full = torch.cat([cls_act, comment_acts], dim=1)
        else:
            acts_full = None

        bias = self._build_bias(is_op_full, acts_full)          # (B, N+1, N+1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=comment_mask.device)
        full_mask = torch.cat([cls_mask, comment_mask], dim=1)  # (B, N+1) True = real
        key_padding_mask = ~full_mask                            # True = pad

        for layer in self.layers:
            x = layer(x, bias, key_padding_mask)

        x = self.ln_final(x)
        cls_out = x[:, 0]                                        # (B, D)
        logits = self.classifier(cls_out)                        # (B, num_labels)

        result = {"logits": logits, "thread_embedding": cls_out}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result
