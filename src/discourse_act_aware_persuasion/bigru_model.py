"""
Bidirectional GRU persuasion model.

Input:  padded sequence of discourse-aware comment latents  (B, T, input_dim)
Output: binary logit per chain  (B, 1)  — use BCEWithLogitsLoss

Architecture:
    BiGRU (input_dim → hidden_dim*2)
        ↓  mean-pool over valid (non-padding) timesteps
    Linear(hidden_dim*2 → 128) → ReLU → Dropout
    Linear(128 → 1)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiGRUPersuasionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 832,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        latents: torch.Tensor,   # (B, T, input_dim)
        lengths: torch.Tensor,   # (B,) actual sequence lengths
    ) -> Dict[str, torch.Tensor]:
        # Pack → GRU → unpack
        packed = pack_padded_sequence(
            latents, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T, hidden*2)

        # Mean-pool over valid timesteps only
        mask = self._length_mask(lengths, out.size(1), out.device)  # (B, T, 1)
        pooled = (out * mask).sum(dim=1) / lengths.unsqueeze(1).float()  # (B, hidden*2)

        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return {"logits": logits}

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int, device) -> torch.Tensor:
        """Boolean mask (B, T, 1) — True for valid positions."""
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        mask = (idx < lengths.unsqueeze(1)).unsqueeze(-1).float()  # (B, T, 1)
        return mask
