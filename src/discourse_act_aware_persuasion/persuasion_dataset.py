"""
Dataset for the Winning-Arguments persuasion task.

Each example is one argument chain (variable-length sequence of comments).
The DiscoursePredictor encodes each comment into a 832-dim discourse-aware
latent vector (768 BERT [CLS] + 64 discourse-act embedding).

Encodings are cached to disk so the classifier only runs once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class PersuasionDataset(Dataset):
    """
    Args:
        df:        DataFrame slice (one split).
        predictor: DiscoursePredictor (frozen). Pass None to load from cache only.
        cache_dir: Directory to cache encoded latents. Encoding runs once, then
                   subsequent loads are instant.
        batch_size: Encoding batch size passed to predictor.encode().
    """

    LATENT_FILE  = "latents.npy"
    LENGTHS_FILE = "lengths.npy"
    LABELS_FILE  = "labels.npy"
    ACT_IDS_FILE = "act_ids.npy"   # discourse act label ID per comment (for transition attention)

    def __init__(
        self,
        df: pd.DataFrame,
        predictor,
        cache_dir: Path,
        batch_size: int = 32,
    ):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        latent_path  = cache_dir / self.LATENT_FILE
        lengths_path = cache_dir / self.LENGTHS_FILE
        labels_path  = cache_dir / self.LABELS_FILE
        act_ids_path = cache_dir / self.ACT_IDS_FILE

        all_cached = (latent_path.exists() and lengths_path.exists()
                      and labels_path.exists() and act_ids_path.exists())

        if all_cached:
            print(f"Loading cached encodings from {cache_dir}")
            all_latents_flat    = np.load(latent_path)
            self._lengths       = np.load(lengths_path).tolist()
            self._labels        = np.load(labels_path).tolist()
            all_act_ids_flat    = np.load(act_ids_path)
        else:
            if predictor is None:
                raise ValueError(f"No cache found at {cache_dir} and predictor is None.")
            print(f"Encoding {len(df)} chains → {cache_dir}")
            all_latents_flat, all_act_ids_flat, self._lengths, self._labels = self._encode(
                df, predictor, batch_size
            )
            np.save(latent_path,  all_latents_flat)
            np.save(lengths_path, np.array(self._lengths))
            np.save(labels_path,  np.array(self._labels))
            np.save(act_ids_path, all_act_ids_flat)

        # Reconstruct per-chain arrays from the flat arrays
        self._latents:  List[np.ndarray] = []
        self._act_ids:  List[np.ndarray] = []
        offset = 0
        for length in self._lengths:
            self._latents.append(all_latents_flat[offset : offset + length])
            self._act_ids.append(all_act_ids_flat[offset : offset + length])
            offset += length

    @staticmethod
    def _encode(
        df: pd.DataFrame,
        predictor,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """Encode all chains. Returns flat latents, flat act_ids, lengths, labels."""
        all_latents:  List[np.ndarray] = []
        all_act_ids:  List[np.ndarray] = []
        lengths:      List[int] = []
        labels:       List[int] = []

        for _, row in df.iterrows():
            comments  = json.loads(row["comment_texts"])
            ack_flags = json.loads(row["comment_is_delta_ack"])

            # ── Critical: remove delta-ack comments before encoding ───────
            # 61% of winning chains contain an OP comment explicitly saying
            # "∆ you changed my mind." Keeping these causes leakage — the
            # model learns to detect concession language, not persuasion.
            comments = [c for c, is_ack in zip(comments, ack_flags) if not is_ack]
            comments = [c.strip() for c in comments if c and c.strip()]
            if not comments:
                comments = [""]

            encoded = predictor.encode(comments, batch_size=batch_size)

            all_latents.append(encoded["latent"])        # (N, latent_dim)
            all_act_ids.append(encoded["predicted_ids"]) # (N,)  int64 label IDs
            lengths.append(len(comments))
            labels.append(int(row["label"]))

        return (
            np.concatenate(all_latents,  axis=0),
            np.concatenate(all_act_ids,  axis=0),
            lengths,
            labels,
        )

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "latents": torch.tensor(self._latents[idx], dtype=torch.float32),
            "act_ids": torch.tensor(self._act_ids[idx], dtype=torch.long),
            "length":  torch.tensor(self._lengths[idx], dtype=torch.long),
            "label":   torch.tensor(self._labels[idx],  dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length chains to the max length in the batch."""
    latents  = [item["latents"]  for item in batch]
    act_ids  = [item["act_ids"]  for item in batch]
    lengths  = torch.stack([item["length"] for item in batch])
    labels   = torch.stack([item["label"]  for item in batch])

    max_len    = int(lengths.max().item())
    latent_dim = latents[0].shape[-1]

    padded_latents  = torch.zeros(len(batch), max_len, latent_dim)
    padded_act_ids  = torch.zeros(len(batch), max_len, dtype=torch.long)  # pad with label 0

    for i, (lat, act, length) in enumerate(zip(latents, act_ids, lengths)):
        n = length.item()
        padded_latents[i, :n] = lat
        padded_act_ids[i, :n] = act

    return {
        "latents":  padded_latents,
        "act_ids":  padded_act_ids,
        "lengths":  lengths,
        "labels":   labels,
    }


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    parquet_path: Path,
    predictor,
    cache_root: Path,
    batch_size: int = 32,
    encode_batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Returns a dict of DataLoaders keyed by split name: "train", "val", "test".
    """
    df = pd.read_parquet(parquet_path)
    loaders = {}
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split].reset_index(drop=True)
        dataset = PersuasionDataset(
            df=split_df,
            predictor=predictor,
            cache_dir=cache_root / split,
            batch_size=encode_batch_size,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        print(f"  {split}: {len(dataset)} examples")
    return loaders
