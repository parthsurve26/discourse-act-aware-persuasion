"""Train/eval loop for the thread transformer (variant-agnostic)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .encoders import FrozenBertCommentEncoder, FrozenDiscourseEncoder
from .model import TransitionAwareThreadTransformer


def _binary_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = (probs >= 0.5).astype(np.int64)
    acc = float((preds == labels).mean()) if labels.size else 0.0

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    auc = _roc_auc(probs, labels)

    if labels.size:
        clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
        ce = float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean())
    else:
        ce = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cross_entropy": ce,
    }


def _roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.0
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, probs.size + 1)
    sum_pos = ranks[labels == 1].sum()
    n_pos = pos.size
    n_neg = neg.size
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class ThreadTrainer:
    def __init__(
        self,
        encoder,
        model: TransitionAwareThreadTransformer,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
    ):
        self.encoder = encoder.to(device)
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _encode_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        thread_texts = batch["comment_texts"]
        comment_mask = batch["comment_mask"].to(self.device)
        if isinstance(self.encoder, FrozenDiscourseEncoder):
            embs, acts = self.encoder(thread_texts, comment_mask)
            return embs, acts
        embs = self.encoder(thread_texts, comment_mask)
        return embs, None

    def _step(self, batch, train: bool) -> Dict[str, torch.Tensor]:
        comment_mask = batch["comment_mask"].to(self.device)
        is_op = batch["is_op"].to(self.device)
        labels = batch["labels"].to(self.device)

        embs, acts = self._encode_batch(batch)
        out = self.model(
            comment_embs=embs,
            is_op=is_op,
            comment_mask=comment_mask,
            labels=labels,
            comment_acts=acts,
        )
        if train:
            self.optimizer.zero_grad()
            out["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return out, labels

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        for batch in loader:
            out, labels = self._step(batch, train=True)
            total_loss += float(out["loss"].item())
            probs = torch.softmax(out["logits"].detach(), dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        metrics = _binary_metrics(np.concatenate(all_probs), np.concatenate(all_labels))
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        for batch in loader:
            out, labels = self._step(batch, train=False)
            total_loss += float(out["loss"].item())
            probs = torch.softmax(out["logits"], dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        metrics = _binary_metrics(np.concatenate(all_probs), np.concatenate(all_labels))
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stop_metric: str = "auc",
        early_stop_patience: int = 3,
    ) -> List[Dict]:
        history: List[Dict] = []
        best_score = -float("inf")
        best_state = None
        epochs_since_best = 0

        for epoch in range(1, epochs + 1):
            train_m = self.train_epoch(train_loader)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}}
            if val_loader is not None:
                val_m = self.evaluate(val_loader)
                row.update({f"val_{k}": v for k, v in val_m.items()})
                score = val_m[early_stop_metric]
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
            history.append(row)
            self._log(row)
            if val_loader is not None and epochs_since_best >= early_stop_patience:
                print(f"Early stop at epoch {epoch}; best val_{early_stop_metric}={best_score:.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    @staticmethod
    def _log(row: Dict) -> None:
        parts = [f"Epoch {row['epoch']:>2}"]
        for k, v in row.items():
            if k == "epoch":
                continue
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4f}")
        print("  |  ".join(parts))
