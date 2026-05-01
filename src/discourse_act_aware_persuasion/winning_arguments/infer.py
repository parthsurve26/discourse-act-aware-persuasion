"""Inference helper for the trained thread transformer."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .data import ThreadRecord, _strip_delta_tokens, collate_threads
from .encoders import FrozenDiscourseEncoder
from .model import TransitionAwareThreadTransformer


class WinningArgumentsPredictor:
    def __init__(
        self,
        encoder,
        model: TransitionAwareThreadTransformer,
        device: torch.device,
    ):
        self.encoder = encoder.to(device).eval()
        self.model = model.to(device).eval()
        self.device = device

    @staticmethod
    def thread_from_dict(thread: Dict) -> ThreadRecord:
        comment_texts = [_strip_delta_tokens(str(t)) for t in thread["comment_texts"]]
        comment_is_op = [bool(x) for x in thread["comment_is_op"]]
        if "comment_is_delta_ack" in thread:
            keep = [not bool(a) for a in thread["comment_is_delta_ack"]]
            comment_texts = [t for t, k in zip(comment_texts, keep) if k and t]
            comment_is_op = [s for s, k in zip(comment_is_op, keep) if k]
        return ThreadRecord(
            comment_texts=comment_texts,
            comment_is_op=comment_is_op,
            label=int(thread.get("label", -1)),
            thread_id=str(thread.get("argument_thread_id", "unknown")),
        )

    @torch.no_grad()
    def predict(self, threads: List[Dict]) -> List[Dict]:
        records = [self.thread_from_dict(t) for t in threads]
        batch = collate_threads(records)

        comment_mask = batch["comment_mask"].to(self.device)
        is_op = batch["is_op"].to(self.device)

        if isinstance(self.encoder, FrozenDiscourseEncoder):
            embs, acts = self.encoder(batch["comment_texts"], comment_mask)
        else:
            embs = self.encoder(batch["comment_texts"], comment_mask)
            acts = None

        out = self.model(
            comment_embs=embs,
            is_op=is_op,
            comment_mask=comment_mask,
            comment_acts=acts,
        )
        probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()

        results: List[Dict] = []
        for i, p in enumerate(probs):
            results.append({
                "thread_id": records[i].thread_id,
                "label": int(p >= 0.5),
                "prob_winning": float(p),
                "num_comments": len(records[i]),
            })
        return results
