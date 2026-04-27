"""Frozen per-comment encoders for the two model variants.

Variant 1 (standalone): FrozenBertCommentEncoder — fresh `bert-base-uncased`,
                        returns the [CLS] embedding (768-d) per comment.
Variant 2 (with discourse): FrozenDiscourseEncoder — loads a trained
                            BertDiscourseClassifier from HF Hub and returns
                            its discourse-aware latent (832-d) and the
                            argmax discourse-act id per comment.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import BertModel, BertTokenizer


def _flatten_thread_texts(thread_texts: List[List[str]]) -> Tuple[List[str], List[Tuple[int, int]]]:
    flat: List[str] = []
    index: List[Tuple[int, int]] = []
    for b, comments in enumerate(thread_texts):
        for i, text in enumerate(comments):
            flat.append(text)
            index.append((b, i))
    return flat, index


class FrozenBertCommentEncoder(nn.Module):
    """Comment encoder for variant 1 (standalone).

    Wraps a frozen `bert-base-uncased`. Encodes a ragged batch of threads
    (List[List[str]]) into a (B, N_max, 768) tensor of [CLS] embeddings,
    flattening internally so the BERT forward sees shape
    (sum_of_comments, max_length).
    """

    OUTPUT_DIM = 768

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        max_length: int = 128,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.max_length = max_length
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.bert.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        thread_texts: List[List[str]],
        comment_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N_max = comment_mask.shape
        flat, index = _flatten_thread_texts(thread_texts)
        if not flat:
            return torch.zeros(B, N_max, self.OUTPUT_DIM, device=self.device)

        enc = self.tokenizer(
            flat,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.bert(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        )
        cls = out.last_hidden_state[:, 0, :]  # (sum_n, 768)

        result = torch.zeros(B, N_max, self.OUTPUT_DIM, device=cls.device, dtype=cls.dtype)
        for k, (b, i) in enumerate(index):
            result[b, i] = cls[k]
        return result


class FrozenDiscourseEncoder(nn.Module):
    """Comment encoder for variant 2 (uses coarse_discourse_cls).

    Loads a trained BertDiscourseClassifier checkpoint from Hugging Face Hub
    and returns, per comment, the 832-d discourse-aware latent plus the
    predicted discourse-act id.
    """

    NUM_LABELS = 10

    def __init__(
        self,
        hf_repo_id: str = "Vijayrathank/discourse_act_classifier",
        checkpoint_filename: str = "bert_discourse_classifier.pt",
        max_length: int = 128,
        local_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self._ensure_repo_root_on_path()
        from coarse_discourse_cls.model import BertDiscourseClassifier  # noqa: WPS433

        if local_checkpoint is not None:
            ckpt_path = local_checkpoint
        else:
            ckpt_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=checkpoint_filename,
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        bert_name = ckpt.get("bert_model_name", "bert-base-uncased")
        class_dim = ckpt.get("class_embedding_dim", 64)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.classifier = BertDiscourseClassifier(
            bert_model_name=bert_name,
            num_labels=self.NUM_LABELS,
            class_embedding_dim=class_dim,
        )
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False

        self.max_length = max_length
        self.latent_dim = self.classifier.latent_dim  # 768 + 64

    @property
    def output_dim(self) -> int:
        return self.latent_dim

    @property
    def device(self) -> torch.device:
        return next(self.classifier.parameters()).device

    @staticmethod
    def _ensure_repo_root_on_path() -> None:
        # coarse_discourse_cls/ lives at the repo root and is a PEP-420
        # namespace package (no __init__.py). Add the repo root to sys.path
        # so the import resolves.
        here = Path(__file__).resolve()
        repo_root = here.parents[4]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def train(self, mode: bool = True):
        super().train(mode)
        self.classifier.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        thread_texts: List[List[str]],
        comment_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_max = comment_mask.shape
        flat, index = _flatten_thread_texts(thread_texts)
        if not flat:
            latent = torch.zeros(B, N_max, self.latent_dim, device=self.device)
            acts = torch.zeros(B, N_max, dtype=torch.long, device=self.device)
            return latent, acts

        enc = self.tokenizer(
            flat,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.classifier.encode(
            enc["input_ids"],
            enc["attention_mask"],
            enc.get("token_type_ids"),
        )
        latent = out["latent"]              # (sum_n, 832)
        pred = out["predicted_ids"]         # (sum_n,)

        latent_full = torch.zeros(B, N_max, self.latent_dim, device=latent.device, dtype=latent.dtype)
        acts_full = torch.zeros(B, N_max, dtype=torch.long, device=pred.device)
        for k, (b, i) in enumerate(index):
            latent_full[b, i] = latent[k]
            acts_full[b, i] = pred[k]
        return latent_full, acts_full
