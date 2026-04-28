from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

from .evaluation import classification_metrics
from .io import read_parquet
from .utils import ensure_parent


def _infer_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_comment_list(values: Iterable[object]) -> List[str]:
    cleaned = [str(value).strip() for value in values]
    return [value for value in cleaned if value]


def _parse_json_list(raw: object) -> List[object]:
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def _parse_comment_texts(row: pd.Series, use_ack_masked: bool = True) -> List[str]:
    raw = row.get("comment_texts")
    parsed = _parse_json_list(raw)
    comments = _clean_comment_list(parsed)
    if comments and use_ack_masked:
        ack_flags_raw = _parse_json_list(row.get("comment_is_delta_ack"))
        ack_flags = [bool(flag) for flag in ack_flags_raw]
        if len(ack_flags) == len(comments):
            comments = [text for text, is_ack in zip(comments, ack_flags) if not is_ack]
            if comments:
                return comments
    if comments:
        return comments

    fallback_col = "text_ack_masked" if use_ack_masked and "text_ack_masked" in row else "text"
    text = str(row.get(fallback_col, "") or "").strip()
    return [text] if text else []


@dataclass
class EncoderBundle:
    model: nn.Module
    tokenizer: AutoTokenizer
    feature_dim: int
    discourse_dim: int


class TextClassificationDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class ThreadTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_col: str):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        return {"text": row[self.text_col], "label": int(row["label"])}


class CommentThreadDataset(Dataset):
    def __init__(self, df: pd.DataFrame, use_ack_masked: bool = True):
        self.df = df.reset_index(drop=True)
        self.use_ack_masked = use_ack_masked

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        return {
            "comments": _parse_comment_texts(row, use_ack_masked=self.use_ack_masked),
            "label": int(row["label"]),
        }


class BatchTokenizerCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in batch]
        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded


class CommentThreadCollator:
    def __init__(self, tokenizer, max_comments: int, max_comment_length: int):
        self.tokenizer = tokenizer
        self.max_comments = max_comments
        self.max_comment_length = max_comment_length

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        comments_per_row = []
        flat_comments: List[str] = []
        for sample in batch:
            comments = [text for text in sample["comments"] if text][: self.max_comments]
            if not comments:
                comments = [""]
            comments_per_row.append(comments)
            flat_comments.extend(comments)

        encoded = self.tokenizer(
            flat_comments,
            truncation=True,
            padding="max_length",
            max_length=self.max_comment_length,
            return_tensors="pt",
        )

        max_comments = max(len(comments) for comments in comments_per_row)
        input_ids = torch.zeros((batch_size, max_comments, encoded["input_ids"].shape[-1]), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = (
            torch.zeros_like(input_ids)
            if "token_type_ids" in encoded
            else None
        )
        comment_mask = torch.zeros((batch_size, max_comments), dtype=torch.bool)

        cursor = 0
        for row_idx, comments in enumerate(comments_per_row):
            count = len(comments)
            input_ids[row_idx, :count] = encoded["input_ids"][cursor : cursor + count]
            attention_mask[row_idx, :count] = encoded["attention_mask"][cursor : cursor + count]
            if token_type_ids is not None:
                token_type_ids[row_idx, :count] = encoded["token_type_ids"][cursor : cursor + count]
            comment_mask[row_idx, :count] = True
            cursor += count

        batch_tensors = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "comment_mask": comment_mask,
            "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
        }
        if token_type_ids is not None:
            batch_tensors["token_type_ids"] = token_type_ids
        return batch_tensors


class DiscourseEncoderModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        class_embedding_dim: int = 64,
        dropout: float = 0.3,
        local_files_only: bool = False,
        init_from_pretrained: bool = True,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        # Keep the module name aligned with the teammate checkpoint keys.
        if init_from_pretrained:
            self.bert = AutoModel.from_pretrained(base_model_name, local_files_only=local_files_only)
        else:
            if base_model_name != "bert-base-uncased":
                raise ValueError(
                    "Checkpoint-only initialization is currently implemented for `bert-base-uncased`."
                )
            self.bert = BertModel(BertConfig())
        self.hidden_size = self.bert.config.hidden_size
        self.class_embedding_dim = class_embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )
        self.class_embeddings = nn.Embedding(num_labels, class_embedding_dim)
        nn.init.normal_(self.class_embeddings.weight, mean=0.0, std=0.02)

    @property
    def latent_dim(self) -> int:
        return self.hidden_size + self.class_embedding_dim

    def encode_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        probs = torch.softmax(logits, dim=-1)
        discourse_embedding = probs @ self.class_embeddings.weight
        latent = torch.cat([pooled, discourse_embedding], dim=-1)
        return {
            "logits": logits,
            "pooled_output": pooled,
            "probabilities": probs,
            "discourse_embedding": discourse_embedding,
            "latent": latent,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        result = self.encode_features(input_ids, attention_mask, token_type_ids)
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(result["logits"], labels)
        return result


class TransformerThreadClassifier(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_labels: int = 2,
        dropout: float = 0.2,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(base_model_name, local_files_only=local_files_only)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result


class CommentAttentionClassifier(nn.Module):
    def __init__(
        self,
        encoder_bundle: EncoderBundle,
        num_labels: int = 2,
        lstm_hidden_size: int = 128,
        dropout: float = 0.2,
        use_bilstm: bool = True,
        use_attention_bias: bool = False,
        freeze_encoder: bool = False,
        use_latent_features: bool = False,
    ):
        super().__init__()
        self.encoder = encoder_bundle.model
        self.feature_dim = encoder_bundle.feature_dim
        self.discourse_dim = encoder_bundle.discourse_dim
        self.use_attention_bias = use_attention_bias and self.discourse_dim > 0
        self.use_latent_features = use_latent_features and self.discourse_dim > 0

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        sequence_dim = self.feature_dim
        if self.use_latent_features:
            sequence_dim = self.feature_dim + self.discourse_dim

        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.sequence_encoder = nn.LSTM(
                input_size=sequence_dim,
                hidden_size=lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            attention_input_dim = lstm_hidden_size * 2
        else:
            self.sequence_encoder = None
            attention_input_dim = sequence_dim

        self.attention_mlp = nn.Sequential(
            nn.Linear(attention_input_dim, attention_input_dim),
            nn.Tanh(),
        )
        self.attention_vector = nn.Linear(attention_input_dim, 1, bias=False)
        self.attention_bias = nn.Linear(self.discourse_dim, 1, bias=False) if self.use_attention_bias else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attention_input_dim, attention_input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attention_input_dim, num_labels),
        )

    def _encode_comments(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_comments, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_comments, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_comments, seq_len)
        flat_token_type_ids = (
            token_type_ids.view(batch_size * num_comments, seq_len)
            if token_type_ids is not None
            else None
        )

        outputs = self.encoder.encode_features(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
        )
        pooled = outputs["pooled_output"].view(batch_size, num_comments, -1)
        discourse_embedding = outputs["discourse_embedding"].view(batch_size, num_comments, -1)
        if self.use_latent_features:
            features = outputs["latent"].view(batch_size, num_comments, -1)
        else:
            features = pooled
        return {
            "features": features,
            "discourse_embedding": discourse_embedding,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        comment_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self._encode_comments(input_ids, attention_mask, token_type_ids)
        sequence = encoded["features"]

        if self.sequence_encoder is not None:
            sequence, _ = self.sequence_encoder(sequence)

        attention_hidden = self.attention_mlp(sequence)
        attention_logits = self.attention_vector(attention_hidden).squeeze(-1)
        if self.attention_bias is not None:
            attention_logits = attention_logits + self.attention_bias(encoded["discourse_embedding"]).squeeze(-1)

        attention_logits = attention_logits.masked_fill(~comment_mask, float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), sequence).squeeze(1)
        logits = self.classifier(context)

        result = {"logits": logits, "attention_weights": attention_weights}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result


def build_label_vocab(labels: Iterable[str]) -> tuple[Dict[str, int], Dict[int, str]]:
    unique = sorted(set(str(label) for label in labels))
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def make_text_dataloader(
    df: pd.DataFrame,
    tokenizer,
    text_col: str,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ThreadTextDataset(df, text_col=text_col)
    collator = BatchTokenizerCollator(tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def make_comment_dataloader(
    df: pd.DataFrame,
    tokenizer,
    max_comments: int,
    max_comment_length: int,
    batch_size: int,
    shuffle: bool,
    use_ack_masked: bool = True,
) -> DataLoader:
    dataset = CommentThreadDataset(df, use_ack_masked=use_ack_masked)
    collator = CommentThreadCollator(
        tokenizer=tokenizer,
        max_comments=max_comments,
        max_comment_length=max_comment_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if training:
            optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs["loss"]
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += float(loss.item())
        preds = outputs["logits"].argmax(dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["labels"].detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float = 0.01,
    device: torch.device | None = None,
) -> tuple[nn.Module, Dict[str, List[Dict[str, float]]]]:
    device = device or _infer_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    history: Dict[str, List[Dict[str, float]]] = {"train": [], "val": []}
    best_state = None
    best_val_f1 = float("-inf")

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> Dict[str, float]:
    device = device or _infer_device()
    model = model.to(device)
    return run_epoch(model, loader, device, optimizer=None)


def build_fresh_encoder_bundle(base_model_name: str, local_files_only: bool = False) -> EncoderBundle:
    class FreshTransformerEncoder(nn.Module):
        def __init__(self, model_name: str, local_only: bool):
            super().__init__()
            self.transformer = AutoModel.from_pretrained(model_name, local_files_only=local_only)
            self.hidden_size = self.transformer.config.hidden_size

        def encode_features(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
        ) -> Dict[str, torch.Tensor]:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled = outputs.last_hidden_state[:, 0, :]
            return {
                "pooled_output": pooled,
                "discourse_embedding": torch.zeros(
                    pooled.shape[0], 0, device=pooled.device, dtype=pooled.dtype
                ),
                "latent": pooled,
            }

    model = FreshTransformerEncoder(base_model_name, local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, local_files_only=local_files_only)
    return EncoderBundle(
        model=model,
        tokenizer=tokenizer,
        feature_dim=model.hidden_size,
        discourse_dim=0,
    )


def load_discourse_encoder_bundle(
    checkpoint_path: str | Path,
    base_model_name: str,
    local_files_only: bool = False,
) -> EncoderBundle:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_base_model_name = checkpoint.get("bert_model_name", base_model_name)
    label_to_id = checkpoint.get("label_to_id", checkpoint.get("label2id"))
    if label_to_id is None:
        raise KeyError("Checkpoint is missing `label_to_id`/`label2id` metadata.")
    model = DiscourseEncoderModel(
        base_model_name=checkpoint_base_model_name,
        num_labels=len(label_to_id),
        class_embedding_dim=int(checkpoint["class_embedding_dim"]),
        local_files_only=local_files_only,
        init_from_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer_name = checkpoint.get("tokenizer_name", checkpoint_base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
    return EncoderBundle(
        model=model,
        tokenizer=tokenizer,
        feature_dim=model.hidden_size,
        discourse_dim=model.class_embedding_dim,
    )


def save_checkpoint(path: str | Path, payload: Dict[str, object]) -> Path:
    path = Path(path)
    ensure_parent(path)
    torch.save(payload, path)
    return path


def load_winning_arguments_splits(path: str | Path) -> Dict[str, pd.DataFrame]:
    df = read_parquet(Path(path))
    return {
        split: df[df["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
