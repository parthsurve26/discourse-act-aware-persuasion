"""
BERT-based Discourse Act Classifier
Labels: question, answer, announcement, agreement, disagreement,
        appreciation, elaboration, humor, other
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
# ─────────────────────────────────────────────
# 1. Label definitions
# ─────────────────────────────────────────────

LABELS = [
    "question",
    "answer",
    "announcement",
    "agreement",
    "disagreement",
    "appreciation",
    "elaboration",
    "humor",
    "other",
]
NUM_LABELS = len(LABELS)
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


# ─────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────

class BertDiscourseClassifier(nn.Module):
    """
    BERT encoder + classification head for discourse-act classification.

    Architecture:
        BERT [CLS] token → Dropout → Linear(768 → 256) → GELU → Dropout → Linear(256 → 9)
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_labels: int = NUM_LABELS,
        dropout_prob: float = 0.3,
        freeze_bert_layers: int = 0,   # 0 = fine-tune all; N = freeze first N encoder layers
    ):
        super().__init__()
        self.num_labels = num_labels

        # BERT backbone
        self.bert = BertModel.from_pretrained(bert_model_name)

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # Optionally freeze early BERT layers to speed up training
        if freeze_bert_layers > 0:
            self._freeze_layers(freeze_bert_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_labels),
        )

    def _freeze_layers(self, n: int) -> None:
        """Freeze embeddings + first n encoder layers."""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        token_type_ids: Optional[torch.Tensor] = None,  # (B, L)
        labels: Optional[torch.Tensor] = None,           # (B,)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
            logits  – (B, num_labels), always present
            loss    – scalar, only when labels are provided
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        logits = self.classifier(cls_output)  # (B, num_labels)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            predicted_ids  – (B,)  integer class indices
            probabilities  – (B, num_labels)  softmax probabilities
        """
        self.eval()
        out = self.forward(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(out["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────

class DiscourseDataset(Dataset):
    """
    Expects a list of (text, label_string) pairs.

    Example
    -------
    data = [
        ("Great point, I totally agree!", "agreement"),
        ("What time does the meeting start?", "question"),
    ]
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 128,
    ):
        assert len(texts) == len(labels), "texts and labels must have the same length"
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            [LABEL2ID[lbl] for lbl in labels], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings["token_type_ids"][idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────
# 4. Trainer
# ─────────────────────────────────────────────

class Trainer:
    """Minimal training loop with accuracy tracking."""

    def __init__(
        self,
        model: BertDiscourseClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def _batch_to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            self.optimizer.zero_grad()

            out = self.model(**batch)
            out["loss"].backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += out["loss"].item()
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

        return {"loss": total_loss / len(loader), "train_accuracy": correct / total}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            out = self.model(**batch)

            total_loss += out["loss"].item()
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

        return {"loss": total_loss / len(loader), "test_accuracy": correct / total}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
    ) -> List[Dict]:
        history = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                row.update({f"val_{k}": v for k, v in val_metrics.items()})

            history.append(row)
            self._log(row)

        return history

    @staticmethod
    def _log(row: Dict) -> None:
        parts = [f"Epoch {row['epoch']:>2}"]
        for k, v in row.items():
            if k == "epoch":
                continue
            parts.append(f"{k}: {v:.4f}")
        print("  |  ".join(parts))


# ─────────────────────────────────────────────
# 5. Inference helper
# ─────────────────────────────────────────────

class DiscoursePredictor:
    """Single-text and batch inference interface."""

    def __init__(
        self,
        model: BertDiscourseClassifier,
        tokenizer: BertTokenizer,
        device: torch.device,
        max_length: int = 128,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Returns a list of dicts:
            {"label": str, "confidence": float, "probabilities": {label: prob}}
        """
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        pred_ids, probs = self.model.predict(
            enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]
        )

        results = []
        for i, (pid, prob_vec) in enumerate(zip(pred_ids.cpu(), probs.cpu())):
            prob_dict = {ID2LABEL[j]: round(prob_vec[j].item(), 4) for j in range(NUM_LABELS)}
            results.append({
                "text":          texts[i],
                "label":         ID2LABEL[pid.item()],
                "confidence":    round(prob_vec[pid].item(), 4),
                "probabilities": prob_dict,
            })
        return results


# ─────────────────────────────────────────────
# 6. Quick-start demo
# ─────────────────────────────────────────────

def main():
    BERT_MODEL  = "bert-base-uncased"
    MAX_LEN     = 128
    BATCH_SIZE  = 16
    EPOCHS      = 3
    LR          = 2e-5
    FREEZE_BERT = 6          # freeze bottom 6 of 12 encoder layers
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ────────────────────────────


    TRAIN_FRAC = 0.85

    discourse_df = pd.read_parquet("../data/processed/coarse_discourse.parquet")

    rng = np.random.default_rng(42)
    shuffled_idx = rng.permutation(len(discourse_df))
    discourse_df = discourse_df.iloc[shuffled_idx].reset_index(drop=True)

    split_at = int(len(discourse_df) * TRAIN_FRAC)
    train_df = discourse_df.iloc[:split_at]
    val_df   = discourse_df.iloc[split_at:]

    train_texts  = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    val_texts    = val_df['text'].tolist()
    val_labels   = val_df['label'].tolist()

    print(f"Train size: {len(train_texts)} | Val size: {len(val_texts)}")

    # ────────────────────────────────────────────────────────────────────────

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    train_ds  = DiscourseDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds    = DiscourseDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)
    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = BertDiscourseClassifier(
        bert_model_name=BERT_MODEL,
        num_labels=NUM_LABELS,
        dropout_prob=0.3,
        freeze_bert_layers=FREEZE_BERT,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    # Linear warmup + decay scheduler (optional but recommended)

    total_steps   = len(train_dl) * EPOCHS
    warmup_steps  = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    trainer = Trainer(model, optimizer, device, scheduler)
    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS)

    # ── Save checkpoint ──────────────────────────────────────────────────
    torch.save({
        "model_state_dict":     model.state_dict(),
        "label2id":             LABEL2ID,
        "id2label":             ID2LABEL,
        "bert_model_name":      BERT_MODEL,
    }, "bert_discourse_classifier.pt")
    print("\nCheckpoint saved to bert_discourse_classifier.pt")

    # ── Inference demo ───────────────────────────────────────────────────
    predictor = DiscoursePredictor(model, tokenizer, device)
    test_texts = [
        "Could you share the updated slides?",
        "Haha that error message is hilarious.",
        "I strongly disagree with this approach.",
    ]
    print("\n── Inference demo ──")
    for result in predictor.predict(test_texts):
        print(f"  [{result['label']:>14}]  ({result['confidence']:.2%})  \"{result['text']}\"")


if __name__ == "__main__":
    main()