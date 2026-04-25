"""
BERT-based Discourse Act Classifier
Labels: announcement, elaboration, humor, appreciation, question,
        answer, agreement, negativereaction, disagreement, other
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from huggingface_hub import hf_hub_download
# ─────────────────────────────────────────────
# 1. Label definitions
# ─────────────────────────────────────────────

LABELS = [
    "announcement",
    "elaboration",
    "humor",
    "appreciation",
    "question",
    "answer",
    "agreement",
    "negativereaction",
    "disagreement",
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
        BERT [CLS] token → Dropout → Linear(768 → 256) → GELU → Dropout → Linear(256 → 10)
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        num_labels: int = NUM_LABELS,
        dropout_prob: float = 0.3,
        freeze_bert_layers: int = 0,   # 0 = fine-tune all; N = freeze first N encoder layers
        class_embedding_dim: int = 64, # size of the learned per-label "discourse act" embedding
    ):
        super().__init__()
        self.num_labels = num_labels
        self.class_embedding_dim = class_embedding_dim

        # BERT backbone
        self.bert = BertModel.from_pretrained(bert_model_name)

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        self.hidden_size = hidden_size

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

        # Learned embedding for each discourse-act label.
        # The soft (probability-weighted) lookup of this table acts as the
        # "classification embedding" that gets concatenated with the comment
        # embedding to form a discourse-aware latent representation.
        self.class_embeddings = nn.Embedding(num_labels, class_embedding_dim)
        nn.init.normal_(self.class_embeddings.weight, mean=0.0, std=0.02)

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the fused (comment + classification) representation."""
        return self.hidden_size + self.class_embedding_dim

    def _freeze_layers(self, n: int) -> None:
        """Freeze embeddings + first n encoder layers."""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def _class_embedding_from_logits(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_gold_when_available: bool = False,
    ) -> torch.Tensor:
        """
        Build a per-example classification embedding from logits.

        - During training (or when explicitly requested), if gold labels are
          provided we look up the embedding of the true label. This gives a
          clean teacher-forced signal so the class embedding table actually
          carries discourse-act semantics.
        - Otherwise we use a soft, probability-weighted mixture of class
          embeddings. This is fully differentiable and works at inference
          time when no labels are available.
        """
        if use_gold_when_available and labels is not None:
            return self.class_embeddings(labels)  # (B, class_embedding_dim)
        probs = torch.softmax(logits, dim=-1)  # (B, num_labels)
        return probs @ self.class_embeddings.weight  # (B, class_embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, L)
        attention_mask: torch.Tensor,   # (B, L)
        token_type_ids: Optional[torch.Tensor] = None,  # (B, L)
        labels: Optional[torch.Tensor] = None,           # (B,)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
            logits          – (B, num_labels), always present
            cls_embedding   – (B, hidden_size), BERT [CLS] (the "comment embedding")
            class_embedding – (B, class_embedding_dim), discourse-act embedding
            latent          – (B, hidden_size + class_embedding_dim),
                              concatenation of cls_embedding and class_embedding;
                              this is the discourse-aware comment representation
                              used for the downstream Winning Arguments task
            loss            – scalar, only when labels are provided
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Use [CLS] token representation as the comment embedding
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)

        logits = self.classifier(cls_output)  # (B, num_labels)

        # Use gold labels (when training) so the class-embedding table learns
        # crisp per-discourse-act semantics; fall back to soft mixture at eval.
        class_embedding = self._class_embedding_from_logits(
            logits,
            labels=labels,
            use_gold_when_available=self.training,
        )

        # Discourse-aware comment representation: comment + classification
        latent = torch.cat([cls_output, class_embedding], dim=-1)

        result = {
            "logits": logits,
            "cls_embedding": cls_output,
            "class_embedding": class_embedding,
            "latent": latent,
        }

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

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract the discourse-aware latent representation for a batch of
        comments. Intended for use as a frozen feature extractor when training
        a downstream model on the Winning Arguments dataset.

        Returns dict with:
            latent          – (B, hidden_size + class_embedding_dim)
            cls_embedding   – (B, hidden_size)
            class_embedding – (B, class_embedding_dim)
            probabilities   – (B, num_labels)
            predicted_ids   – (B,)
        """
        self.eval()
        out = self.forward(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(out["logits"], dim=-1)
        return {
            "latent":          out["latent"],
            "cls_embedding":   out["cls_embedding"],
            "class_embedding": out["class_embedding"],
            "probabilities":   probs,
            "predicted_ids":   probs.argmax(dim=-1),
        }


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

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Returns a list of dicts:
            {"label": str, "confidence": float, "probabilities": {label: prob}}
        """
        enc = self._tokenize(texts)

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

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        Encode a list of comments into discourse-aware latent representations.

        This is the entry point for downstream use on the Winning Arguments
        dataset: each comment is mapped to a vector that fuses its BERT
        comment embedding with the learned discourse-act embedding.

        Returns dict of numpy arrays:
            latent          – (N, hidden_size + class_embedding_dim)
            cls_embedding   – (N, hidden_size)
            class_embedding – (N, class_embedding_dim)
            probabilities   – (N, num_labels)
            predicted_ids   – (N,)
            predicted_label – list[str] of length N
        """
        self.model.eval()

        latents, cls_embs, class_embs, all_probs, all_pred_ids = [], [], [], [], []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = self._tokenize(batch_texts)
            out = self.model.encode(
                enc["input_ids"],
                enc["attention_mask"],
                enc.get("token_type_ids"),
            )
            latents.append(out["latent"].cpu().numpy())
            cls_embs.append(out["cls_embedding"].cpu().numpy())
            class_embs.append(out["class_embedding"].cpu().numpy())
            all_probs.append(out["probabilities"].cpu().numpy())
            all_pred_ids.append(out["predicted_ids"].cpu().numpy())

        if not latents:
            empty = np.zeros((0, self.model.latent_dim), dtype=np.float32)
            return {
                "latent":          empty,
                "cls_embedding":   np.zeros((0, self.model.hidden_size), dtype=np.float32),
                "class_embedding": np.zeros((0, self.model.class_embedding_dim), dtype=np.float32),
                "probabilities":   np.zeros((0, NUM_LABELS), dtype=np.float32),
                "predicted_ids":   np.zeros((0,), dtype=np.int64),
                "predicted_label": [],
            }

        predicted_ids = np.concatenate(all_pred_ids, axis=0)
        return {
            "latent":          np.concatenate(latents, axis=0),
            "cls_embedding":   np.concatenate(cls_embs, axis=0),
            "class_embedding": np.concatenate(class_embs, axis=0),
            "probabilities":   np.concatenate(all_probs, axis=0),
            "predicted_ids":   predicted_ids,
            "predicted_label": [ID2LABEL[int(i)] for i in predicted_ids],
        }


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

    HF_DATASET_ID = "Vijayrathank/reddit_discourse_cleaned"
    parquet_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    discourse_df = pd.read_parquet(parquet_path)

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
        "class_embedding_dim":  model.class_embedding_dim,
        "hidden_size":          model.hidden_size,
        "latent_dim":           model.latent_dim,
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

    # ── Latent extraction demo (used for the Winning Arguments task) ─────
    encoded = predictor.encode(test_texts)
    print("\n── Discourse-aware latent representation ──")
    print(f"  latent shape:          {encoded['latent'].shape}        "
          f"(= {model.hidden_size} comment + {model.class_embedding_dim} classification)")
    print(f"  cls_embedding shape:   {encoded['cls_embedding'].shape}")
    print(f"  class_embedding shape: {encoded['class_embedding'].shape}")


if __name__ == "__main__":
    main()