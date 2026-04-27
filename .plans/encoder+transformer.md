# Plan: Transition-Aware Self-Attention Transformer for Winning Arguments

## Context

The Winning Arguments task (CMV) is currently served only by a TF-IDF + Logistic Regression baseline ([train_baselines.py](src/discourse_act_aware_persuasion/train_baselines.py)). We want a neural model that exploits the *thread* structure of each argument: a sequence of comments alternating between OP and challenger, where the persuasive trajectory — not just bag-of-words content — drives the win/lose label.

The model is a **hierarchical transition-aware self-attention transformer**: each comment is encoded into a single vector, and a comment-level transformer attends across the thread with attention biases that encode the *transition* between comments (speaker change, turn distance, and — for variant 2 — the discourse-act pair). Two variants are built so we can isolate the contribution of the discourse-classifier latents:

- **Variant 1 (standalone)** — no dependency on `coarse_discourse_cls`. Per-comment encoder is a frozen fresh `bert-base-uncased`. Transition bias uses speaker + relative turn distance only.
- **Variant 2 (with discourse latents)** — per-comment encoder is the frozen `BertDiscourseClassifier` pulled from `Vijayrathank/discourse_act_classifier` on HF Hub; each comment becomes an 832-dim latent (768 BERT [CLS] + 64 class embedding) plus a predicted discourse-act ID. Transition bias adds a learned (act_i, act_j) transition embedding on top of the speaker/turn bias.

## Data understanding (verified)

Source: [data/processed/winning_arguments.parquet](data/processed/winning_arguments.parquet), produced by [winning_arguments.py](src/discourse_act_aware_persuasion/data/winning_arguments.py). 8,526 rows, balanced 4,263/4,263, with a precomputed `split` column (train/val/test) from [`assign_holdout_split`](src/discourse_act_aware_persuasion/splits.py).

Each row is one **argument thread** = the comments authored by one debater in one top-level reply chain of a CMV pair. Per-comment fields are JSON-serialized arrays, parsed at load time:

- `comment_texts` — `List[str]`, raw comment bodies (use the `_ack_masked` variant — see below)
- `comment_speakers` — `List[str]`, speaker user IDs
- `comment_is_op` — `List[bool]`, whether each comment is from OP
- `comment_is_delta_ack` — `List[bool]`, delta acknowledgements
- `label` — 0 (losing) or 1 (winning)

To prevent label leakage from delta tokens (∆/!delta), we use the **masked** comment texts. The dataset stores `text_ack_masked` and `input_text_ack_masked` at the thread level, but per-comment masked arrays are not separately stored — we will reconstruct them from `comment_texts` + `comment_is_delta_ack` by stripping `DELTA_TOKEN_RE` (already defined in [winning_arguments.py:13](src/discourse_act_aware_persuasion/data/winning_arguments.py#L13)) from any comment flagged as a delta-ack. This keeps preprocessing in one place and is reused by both variants.

## Shared architecture

```
comment_texts (≤ N comments)
  │
  ├─► per-comment encoder (frozen)              # variant-specific
  │     variant 1: BERT [CLS]      → 768-d
  │     variant 2: BertDiscourseClassifier.encode()
  │                                 → latent (832-d) + predicted act id (∈ 0..9)
  │
  ▼
project to d_model (e.g. 256)        Linear(768→256) or Linear(832→256)
  +  speaker embedding (is_op)       Embedding(2, d_model)
  +  positional embedding (turn idx) Embedding(N_max, d_model)
  +  [CLS_thread] token              learned vector prepended
  ▼
Transition-aware self-attention transformer (L layers, e.g. 4 layers, 8 heads)
  attn_logits[i,j] = QKᵀ/√d  +  B[i,j]
  B[i,j] is a sum of biases, each a scalar produced by a small MLP / learned table:
    • speaker-pair bias            table[speaker_i, speaker_j]  (2×2)
    • turn-distance bias           bucketed |i−j|  (e.g. 8 buckets)
    • [variant 2 only] discourse-act-pair bias  table[act_i, act_j]  (10×10)
  ▼
[CLS_thread] output  →  Linear(d_model → 2)  →  CrossEntropy
```

Padding masks handle variable thread lengths. Attention bias is broadcast to all heads (or per-head if we add a head dim to the bias tables — keep it shared per head for simplicity).

### Why this satisfies "transition-aware"

Standard self-attention is permutation-invariant up to position embeddings. By injecting a *learned* additive bias indexed by the (speaker, speaker), (turn-bucket), and (act, act) pair of every (i, j) attention edge, the model learns that, e.g., a `question → answer` transition between OP and challenger should attend strongly, while a `disagreement → disagreement` self-loop on the same speaker is downweighted. This is the same mechanism used in T5's relative-position bias and ALiBi, generalised to discourse-act and speaker pairs.

## File layout

All new code under [src/discourse_act_aware_persuasion/winning_arguments/](src/discourse_act_aware_persuasion/winning_arguments/):

- `data.py` — `WinningArgumentsDataset` (parses JSON arrays, applies delta-mask, batches with padding); `collate_fn`. Reads from the parquet path returned by [paths.py](src/discourse_act_aware_persuasion/paths.py).
- `encoders.py` — two thin wrappers:
  - `FrozenBertCommentEncoder` (variant 1): `bert-base-uncased`, returns [CLS] per comment, frozen, batched across all comments in a minibatch.
  - `FrozenDiscourseEncoder` (variant 2): downloads weights from HF Hub `Vijayrathank/discourse_act_classifier` via `huggingface_hub.hf_hub_download`, instantiates `BertDiscourseClassifier` from [coarse_discourse_cls/model.py](coarse_discourse_cls/model.py), loads `model_state_dict`, calls `.encode()` per batch — returns `(latent, predicted_ids)`. Frozen.
- `model.py` — `TransitionAwareThreadTransformer` (the comment-level transformer described above) with a flag `use_discourse_acts: bool` that toggles the act-pair bias table and the input projection dim (768 vs 832). Custom attention layer that accepts a `(B, N, N)` additive bias tensor.
- `trainer.py` — train/eval loops mirroring the style of [coarse_discourse_cls/model.py:275-359](coarse_discourse_cls/model.py#L275-L359) (`Trainer.fit`, `train_epoch`, `evaluate`). Tracks loss, accuracy, F1, AUC.
- `infer.py` — `WinningArgumentsPredictor` exposing `predict(thread_dict) -> {label, prob}`.
- `__init__.py` — re-exports.

Two CLI entrypoints under [scripts/](scripts/):

- `scripts/train_winning_args_standalone.py` — variant 1 (frozen fresh BERT + speaker/turn bias).
- `scripts/train_winning_args_with_discourse.py` — variant 2 (frozen discourse encoder + act-pair bias).

Both scripts share the same module code; they differ only in which encoder they instantiate and whether `use_discourse_acts=True`. Each script supports `--mode {train,eval,predict}` so a single file covers training, eval-on-test, and inference on user-provided text.

### Critical files to read/modify

- New: all files under [src/discourse_act_aware_persuasion/winning_arguments/](src/discourse_act_aware_persuasion/winning_arguments/) and the two scripts above.
- Read-only reuse:
  - [coarse_discourse_cls/model.py](coarse_discourse_cls/model.py) — `BertDiscourseClassifier`, `LABELS`, `LABEL2ID`, `ID2LABEL`, `.encode()` API.
  - [src/discourse_act_aware_persuasion/data/winning_arguments.py:13](src/discourse_act_aware_persuasion/data/winning_arguments.py#L13) — `DELTA_TOKEN_RE` for the delta-mask.
  - [src/discourse_act_aware_persuasion/paths.py](src/discourse_act_aware_persuasion/paths.py) — for the parquet path.
- No edits required to existing files.

## Hyperparameters (defaults, configurable via CLI)

- d_model = 256, n_heads = 8, n_layers = 4, ffn_dim = 1024, dropout = 0.1
- Per-comment max tokens = 128 (matches discourse classifier)
- Max comments per thread = 32 (truncate from the start; thread tails contain the persuasive turn)
- Batch size = 8 threads, grad accumulation if memory tight (variant 2 holds BERT in eval mode)
- Optimizer = AdamW, lr = 3e-4 on the new transformer (encoders are frozen so no separate lr group), weight_decay = 0.01
- Linear warmup 10%, cosine decay, epochs = 10, early-stop on val AUC

## Verification

1. **Smoke**: run each script with `--limit 64 --epochs 1` to confirm forward/backward, padding mask correctness, HF Hub download succeeds, no NaNs.
2. **Sanity vs baseline**: full training run; both variants should clear the TF-IDF + LR baseline from `train_baselines.py` on the test split (val AUC > baseline AUC). Variant 2 should beat variant 1 if discourse acts add signal — that comparison is the headline experiment.
3. **Leakage check**: confirm delta-ack comments are masked by spot-checking 10 threads where `any(comment_is_delta_ack)` is True; the masked input must contain no ∆/!delta substrings. Add a unit test in `tests/test_winning_args_data.py` (one assertion).
4. **Inference**: `python scripts/train_winning_args_with_discourse.py --mode predict --thread-json examples/sample_thread.json` returns `{label, prob}` on a held-out thread; eyeball that confidence is sane.
5. **Reproducibility**: fix seeds; rerun training twice and confirm test accuracy within ±1 pp.
