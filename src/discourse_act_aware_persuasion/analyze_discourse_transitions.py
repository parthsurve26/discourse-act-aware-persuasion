from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .io import read_parquet
from .neural import _infer_device, _parse_comment_texts, load_discourse_encoder_bundle
from .paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from .utils import ensure_parent, seed_everything, setup_logging, write_json

DEFAULT_DISCOURSE_CHECKPOINT = MODELS_DIR / "external" / "model.pt"
LOGGER = logging.getLogger(__name__)


def _load_discourse_labels(checkpoint_path: Path) -> List[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_to_id = checkpoint.get("label_to_id", checkpoint.get("label2id"))
    if label_to_id is None:
        raise KeyError("Checkpoint is missing `label_to_id`/`label2id` metadata.")
    return [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]


def _prepare_threads(
    df: pd.DataFrame,
    use_ack_masked: bool,
    max_comments: int | None,
) -> Tuple[pd.DataFrame, List[str]]:
    thread_rows: List[Dict[str, object]] = []
    flat_comments: List[str] = []

    for row in df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        comments = _parse_comment_texts(row_series, use_ack_masked=use_ack_masked)
        if max_comments is not None and max_comments > 0:
            comments = comments[:max_comments]
        if len(comments) < 2:
            continue

        start_idx = len(flat_comments)
        flat_comments.extend(comments)
        end_idx = len(flat_comments)
        thread_rows.append(
            {
                "pair_id": row_series.get("pair_id"),
                "argument_thread_id": row_series.get("argument_thread_id"),
                "split": row_series.get("split"),
                "label": int(row_series["label"]),
                "comment_start": start_idx,
                "comment_end": end_idx,
                "num_comments_used": end_idx - start_idx,
            }
        )

    if not thread_rows:
        return pd.DataFrame(), []

    return pd.DataFrame(thread_rows), flat_comments


def _predict_discourse_labels(
    comments: Sequence[str],
    checkpoint_path: Path,
    base_model_name: str,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
) -> Tuple[List[int], List[float], List[str]]:
    labels = _load_discourse_labels(checkpoint_path)
    bundle = load_discourse_encoder_bundle(
        checkpoint_path=checkpoint_path,
        base_model_name=base_model_name,
        local_files_only=local_files_only,
    )
    device = _infer_device()
    model = bundle.model.to(device)
    model.eval()

    predicted_ids: List[int] = []
    confidences: List[float] = []

    with torch.no_grad():
        for start in range(0, len(comments), batch_size):
            batch_comments = list(comments[start : start + batch_size])
            encoded = bundle.tokenizer(
                batch_comments,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model.encode_features(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids"),
            )
            probabilities = outputs["probabilities"].detach().cpu()
            predicted_ids.extend(probabilities.argmax(dim=-1).tolist())
            confidences.extend(probabilities.max(dim=-1).values.tolist())

            if (start // batch_size) % 25 == 0:
                LOGGER.info(
                    "Predicted discourse labels for %s / %s comments",
                    min(start + len(batch_comments), len(comments)),
                    len(comments),
                )

    return predicted_ids, confidences, labels


def _transition_name(source: str, target: str) -> str:
    return f"{source} -> {target}"


def _count_labels(labels: Iterable[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    counter.update(labels)
    return counter


def _build_transition_outputs(
    threads: pd.DataFrame,
    predicted_ids: Sequence[int],
    confidences: Sequence[float],
    discourse_labels: Sequence[str],
    min_occurrences: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    positive_occurrences: Counter[str] = Counter()
    negative_occurrences: Counter[str] = Counter()
    positive_threads: Counter[str] = Counter()
    negative_threads: Counter[str] = Counter()
    positive_acts: Counter[str] = Counter()
    negative_acts: Counter[str] = Counter()
    thread_records: List[Dict[str, object]] = []

    for row in threads.itertuples(index=False):
        label_ids = predicted_ids[row.comment_start : row.comment_end]
        label_names = [discourse_labels[label_id] for label_id in label_ids]
        comment_confidences = confidences[row.comment_start : row.comment_end]
        transitions = [
            _transition_name(source, target)
            for source, target in zip(label_names, label_names[1:])
        ]
        transition_counter = Counter(transitions)
        distinct_transitions = set(transitions)
        label_counter = _count_labels(label_names)

        record = {
            "pair_id": row.pair_id,
            "argument_thread_id": row.argument_thread_id,
            "split": row.split,
            "label": int(row.label),
            "num_comments_used": int(row.num_comments_used),
            "num_transitions": len(transitions),
            "mean_discourse_confidence": float(np.mean(comment_confidences)) if comment_confidences else None,
        }
        thread_records.append(record)

        if int(row.label) == 1:
            positive_occurrences.update(transition_counter)
            positive_threads.update(distinct_transitions)
            positive_acts.update(label_counter)
        else:
            negative_occurrences.update(transition_counter)
            negative_threads.update(distinct_transitions)
            negative_acts.update(label_counter)

    transition_vocab = sorted(set(positive_occurrences) | set(negative_occurrences))
    total_positive_occurrences = sum(positive_occurrences.values())
    total_negative_occurrences = sum(negative_occurrences.values())
    positive_thread_total = int((threads["label"] == 1).sum())
    negative_thread_total = int((threads["label"] == 0).sum())
    vocab_size = max(len(transition_vocab), 1)
    smoothing = 1.0

    transition_rows: List[Dict[str, object]] = []
    for transition in transition_vocab:
        pos_occ = int(positive_occurrences[transition])
        neg_occ = int(negative_occurrences[transition])
        pos_thread = int(positive_threads[transition])
        neg_thread = int(negative_threads[transition])
        total_occ = pos_occ + neg_occ

        pos_occ_freq = pos_occ / total_positive_occurrences if total_positive_occurrences else 0.0
        neg_occ_freq = neg_occ / total_negative_occurrences if total_negative_occurrences else 0.0
        pos_thread_freq = pos_thread / positive_thread_total if positive_thread_total else 0.0
        neg_thread_freq = neg_thread / negative_thread_total if negative_thread_total else 0.0
        log_odds = float(
            np.log((pos_occ + smoothing) / (total_positive_occurrences + smoothing * vocab_size))
            - np.log((neg_occ + smoothing) / (total_negative_occurrences + smoothing * vocab_size))
        )

        transition_rows.append(
            {
                "transition": transition,
                "positive_occurrences": pos_occ,
                "negative_occurrences": neg_occ,
                "total_occurrences": total_occ,
                "positive_occurrence_rate": float(pos_occ_freq),
                "negative_occurrence_rate": float(neg_occ_freq),
                "occurrence_rate_gap": float(pos_occ_freq - neg_occ_freq),
                "positive_threads": pos_thread,
                "negative_threads": neg_thread,
                "positive_thread_rate": float(pos_thread_freq),
                "negative_thread_rate": float(neg_thread_freq),
                "thread_rate_gap": float(pos_thread_freq - neg_thread_freq),
                "log_odds_delta_vs_no_delta": log_odds,
            }
        )

    transition_df = pd.DataFrame(transition_rows).sort_values(
        by=["log_odds_delta_vs_no_delta", "total_occurrences"],
        ascending=[False, False],
    )

    act_rows: List[Dict[str, object]] = []
    for act in discourse_labels:
        pos_count = int(positive_acts[act])
        neg_count = int(negative_acts[act])
        total_count = pos_count + neg_count
        act_rows.append(
            {
                "discourse_act": act,
                "positive_comments": pos_count,
                "negative_comments": neg_count,
                "total_comments": total_count,
                "positive_comment_rate": float(
                    pos_count / sum(positive_acts.values()) if positive_acts else 0.0
                ),
                "negative_comment_rate": float(
                    neg_count / sum(negative_acts.values()) if negative_acts else 0.0
                ),
            }
        )
    act_df = pd.DataFrame(act_rows).sort_values(by="total_comments", ascending=False)

    filtered = transition_df[transition_df["total_occurrences"] >= min_occurrences].copy()
    top_positive = filtered.head(10).to_dict(orient="records")
    top_negative = filtered.sort_values(
        by=["log_odds_delta_vs_no_delta", "total_occurrences"],
        ascending=[True, False],
    ).head(10).to_dict(orient="records")

    summary = {
        "threads_analyzed": int(len(threads)),
        "positive_threads": positive_thread_total,
        "negative_threads": negative_thread_total,
        "comments_scored": int(len(predicted_ids)),
        "average_comments_per_thread": float(threads["num_comments_used"].mean()) if len(threads) else 0.0,
        "average_discourse_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "transition_vocab_size": int(len(transition_vocab)),
        "total_positive_transition_occurrences": int(total_positive_occurrences),
        "total_negative_transition_occurrences": int(total_negative_occurrences),
        "min_occurrences_for_ranked_lists": int(min_occurrences),
        "top_positive_transitions": top_positive,
        "top_negative_transitions": top_negative,
        "thread_records_preview": thread_records[:5],
    }
    return transition_df, act_df, summary


def analyze_discourse_transitions(
    winning_arguments_path: Path,
    discourse_checkpoint: Path,
    output_dir: Path,
    split: str,
    max_comments: int | None,
    max_comment_length: int,
    batch_size: int,
    use_ack_masked: bool,
    base_model_name: str,
    local_files_only: bool,
    min_occurrences: int,
    limit: int | None,
) -> Dict[str, object]:
    df = read_parquet(winning_arguments_path)
    if split != "all":
        df = df[df["split"] == split].reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)

    LOGGER.info("Loaded %s rows from %s for split=%s", len(df), winning_arguments_path, split)
    threads, flat_comments = _prepare_threads(
        df=df,
        use_ack_masked=use_ack_masked,
        max_comments=max_comments,
    )
    if threads.empty:
        raise ValueError("No threads with at least two usable comments were found.")

    LOGGER.info(
        "Prepared %s threads and %s comments for discourse-transition analysis",
        len(threads),
        len(flat_comments),
    )
    predicted_ids, confidences, discourse_labels = _predict_discourse_labels(
        comments=flat_comments,
        checkpoint_path=discourse_checkpoint,
        base_model_name=base_model_name,
        batch_size=batch_size,
        max_length=max_comment_length,
        local_files_only=local_files_only,
    )
    transition_df, act_df, summary = _build_transition_outputs(
        threads=threads,
        predicted_ids=predicted_ids,
        confidences=confidences,
        discourse_labels=discourse_labels,
        min_occurrences=min_occurrences,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_suffix = split if split != "all" else "all_splits"
    transition_path = output_dir / f"discourse_transition_stats_{split_suffix}.csv"
    act_path = output_dir / f"discourse_act_stats_{split_suffix}.csv"
    summary_path = output_dir / f"discourse_transition_summary_{split_suffix}.json"

    ensure_parent(transition_path)
    transition_df.to_csv(transition_path, index=False)
    act_df.to_csv(act_path, index=False)
    summary_payload = {
        "input_path": str(winning_arguments_path),
        "checkpoint_path": str(discourse_checkpoint),
        "split": split,
        "max_comments": max_comments,
        "max_comment_length": max_comment_length,
        "batch_size": batch_size,
        "use_ack_masked": use_ack_masked,
        "base_model_name": base_model_name,
        "discourse_labels": list(discourse_labels),
        "summary": summary,
        "transition_stats_path": str(transition_path),
        "act_stats_path": str(act_path),
    }
    write_json(summary_path, summary_payload)

    LOGGER.info("Saved transition stats to %s", transition_path)
    LOGGER.info("Saved discourse-act stats to %s", act_path)
    LOGGER.info("Saved summary to %s", summary_path)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze adjacent discourse-act transitions in Winning Arguments threads."
    )
    parser.add_argument(
        "--winning-arguments-path",
        type=Path,
        default=PROCESSED_DIR / "winning_arguments.parquet",
    )
    parser.add_argument(
        "--discourse-checkpoint",
        type=Path,
        default=DEFAULT_DISCOURSE_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which Winning Arguments split to analyze.",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=16,
        help="Maximum comments per thread to analyze. Use 0 to keep all comments.",
    )
    parser.add_argument("--max-comment-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--use-ack-masked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop delta-acknowledgement comments when comment-level masks are available.",
    )
    parser.add_argument("--base-model", default="bert-base-uncased")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load the tokenizer/base model from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=3,
        help="Minimum transition count for the ranked summary lists in the JSON output.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick debugging.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)
    analyze_discourse_transitions(
        winning_arguments_path=args.winning_arguments_path,
        discourse_checkpoint=args.discourse_checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        max_comments=args.max_comments if args.max_comments > 0 else None,
        max_comment_length=args.max_comment_length,
        batch_size=args.batch_size,
        use_ack_masked=args.use_ack_masked,
        base_model_name=args.base_model,
        local_files_only=args.local_files_only,
        min_occurrences=args.min_occurrences,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
