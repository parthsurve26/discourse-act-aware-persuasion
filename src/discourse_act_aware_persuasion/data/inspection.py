from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List


def _safe_meta_keys(obj) -> List[str]:
    meta = getattr(obj, "meta", {}) or {}
    return sorted(list(meta.keys()))


def _top_level_reply_id(utt, utterances_by_id: Dict[str, Any]) -> str:
    cur = utt
    seen = set()
    while cur.reply_to is not None and cur.reply_to in utterances_by_id:
        if cur.id in seen:
            break
        seen.add(cur.id)
        parent = utterances_by_id[cur.reply_to]
        if parent.id == cur.conversation_id:
            return cur.id
        cur = parent
    return cur.id


def inspect_winning_arguments_corpus(corpus) -> Dict[str, Any]:
    conversations = list(corpus.iter_conversations())
    utterances = list(corpus.iter_utterances())

    labeled_success = Counter()
    conversation_train = Counter()
    conversation_meta_keys = set()
    utterance_meta_keys = set()
    utterance_pair_lengths = Counter()
    labeled_pair_lengths = Counter()
    pair_to_success = defaultdict(Counter)
    pair_to_top_threads = defaultdict(set)
    pair_to_conversation = defaultdict(set)
    prediction_units = defaultdict(list)
    samples: List[Dict[str, Any]] = []
    utterances_by_id = {utt.id: utt for utt in utterances}

    for convo in conversations:
        conversation_meta_keys.update(_safe_meta_keys(convo))
        conversation_train[str(convo.meta.get("train"))] += 1
        if len(samples) < 5:
            convo_utts = list(convo.iter_utterances())
            samples.append(
                {
                    "conversation_id": convo.id,
                    "conversation_meta": dict(convo.meta),
                    "root_text_preview": (convo_utts[0].text if convo_utts else "")[:200],
                }
            )

    for utt in utterances:
        utterance_meta_keys.update(_safe_meta_keys(utt))
        pair_ids = utt.meta.get("pair_ids") or []
        utterance_pair_lengths[len(pair_ids)] += 1
        if utt.meta.get("success") in (0, 1, "0", "1"):
            labeled_success[str(int(utt.meta.get("success")))] += 1
            labeled_pair_lengths[len(pair_ids)] += 1
            top_id = _top_level_reply_id(utt, utterances_by_id)
            for pair_id in pair_ids:
                success = int(utt.meta["success"])
                pair_to_success[pair_id][success] += 1
                pair_to_top_threads[pair_id].add(top_id)
                pair_to_conversation[pair_id].add(utt.conversation_id)
                prediction_units[(pair_id, top_id)].append(success)

    prediction_unit_label_counts = Counter()
    invalid_prediction_units = 0
    for labels in prediction_units.values():
        unique_labels = set(labels)
        if len(unique_labels) != 1:
            invalid_prediction_units += 1
            continue
        prediction_unit_label_counts[str(next(iter(unique_labels)))] += 1

    pair_label_shapes = Counter()
    for counts in pair_to_success.values():
        pair_label_shapes[
            ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))
        ] += 1

    return {
        "num_speakers": len(list(corpus.iter_speakers())),
        "num_utterances": len(utterances),
        "num_conversations": len(conversations),
        "conversation_meta_keys": sorted(conversation_meta_keys),
        "utterance_meta_keys": sorted(utterance_meta_keys),
        "train_flag_counts": dict(conversation_train),
        "success_label_counts": dict(labeled_success),
        "utterance_pair_id_length_counts": dict(utterance_pair_lengths),
        "labeled_utterance_pair_id_length_counts": dict(labeled_pair_lengths),
        "num_pair_ids": len(pair_to_success),
        "pair_top_level_thread_count_counts": dict(
            Counter(len(threads) for threads in pair_to_top_threads.values())
        ),
        "pair_conversation_count_counts": dict(
            Counter(len(convos) for convos in pair_to_conversation.values())
        ),
        "pair_label_shapes_top_20": dict(pair_label_shapes.most_common(20)),
        "recommended_prediction_unit": "one row per (pair_id, top-level reply thread)",
        "prediction_unit_label_counts": dict(prediction_unit_label_counts),
        "invalid_prediction_unit_count": invalid_prediction_units,
        "sample_conversations": samples,
    }
