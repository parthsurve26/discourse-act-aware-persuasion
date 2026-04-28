from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from ..splits import assign_holdout_split


DELTA_TOKEN_RE = re.compile(r"(?:∆|Δ|!delta|&#8710;|&#916;)")


def _conversation_text(convo) -> str:
    title = convo.meta.get("op-title", "") or ""
    body = convo.meta.get("op-text-body", "") or ""
    return f"{title}\n\n{body}".strip()


def _clean_text(text: object) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if text.lower() == "none":
        return ""
    return text


def _is_delta_ack(comment_text: str, comment_speaker_id: str, op_user_id: str) -> bool:
    if not op_user_id or comment_speaker_id != op_user_id:
        return False
    return bool(DELTA_TOKEN_RE.search(comment_text or ""))


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


def build_winning_arguments_dataframe(corpus, seed: int = 42) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for convo in corpus.iter_conversations():
        op_user = convo.meta.get("op-userID")
        train_flag = int(convo.meta.get("train", 0) or 0)
        utterances = list(convo.iter_utterances())
        utterances_by_id = {utt.id: utt for utt in utterances}
        grouped: Dict[tuple[str, str], List[Any]] = defaultdict(list)

        for utt in utterances:
            success = utt.meta.get("success")
            if success not in (0, 1, "0", "1"):
                continue
            for pair_id in utt.meta.get("pair_ids", []) or []:
                argument_thread_id = _top_level_reply_id(utt, utterances_by_id)
                grouped[(pair_id, argument_thread_id)].append(utt)

        for (pair_id, argument_thread_id), group_utts in sorted(grouped.items()):
            labels = {int(utt.meta["success"]) for utt in group_utts}
            if len(labels) != 1:
                continue
            label = labels.pop()
            group_utts = sorted(
                group_utts,
                key=lambda utt: (
                    utt.timestamp if utt.timestamp is not None else -1,
                    utt.id,
                ),
            )
            raw = [(utt, _clean_text(utt.text)) for utt in group_utts]
            kept = [(utt, text) for utt, text in raw if text]
            if not kept:
                continue

            comment_ids = [utt.id for utt, _ in kept]
            comment_texts = [text for _, text in kept]
            comment_speakers = [
                getattr(getattr(utt, "speaker", None), "id", "") for utt, _ in kept
            ]
            comment_is_op = [
                bool(op_user) and speaker_id == op_user for speaker_id in comment_speakers
            ]
            comment_is_delta_ack = [
                _is_delta_ack(text, speaker_id, op_user)
                for text, speaker_id in zip(comment_texts, comment_speakers)
            ]

            thread_text = "\n\n".join(comment_texts)
            masked_texts = [
                "" if is_ack else text
                for text, is_ack in zip(comment_texts, comment_is_delta_ack)
            ]
            thread_text_masked = "\n\n".join(text for text in masked_texts if text)
            input_text = "\n\n".join(
                [
                    f"[TITLE] {convo.meta.get('op-title', '') or ''}",
                    f"[POST] {convo.meta.get('op-text-body', '') or ''}",
                    f"[ARGUMENT_THREAD] {thread_text}",
                ]
            ).strip()
            input_text_masked = "\n\n".join(
                [
                    f"[TITLE] {convo.meta.get('op-title', '') or ''}",
                    f"[POST] {convo.meta.get('op-text-body', '') or ''}",
                    f"[ARGUMENT_THREAD] {thread_text_masked}",
                ]
            ).strip()
            rows.append(
                {
                    "conversation_id": convo.id,
                    "prediction_unit": "pair_member_argument_thread",
                    "pair_id": pair_id,
                    "argument_thread_id": argument_thread_id,
                    "labeled_utterance_ids": json.dumps([utt.id for utt in group_utts], ensure_ascii=True),
                    "speaker_ids": json.dumps(
                        sorted(
                            {
                                getattr(getattr(utt, "speaker", None), "id", "")
                                for utt in group_utts
                            }
                        ),
                        ensure_ascii=True,
                    ),
                    "comment_ids": json.dumps(comment_ids, ensure_ascii=True),
                    "comment_texts": json.dumps(comment_texts, ensure_ascii=False),
                    "comment_speakers": json.dumps(comment_speakers, ensure_ascii=True),
                    "comment_is_op": json.dumps(comment_is_op),
                    "comment_is_delta_ack": json.dumps(comment_is_delta_ack),
                    "op_user_id": op_user,
                    "op_title": convo.meta.get("op-title", ""),
                    "op_text_body": convo.meta.get("op-text-body", ""),
                    "conversation_pair_ids": json.dumps(
                        convo.meta.get("pair_ids", []), ensure_ascii=True
                    ),
                    "text": thread_text,
                    "input_text": input_text,
                    "text_ack_masked": thread_text_masked,
                    "input_text_ack_masked": input_text_masked,
                    "label": label,
                    "train_flag": train_flag,
                    "num_labeled_utterances": len(group_utts),
                    "num_kept_comments": len(kept),
                    "num_delta_ack_comments": int(sum(comment_is_delta_ack)),
                    "num_chars": len(thread_text),
                    "num_words": len(thread_text.split()),
                    "num_chars_ack_masked": len(thread_text_masked),
                    "num_words_ack_masked": len(thread_text_masked.split()),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "conversation_id",
                "prediction_unit",
                "pair_id",
                "argument_thread_id",
                "labeled_utterance_ids",
                "speaker_ids",
                "comment_ids",
                "comment_texts",
                "comment_speakers",
                "comment_is_op",
                "comment_is_delta_ack",
                "op_user_id",
                "op_title",
                "op_text_body",
                "conversation_pair_ids",
                "text",
                "input_text",
                "text_ack_masked",
                "input_text_ack_masked",
                "label",
                "train_flag",
                "num_labeled_utterances",
                "num_kept_comments",
                "num_delta_ack_comments",
                "num_chars",
                "num_words",
                "num_chars_ack_masked",
                "num_words_ack_masked",
                "split",
            ]
        )
        return df
    return assign_holdout_split(
        df=df,
        group_col="conversation_id",
        train_flag_col="train_flag",
        holdout_val_ratio=0.5,
        seed=seed,
    )
