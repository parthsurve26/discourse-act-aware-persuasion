from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from ..splits import assign_holdout_split


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
            comments = [_clean_text(utt.text) for utt in group_utts]
            comments = [text for text in comments if text]
            if not comments:
                continue
            thread_text = "\n\n".join(comments)
            input_text = "\n\n".join(
                [
                    f"[TITLE] {convo.meta.get('op-title', '') or ''}",
                    f"[POST] {convo.meta.get('op-text-body', '') or ''}",
                    f"[ARGUMENT_THREAD] {thread_text}",
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
                    "op_user_id": op_user,
                    "op_title": convo.meta.get("op-title", ""),
                    "op_text_body": convo.meta.get("op-text-body", ""),
                    "conversation_pair_ids": json.dumps(
                        convo.meta.get("pair_ids", []), ensure_ascii=True
                    ),
                    "text": thread_text,
                    "input_text": input_text,
                    "label": label,
                    "train_flag": train_flag,
                    "num_labeled_utterances": len(group_utts),
                    "num_chars": len(thread_text),
                    "num_words": len(thread_text.split()),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return assign_holdout_split(
        df=df,
        group_col="conversation_id",
        train_flag_col="train_flag",
        holdout_val_ratio=0.5,
        seed=seed,
    )
