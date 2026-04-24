from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from ..splits import assign_group_split, SplitConfig


def build_coarse_discourse_dataframe(corpus, seed: int = 42) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for convo in corpus.iter_conversations():
        subreddit = convo.meta.get("subreddit")
        title = convo.meta.get("title", "")
        for utt in convo.iter_utterances():
            label = utt.meta.get("majority_type")
            if not label:
                continue
            text = (utt.text or "").strip()
            rows.append(
                {
                    "conversation_id": convo.id,
                    "utterance_id": utt.id,
                    "speaker": getattr(getattr(utt, "speaker", None), "id", None),
                    "subreddit": subreddit,
                    "title": title,
                    "reply_to": utt.reply_to,
                    "comment_depth": utt.meta.get("post_depth"),
                    "majority_link": utt.meta.get("majority_link"),
                    "annotation_types": json.dumps(utt.meta.get("annotation-types"), ensure_ascii=True),
                    "annotation_links": json.dumps(utt.meta.get("annotation-links"), ensure_ascii=True),
                    "ups": utt.meta.get("ups"),
                    "text": text,
                    "input_text": text,
                    "label": label,
                    "num_chars": len(text),
                    "num_words": len(text.split()),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "conversation_id", "utterance_id", "speaker", "subreddit",
                "title", "reply_to", "comment_depth", "majority_link",
                "annotation_types", "annotation_links", "ups", "text",
                "input_text", "label", "num_chars", "num_words", "split",
            ]
        )
        return df
    return assign_group_split(
        df=df,
        group_col="conversation_id",
        split_cfg=SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed),
    )
