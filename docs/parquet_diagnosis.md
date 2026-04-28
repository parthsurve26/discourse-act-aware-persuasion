# Winning Arguments Parquet Diagnosis

File checked:

```text
data/processed/winning_arguments.parquet
```

## Result

This parquet is the **new corrected pair/thread-level parquet**.

It is not the old utterance-level version.

## Checks

File exists:

```text
yes
```

Row count:

```text
8,526
```

Schema:

```text
conversation_id: str
prediction_unit: str
pair_id: str
argument_thread_id: str
labeled_utterance_ids: str
speaker_ids: str
op_user_id: str
op_title: str
op_text_body: str
conversation_pair_ids: str
text: str
input_text: str
label: int64
train_flag: int64
num_labeled_utterances: int64
num_chars: int64
num_words: int64
split: str
```

Label counts:

```text
label 1: 4,263
label 0: 4,263
```

Required corrected-version columns:

```text
prediction_unit: present
pair_id: present
argument_thread_id: present
input_text: present
split: present
num_labeled_utterances: present
```

Duplicate `(pair_id, argument_thread_id)` rows:

```text
0
```

Rows per `pair_id`:

```text
2 rows per pair_id: 4,263 pair_ids
```

Split values:

```text
train
val
test
```

Split counts:

```text
train: 6,912
val: 776
test: 838
```

Pair overlap across splits:

```text
test/train overlap: 0
test/val overlap: 0
train/val overlap: 0
```

## Diagnosis

The parquet matches the corrected official data definition:

```text
one row per (pair_id, top-level reply thread)
```

The key signs are:

- pair/thread-specific columns are present
- labels are balanced at 4,263 successful and 4,263 unsuccessful rows
- every `pair_id` has exactly two rows
- there are no duplicate `(pair_id, argument_thread_id)` rows
- no `pair_id` appears in more than one split

## Regeneration Command

No regeneration is needed.

If this file ever becomes missing, old, or invalid, regenerate it with:

```bash
python scripts/preprocess_datasets.py --dataset winning-arguments
```

