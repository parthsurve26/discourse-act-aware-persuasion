#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/train_persuasion.py \
  --architecture fresh-bert \
  --local-files-only \
  "$@"

python scripts/train_persuasion.py \
  --architecture fresh-attn \
  --local-files-only \
  "$@"

python scripts/train_persuasion.py \
  --architecture fresh-bilstm-attn \
  --local-files-only \
  "$@"

python scripts/train_persuasion.py \
  --architecture discourse-attn \
  --local-files-only \
  "$@"

python scripts/train_persuasion.py \
  --architecture discourse-bilstm-attn \
  --local-files-only \
  "$@"
