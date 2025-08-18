#!/usr/bin/env bash
set -euo pipefail
python -m src.training train_transformer --arch bert --data data/MBIB/all.jsonl --out results/bert
python -m src.training eval_transformer  --arch bert --data data/MBIB/all.jsonl --model_dir results/bert --out results/bert_eval
