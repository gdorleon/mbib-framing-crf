#!/usr/bin/env bash
set -euo pipefail
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf
python -m src.training eval_crf   --data data/MBIB/all.jsonl --model_dir results/crf --out results/crf_eval
