#!/usr/bin/env bash
set -euo pipefail
python -m src.training train_transformer --arch convbert --data data/MBIB/all.jsonl --out results/convbert
python -m src.training eval_transformer  --arch convbert --data data/MBIB/all.jsonl --model_dir results/convbert --out results/convbert_eval
