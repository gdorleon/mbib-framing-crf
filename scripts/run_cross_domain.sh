#!/usr/bin/env bash
set -euo pipefail
python -m src.training cross_domain --train data/BASIL/articles.jsonl --test data/WIKI/articles.jsonl --out results/cross --model crf
python -m src.training cross_domain --train data/WIKI/articles.jsonl  --test data/BASIL/articles.jsonl --out results/cross --model transformer --arch bert
