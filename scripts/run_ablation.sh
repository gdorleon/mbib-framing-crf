#!/usr/bin/env bash
set -euo pipefail
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_no_trans --no_transitions
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_no_lex   --no_lexicons
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_no_emb   --no_embeddings
