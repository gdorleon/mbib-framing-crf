# MBIB Framing Bias Experiments (CRF + Transformers)

Code to reproduce the experiments from the paper *"Detecting Framing Bias in News via Probabilistic Graphical Modeling"*.

- Proposed **CRF** with lexicon + embedding features, sentence-level labels -> article-level.
- Baselines: **BERT**, **ConvBERT**.
- Tasks: main results, ablations, cross-domain, subtype breakdown

## Unified JSONL Format
```json
{
  "article_id": "basil_000123",
  "domain": "BASIL",
  "split": "train",
  "sentences": ["...", "..."],
  "sentence_labels": [0,1,...],
  "subtypes": ["loaded","none", ...]  // optional
}
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CRF
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf
python -m src.training eval_crf   --data data/MBIB/all.jsonl --model_dir results/crf --out results/crf_eval

# BERT
python -m src.training train_transformer --arch bert --data data/MBIB/all.jsonl --out results/bert
python -m src.training eval_transformer  --arch bert --data data/MBIB/all.jsonl --model_dir results/bert --out results/bert_eval

# ConvBERT
python -m src.training train_transformer --arch convbert --data data/MBIB/all.jsonl --out results/convbert
python -m src.training eval_transformer  --arch convbert --data data/MBIB/all.jsonl --model_dir results/convbert --out results/convbert_eval

# Cross-domain
python -m src.training cross_domain --train data/BASIL/articles.jsonl --test data/WIKI/articles.jsonl --out results/cross --model crf
python -m src.training cross_domain --train data/WIKI/articles.jsonl  --test data/BASIL/articles.jsonl --out results/cross --model transformer --arch bert

# Ablations
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_no_trans --no_transitions
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_no_lex   --no_lexicons
python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_no_emb   --no_embeddings

# Aggregate + plots
bash scripts/make_figs.sh
```

## Synthetic data (if you want to use to test only)
```bash
python scripts/make_synthetic.py
```

---

## Authors
removed while paper is under review

## Citation
If you use this code, please cite:

```bibtex
@inproceedings{...,
  title     = {Detecting Framing Bias in News via Probabilistic Graphical Modeling},
  author    = {S Shujaa and G. Dorleon},
  booktitle = {Proceedings of the ACM/IEEE Joint Conference on Digital Libraries (JCDL 2025)},
  year      = {2025},
  month     = dec,
  note      = {Fully virtual conference, December 15--19, 2025}
}

