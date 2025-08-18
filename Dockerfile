FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD python scripts/make_synthetic.py &&     python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_fast --no_embeddings &&     python -m src.training eval_crf   --data data/MBIB/all.jsonl --model_dir results/crf_fast --out results/crf_fast_eval --no_embeddings &&     python -m src.aggregate --results_root results --out_dir results &&     python -m src.plots --results_dir results --out_dir results/figs
