import os, subprocess, pathlib
def run(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout); print(res.stderr); assert res.returncode == 0
def test_crf():
    here = pathlib.Path(__file__).resolve().parents[1]
    os.chdir(here)
    run('python scripts/make_synthetic.py')
    run('python -m src.training train_crf --data data/MBIB/all.jsonl --out results/crf_fast --no_embeddings')
    run('python -m src.training eval_crf   --data data/MBIB/all.jsonl --model_dir results/crf_fast --out results/crf_fast_eval --no_embeddings')
    assert os.path.exists('results/crf_fast_eval/metrics.csv')
