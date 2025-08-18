import argparse, pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preds_a', required=True); ap.add_argument('--preds_b', required=True)
    args=ap.parse_args()
    a=pd.read_csv(args.preds_a).set_index('article_id'); b=pd.read_csv(args.preds_b).set_index('article_id')
    common=a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
    ca=(common['y_true_a']==common['y_pred_a']).astype(int); cb=(common['y_true_b']==common['y_pred_b']).astype(int)
    n01=int(((ca==0)&(cb==1)).sum()); n10=int(((ca==1)&(cb==0)).sum())
    res=mcnemar([[0,n01],[n10,0]], exact=False, correction=True)
    print(f"McNemar: statistic={res.statistic:.4f}, p-value={res.pvalue:.6f} (n01={n01}, n10={n10})")
if __name__=='__main__': main()
