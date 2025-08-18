import os, argparse, pandas as pd, matplotlib.pyplot as plt, numpy as np
def plot_main_results(df, out_dir):
    os.makedirs(out_dir, exist_ok=True); df=df.sort_values('model')
    x=np.arange(len(df)); w=0.25
    plt.figure(); plt.bar(x-w, df['f1'], w, label='F1'); plt.bar(x, df['precision'], w, label='Precision'); plt.bar(x+w, df['recall'], w, label='Recall')
    plt.xticks(x, df['model'], rotation=15); plt.ylabel('Score'); plt.title('Main Results'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'main_results_bar.png'), dpi=300); plt.close()
def plot_learning_curve(df, out_dir):
    os.makedirs(out_dir, exist_ok=True); plt.figure()
    for m,sub in df.groupby('model'): sub=sub.sort_values('train_pct'); plt.plot(sub['train_pct'], sub['f1'], marker='o', label=m)
    plt.xlabel('Training size (%)'); plt.ylabel('F1'); plt.title('Learning Curves'); plt.legend(); plt.grid(True, ls='--', alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'learning_curve.png'), dpi=300); plt.close()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results_dir', required=True); ap.add_argument('--out_dir', required=True); a=ap.parse_args()
    m=os.path.join(a.results_dir,'main_results.csv'); lc=os.path.join(a.results_dir,'learning_curve.csv')
    if os.path.exists(m): plot_main_results(pd.read_csv(m), a.out_dir)
    if os.path.exists(lc): plot_learning_curve(pd.read_csv(lc), a.out_dir)
if __name__=='__main__': main()
