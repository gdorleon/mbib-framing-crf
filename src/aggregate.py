from __future__ import annotations
import os, argparse, glob, pandas as pd
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results_root', default='results')
    ap.add_argument('--out_dir', default='results')
    args=ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)
    dirs=[p for p in glob.glob(os.path.join(args.results_root,'**'), recursive=True) if os.path.isdir(p)]
    rows=[]
    for d in dirs:
        m=os.path.join(d,'metrics.csv')
        if os.path.exists(m):
            df=pd.read_csv(m); df['source_dir']=d; rows.append(df)
    if rows:
        big=pd.concat(rows, ignore_index=True)
        big.to_csv(os.path.join(args.out_dir,'main_results.csv'), index=False)
    # subtype
    sub_rows=[]
    for d in dirs:
        for fn in os.listdir(d):
            if fn.endswith('_by_subtype.csv'):
                sdf=pd.read_csv(os.path.join(d,fn)); sdf['model']=os.path.basename(d).upper(); sub_rows.append(sdf)
    if sub_rows:
        pd.concat(sub_rows, ignore_index=True).to_csv(os.path.join(args.out_dir,'subtype.csv'), index=False)
    print('Aggregation done.')
if __name__=='__main__': main()
