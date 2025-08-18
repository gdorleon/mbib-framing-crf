from __future__ import annotations
import os
from collections import defaultdict

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def article_level_metrics(y_true_seq, y_pred_seq):
    y_true=[int(any(int(x)==1 for x in s)) for s in y_true_seq]
    y_pred=[int(any(int(x)==1 for x in s)) for s in y_pred_seq]
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0); acc=accuracy_score(y_true,y_pred)
    return {'accuracy':acc,'precision':p,'recall':r,'f1':f1}

def subtype_breakdown(articles, y_pred_seq):
    stats=defaultdict(lambda:{'tp':0,'fp':0,'fn':0,'n':0})
    rows=[]
    for art,yhat in zip(articles,y_pred_seq):
        subs=art.get('subtypes'); if not subs: continue
        for gold,pred,st in zip(art['sentence_labels'], yhat, subs):
            stats[st]['n']+=1
            if int(pred)==1 and int(gold)==1: stats[st]['tp']+=1
            elif int(pred)==1 and int(gold)==0: stats[st]['fp']+=1
            elif int(pred)==0 and int(gold)==1: stats[st]['fn']+=1
    for st,d in stats.items():
        tp,fp,fn=d['tp'],d['fp'],d['fn']
        prec= tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1  = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        rows.append({'subtype':st,'precision':prec,'recall':rec,'f1':f1,'support':d['n']})
    return rows
