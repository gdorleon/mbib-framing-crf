from __future__ import annotations
import os, argparse
from .data import load_articles
from .features import load_bias_lexicon, load_mpqa_subjectivity, SentenceEmbedder
from .models.crf_model import CRFExperiment
from .models.bert_classifier import train_transformer as train_tx, eval_transformer as eval_tx
from .eval import article_level_metrics, subtype_breakdown
from .utils import set_seed

def build_feature_ctx(articles, emb_model, emb_layer, emb_pool, use_embeddings, bias_lex_path, mpqa_path):
    ctx={}
    ctx['bias_terms']=load_bias_lexicon(bias_lex_path)
    ctx['mpqa']=load_mpqa_subjectivity(mpqa_path)
    if use_embeddings:
        enc=SentenceEmbedder(model_name=emb_model, layer=emb_layer, pool=emb_pool)
        emb_map={a['article_id']: enc.encode(a['sentences']) for a in articles}
        ctx['embeddings']=emb_map
    else:
        ctx['embeddings']={}
    return ctx

def dump_result_row(out_dir, row, name='metrics.csv'):
    os.makedirs(out_dir, exist_ok=True); import csv
    path=os.path.join(out_dir,name); header=not os.path.exists(path)
    with open(path,'a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=row.keys()); 
        if header: w.writeheader()
        w.writerow(row)

def cmd_train_crf(args):
    set_seed(args.seed)
    arts=[a.__dict__ for a in load_articles(args.data)]
    tr=[a for a in arts if a.get('split')=='train']; va=[a for a in arts if a.get('split')=='val']
    ctx=build_feature_ctx(tr if not args.embed_all else arts, args.embedding_model, args.embedding_layer, args.embedding_pool, (not args.no_embeddings), args.bias_lex, args.mpqa_lex)
    exp=CRFExperiment(use_transitions=not args.no_transitions, use_lexicons=not args.no_lexicons, use_embeddings=not args.no_embeddings, emb_pca_dim=args.embedding_pca_dim, random_state=args.seed)
    exp.fit(tr, ctx); y_true,y_pred=exp.predict(va, ctx)
    y_true_i=[[int(y) for y in s] for s in y_true]; y_pred_i=[[int(y) for y in s] for s in y_pred]
    m=article_level_metrics(y_true_i, y_pred_i); dump_result_row(args.out, {'model':'CRF','domain':'MBIB','split':'val', **m, 'notes':'validation'})
    exp.save(args.out)

def cmd_eval_crf(args):
    arts=[a.__dict__ for a in load_articles(args.data)]; te=[a for a in arts if a.get('split')=='test']
    ctx=build_feature_ctx(te if not args.embed_all else arts, args.embedding_model, args.embedding_layer, args.embedding_pool, (not args.no_embeddings), args.bias_lex, args.mpqa_lex)
    exp=CRFExperiment(); exp.load(args.model_dir); y_true,y_pred=exp.predict(te, ctx)
    y_true_i=[[int(y) for y in s] for s in y_true]; y_pred_i=[[int(y) for y in s] for s in y_pred]
    m=article_level_metrics(y_true_i, y_pred_i); dump_result_row(args.out, {'model':'CRF','domain':'MBIB','split':'test', **m, 'notes':'test'})
    # per-article preds
    import csv
    with open(os.path.join(args.out,'crf_preds.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['article_id','y_true','y_pred']); w.writeheader()
        for art, yts, yps in zip(te, y_true_i, y_pred_i):
            yt=int(any(int(x)==1 for x in yts)); yp=int(any(int(x)==1 for x in yps))
            w.writerow({'article_id': art['article_id'], 'y_true': yt, 'y_pred': yp})
    rows=subtype_breakdown(te, y_pred_i)
    if rows:
        import pandas as pd; pd.DataFrame(rows).to_csv(os.path.join(args.out,'crf_by_subtype.csv'), index=False)

def cmd_train_transformer(args):
    set_seed(args.seed)
    arch=args.arch.lower()
    model_name=args.model_name or ('bert-base-uncased' if arch=='bert' else 'YituTech/conv-bert-base')
    arts=[a.__dict__ for a in load_articles(args.data)]
    tr=[a for a in arts if a.get('split')=='train']; va=[a for a in arts if a.get('split')=='val']
    out=train_tx(arch, model_name, args.out, tr, va, max_len=args.max_seq_length, lr=args.lr, epochs=args.epochs, bs=args.batch_size, fp16=args.fp16, seed=args.seed)
    print('Saved model to', out)

def cmd_eval_transformer(args):
    arch=args.arch.lower()
    arts=[a.__dict__ for a in load_articles(args.data)]; te=[a for a in arts if a.get('split')=='test']
    m=eval_tx(args.model_dir, te, max_len=args.max_seq_length); dump_result_row(args.out, {'model': arch.upper(),'domain':'MBIB','split':'test', **m, 'notes':'test'})
    # per-article preds
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
    import torch, csv, numpy as np
    tok=AutoTokenizer.from_pretrained(args.model_dir); model=AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    class _DS(torch.utils.data.Dataset):
        def __init__(self, arts, tok, max_len): self.arts=arts; self.tok=tok; self.max_len=max_len
        def __len__(self): return len(self.arts)
        def __getitem__(self, idx):
            a=self.arts[idx]; txt=' '.join(a['sentences'])
            enc=self.tok(txt, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            enc={k:v.squeeze(0) for k,v in enc.items()}; enc['labels']=torch.tensor(0); enc['article_id']=a['article_id']; return enc
    ds=_DS(te, tok, args.max_seq_length); trainer=Trainer(model=model); logits=trainer.predict(ds).predictions
    y_pred=logits.argmax(-1); y_true=np.array([1 if any(int(x)==1 for x in a['sentence_labels']) else 0 for a in te])
    with open(os.path.join(args.out, f'{arch}_preds.csv'),'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['article_id','y_true','y_pred']); w.writeheader()
        for a, yt, yp in zip(te, y_true, y_pred): w.writerow({'article_id': a['article_id'], 'y_true': int(yt), 'y_pred': int(yp)})

def cmd_cross_domain(args):
    tr=[a.__dict__ for a in load_articles(args.train)]; te=[a.__dict__ for a in load_articles(args.test)]
    if args.model=='crf':
        ctx=build_feature_ctx(tr, args.embedding_model, args.embedding_layer, args.embedding_pool, True, args.bias_lex, args.mpqa_lex)
        exp=CRFExperiment(); exp.fit(tr, ctx); y_true,y_pred=exp.predict(te, ctx)
        y_true_i=[[int(y) for y in s] for s in y_true]; y_pred_i=[[int(y) for y in s] for s in y_pred]
        m=article_level_metrics(y_true_i, y_pred_i); dump_result_row(args.out, {'model':'CRF','domain':'cross','split':'test', **m, 'notes': f'{args.train}->{args.test}'})
    else:
        arch=args.arch.lower(); model_name='bert-base-uncased' if arch=='bert' else 'YituTech/conv-bert-base'
        out=os.path.join(args.out, f'{arch}_cross'); from .models.bert_classifier import train_transformer as tx, eval_transformer as ex
        tx(arch, model_name, out, tr, te, max_len=args.max_seq_length, lr=args.lr, epochs=args.epochs, bs=args.batch_size, fp16=args.fp16, seed=args.seed)
        m=ex(out, te, max_len=args.max_seq_length); dump_result_row(args.out, {'model': arch.upper(),'domain':'cross','split':'test', **m, 'notes': f'{args.train}->{args.test}'})

def build_argparser():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers()
    ap.add_argument('--seed', type=int, default=42)
    p=sub.add_parser('train_crf')
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--embedding_model', default='bert-base-uncased'); p.add_argument('--embedding_layer', type=int, default=-1)
    p.add_argument('--embedding_pool', default='mean', choices=['mean','cls']); p.add_argument('--embedding_pca_dim', type=int, default=50)
    p.add_argument('--bias_lex', default='assets/lexicons/bias_lexicon.txt'); p.add_argument('--mpqa_lex', default='assets/lexicons/mpqa_subjectivity_sample.tff')
    p.add_argument('--no_transitions', action='store_true'); p.add_argument('--no_lexicons', action='store_true'); p.add_argument('--no_embeddings', action='store_true')
    p.add_argument('--embed_all', action='store_true'); p.set_defaults(func=cmd_train_crf)
    p=sub.add_parser('eval_crf')
    p.add_argument('--data', required=True); p.add_argument('--model_dir', required=True); p.add_argument('--out', required=True)
    p.add_argument('--embedding_model', default='bert-base-uncased'); p.add_argument('--embedding_layer', type=int, default=-1)
    p.add_argument('--embedding_pool', default='mean', choices=['mean','cls']); p.add_argument('--bias_lex', default='assets/lexicons/bias_lexicon.txt')
    p.add_argument('--mpqa_lex', default='assets/lexicons/mpqa_subjectivity_sample.tff'); p.add_argument('--no_embeddings', action='store_true'); p.add_argument('--embed_all', action='store_true')
    p.set_defaults(func=cmd_eval_crf)
    p=sub.add_parser('train_transformer')
    p.add_argument('--arch', required=True, choices=['bert','convbert','roberta_da']); p.add_argument('--model_name', default=None)
    p.add_argument('--data', required=True); p.add_argument('--out', required=True); p.add_argument('--max_seq_length', type=int, default=512)
    p.add_argument('--lr', type=float, default=2e-5); p.add_argument('--epochs', type=int, default=3); p.add_argument('--batch_size', type=int, default=8); p.add_argument('--fp16', action='store_true')
    p.set_defaults(func=cmd_train_transformer)
    p=sub.add_parser('eval_transformer')
    p.add_argument('--arch', required=True, choices=['bert','convbert','roberta_da']); p.add_argument('--data', required=True); p.add_argument('--model_dir', required=True); p.add_argument('--out', required=True)
    p.add_argument('--max_seq_length', type=int, default=512); p.set_defaults(func=cmd_eval_transformer)
    p=sub.add_parser('cross_domain')
    p.add_argument('--train', required=True); p.add_argument('--test', required=True); p.add_argument('--out', required=True)
    p.add_argument('--model', default='crf', choices=['crf','transformer']); p.add_argument('--arch', default='bert', choices=['bert','convbert'])
    p.add_argument('--embedding_model', default='bert-base-uncased'); p.add_argument('--embedding_layer', type=int, default=-1); p.add_argument('--embedding_pool', default='mean', choices=['mean','cls'])
    p.add_argument('--bias_lex', default='assets/lexicons/bias_lexicon.txt'); p.add_argument('--mpqa_lex', default='assets/lexicons/mpqa_subjectivity_sample.tff')
    p.add_argument('--max_seq_length', type=int, default=512); p.add_argument('--lr', type=float, default=2e-5); p.add_argument('--epochs', type=int, default=3); p.add_argument('--batch_size', type=int, default=8); p.add_argument('--fp16', action='store_true')
    p.set_defaults(func=cmd_cross_domain)
    return ap

def main():
    ap=build_argparser(); args=ap.parse_args()
    if hasattr(args,'func'): args.func(args)
    else: ap.print_help()

if __name__=='__main__': main()
