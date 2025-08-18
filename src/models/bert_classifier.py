from __future__ import annotations
import torch, numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def article_label_from_sentences(labels): return int(any(int(x)==1 for x in labels))

class ArticleDataset(Dataset):
    def __init__(self, articles, tokenizer, max_len=512):
        self.articles=articles; self.tok=tokenizer; self.max_len=max_len
    def __len__(self): return len(self.articles)
    def __getitem__(self, idx):
        a=self.articles[idx]; text=' '.join(a['sentences']); y=article_label_from_sentences(a['sentence_labels'])
        enc=self.tok(text, truncation=True, padding='max_length', max_length=self.max_len)
        enc={k: torch.tensor(v) for k,v in enc.items()}; enc['labels']=torch.tensor(int(y)); return enc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits.argmax(-1) if logits.ndim==2 else (logits>0)).astype(int)
    p,r,f1,_=precision_recall_fscore_support(labels,preds,average='binary',zero_division=0)
    acc=accuracy_score(labels,preds); return {'accuracy':acc,'precision':p,'recall':r,'f1':f1}

def train_transformer(arch, model_name, out_dir, train_arts, val_arts, max_len=512, lr=2e-5, epochs=3, bs=8, fp16=False, seed=42):
    tok=AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tr=ArticleDataset(train_arts, tok, max_len); va=ArticleDataset(val_arts, tok, max_len)
    args=TrainingArguments(output_dir=out_dir, learning_rate=lr, num_train_epochs=epochs,
                           per_device_train_batch_size=bs, per_device_eval_batch_size=bs,
                           evaluation_strategy='epoch', save_strategy='epoch',
                           load_best_model_at_end=True, metric_for_best_model='f1', greater_is_better=True,
                           fp16=fp16, seed=seed, logging_steps=50, save_total_limit=2)
    trainer=Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, compute_metrics=compute_metrics)
    trainer.train(); trainer.save_model(out_dir); tok.save_pretrained(out_dir); return out_dir

def eval_transformer(model_dir, data_arts, max_len=512):
    tok=AutoTokenizer.from_pretrained(model_dir); model=AutoModelForSequenceClassification.from_pretrained(model_dir)
    ds=ArticleDataset(data_arts, tok, max_len); trainer=Trainer(model=model)
    preds=trainer.predict(ds).predictions; y_true=np.array([article_label_from_sentences(a['sentence_labels']) for a in data_arts])
    y_pred=preds.argmax(-1); 
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0); acc=accuracy_score(y_true,y_pred)
    return {'accuracy':acc,'precision':p,'recall':r,'f1':f1}
