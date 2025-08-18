import os, re, numpy as np, torch
from typing import List, Dict, Set
from transformers import AutoTokenizer, AutoModel

def load_bias_lexicon(path: str):
    if not os.path.exists(path): return set()
    return set(w.strip().lower() for w in open(path,encoding='utf-8') if w.strip() and not w.startswith('#'))

def load_mpqa_subjectivity(path: str):
    if not os.path.exists(path): return {}
    lex = {}
    for line in open(path,encoding='utf-8'):
        if line.startswith('#') or not line.strip(): continue
        parts = dict(kv.split('=') for kv in line.strip().split() if '=' in kv)
        w = parts.get('word1','').lower(); pol = parts.get('priorpolarity','neutral')
        if w: lex[w]=pol
    return lex

def sentence_stats(sent: str, mpqa: Dict[str,str], bias_terms: Set[str]):
    tokens = re.findall(r"[A-Za-z']+", sent.lower())
    pos = sum(1 for t in tokens if mpqa.get(t)=='positive')
    neg = sum(1 for t in tokens if mpqa.get(t)=='negative')
    subj = sum(1 for t in tokens if mpqa.get(t) in ('positive','negative'))
    bias = sum(1 for t in tokens if t in bias_terms)
    has_quotes = 1.0 if '"' in sent or '\'' in sent else 0.0
    has_modal = 1.0 if re.search(r"\b(might|could|reportedly|allegedly|seems|appears)\b", sent.lower()) else 0.0
    return {'sent_pos':float(pos),'sent_neg':float(neg),'sent_subj':float(subj),'bias_lex':float(bias),
            'has_quotes':has_quotes,'has_modal':has_modal,'len':float(len(tokens))}

class SentenceEmbedder:
    def __init__(self, model_name='bert-base-uncased', layer=-1, pool='mean', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.layer = layer; self.pool = pool
    def encode(self, sentences: List[str]):
        with torch.no_grad():
            batch = self.tok(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            out = self.model(**batch, output_hidden_states=True)
            hs = out.hidden_states[self.layer]
            if self.pool=='mean':
                mask = batch['attention_mask'].unsqueeze(-1)
                emb = (hs*mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                emb = hs[:,0,:]
            return emb.cpu().numpy()
