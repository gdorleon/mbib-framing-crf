from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Article:
    article_id: str
    domain: str
    split: str
    sentences: List[str]
    sentence_labels: List[int]
    subtypes: Optional[List[str]] = None

def read_jsonl(path: str):
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                yield json.loads(line)

def load_articles(path: str) -> List[Article]:
    items = []
    for ex in read_jsonl(path):
        items.append(Article(
            article_id=ex['article_id'], domain=ex.get('domain','MBIB'), split=ex.get('split','train'),
            sentences=ex['sentences'], sentence_labels=[int(x) for x in ex['sentence_labels']],
            subtypes=ex.get('subtypes')
        ))
    return items
