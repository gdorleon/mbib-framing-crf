#!/usr/bin/env python
import os, json, random
random.seed(13)

def mk_article(prefix, idx, split):
    n = random.randint(5, 8)
    sentences = []
    labels = []
    subtypes = []
    biased_positions = set(random.sample(range(n), k=random.randint(0,2)))
    for i in range(n):
        if i in biased_positions:
            sentences.append(f"The administration's draconian policy was reckless according to sources ({prefix}-{idx}-{i}).")
            labels.append(1)
            subtypes.append(random.choice(['loaded','emotive']))
        else:
            sentences.append(f"The government announced a funding reduction in {prefix}-{idx}-{i}.")
            labels.append(0)
            subtypes.append('none')
    return {
        "article_id": f"{prefix}_{idx:05d}",
        "domain": prefix.upper(),
        "split": split,
        "sentences": sentences,
        "sentence_labels": labels,
        "subtypes": subtypes
    }

def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    os.makedirs('data/BASIL', exist_ok=True)
    os.makedirs('data/WIKI', exist_ok=True)
    os.makedirs('data/MFC', exist_ok=True)

    basil = [mk_article('basil', i, 'train' if i<30 else 'val' if i<40 else 'test') for i in range(60)]
    wiki  = [mk_article('wiki',  i, 'train' if i<30 else 'val' if i<40 else 'test') for i in range(60)]
    mfc   = [mk_article('mfc',   i, 'train' if i<30 else 'val' if i<40 else 'test') for i in range(60)]

    write_jsonl('data/BASIL/articles.jsonl', basil)
    write_jsonl('data/WIKI/articles.jsonl', wiki)
    write_jsonl('data/MFC/articles.jsonl', mfc)

    merged = basil + wiki + mfc
    write_jsonl('data/MBIB/all.jsonl', merged)
    print('Synthetic datasets written under data/')

if __name__ == '__main__':
    main()
