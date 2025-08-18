import random, numpy as np, torch
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def article_label_from_sentences(sentence_labels): return int(any(int(x)==1 for x in sentence_labels))
