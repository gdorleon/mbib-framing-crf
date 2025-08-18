from ..features import sentence_stats
import numpy as np
def make_crf_instances(articles, feature_ctx, use_lexicons=True, use_embeddings=True):
    Xseq, Yseq = [], []
    mpqa = feature_ctx.get('mpqa', {}); bias_terms = feature_ctx.get('bias_terms', set())
    embeds = feature_ctx.get('embeddings')
    for art in articles:
        feats_seq, labels_seq = [], []
        sents = art['sentences']; ys = [int(x) for x in art['sentence_labels']]
        emb = embeds.get(art['article_id']) if (use_embeddings and embeds) else None
        for i, sent in enumerate(sents):
            d = {'position': float(i), 'lead': 1.0 if i==0 else 0.0}
            if use_lexicons: d.update(sentence_stats(sent, mpqa, bias_terms))
            if emb is not None:
                vec = emb[i] if i < emb.shape[0] else np.zeros(emb.shape[1], dtype=float)
                for j, val in enumerate(vec.tolist()):
                    d[f'embed_{j}'] = float(val)
            feats_seq.append(d); labels_seq.append(str(ys[i]))
        Xseq.append(feats_seq); Yseq.append(labels_seq)
    return Xseq, Yseq
