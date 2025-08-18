import os, joblib, numpy as np, sklearn_crfsuite
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from .utils_crf import make_crf_instances

class CRFExperiment:
    def __init__(self, use_transitions=True, use_lexicons=True, use_embeddings=True, emb_pca_dim=50, random_state=42):
        self.use_transitions = use_transitions; self.use_lexicons=use_lexicons; self.use_embeddings=use_embeddings
        self.emb_pca_dim = emb_pca_dim; self.random_state=random_state
        self.pca=None
        self.crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=50,
                                        all_possible_transitions=True if use_transitions else False, c2=0.01)
    def _maybe_fit_pca(self, Xseq):
        if not self.use_embeddings: return
        embs=[]
        for seq in Xseq:
            for feats in seq:
                vec=[v for k,v in feats.items() if k.startswith('embed_')]
                if vec: embs.append(vec)
        if not embs: return
        mat=np.array(embs); k=min(self.emb_pca_dim, mat.shape[1])
        self.pca = PCA(n_components=k, random_state=self.random_state).fit(mat)
    def _apply_pca(self, Xseq):
        if not self.pca: return Xseq
        Xnew=[]
        for seq in Xseq:
            seq_new=[]
            for feats in seq:
                emb_keys=[k for k in feats if k.startswith('embed_')]
                if emb_keys:
                    vec=np.array([feats[k] for k in emb_keys]).reshape(1,-1)
                    red=self.pca.transform(vec)[0]
                    for k in emb_keys: feats.pop(k, None)
                    for i,val in enumerate(red): feats[f'emb{i}']=float(val)
                seq_new.append(feats)
            Xnew.append(seq_new)
        return Xnew
    def fit(self, articles, feature_ctx):
        Xseq,Yseq=make_crf_instances(articles, feature_ctx, self.use_lexicons, self.use_embeddings)
        self._maybe_fit_pca(Xseq); Xseq=self._apply_pca(Xseq); self.crf.fit(Xseq,Yseq)
    def predict(self, articles, feature_ctx):
        Xseq,Yseq=make_crf_instances(articles, feature_ctx, self.use_lexicons, self.use_embeddings)
        Xseq=self._apply_pca(Xseq) if self.pca else Xseq
        Yhat=self.crf.predict(Xseq); return Yseq, Yhat
    @staticmethod
    def seq_to_article_labels(seqs): return [int(any(int(x)==1 for x in seq)) for seq in seqs]
    @staticmethod
    def eval_article(y_true_seq, y_pred_seq):
        y_true=CRFExperiment.seq_to_article_labels(y_true_seq); y_pred=CRFExperiment.seq_to_article_labels(y_pred_seq)
        p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
        acc=accuracy_score(y_true,y_pred); return {'accuracy':acc,'precision':p,'recall':r,'f1':f1}
    def save(self, out_dir): os.makedirs(out_dir, exist_ok=True); joblib.dump(self.crf, f"{out_dir}/crf.joblib"); joblib.dump(self.pca, f"{out_dir}/pca.joblib")
    def load(self, out_dir): import os; self.crf=joblib.load(f"{out_dir}/crf.joblib"); self.pca=joblib.load(f"{out_dir}/pca.joblib") if os.path.exists(f"{out_dir}/pca.joblib") else None
