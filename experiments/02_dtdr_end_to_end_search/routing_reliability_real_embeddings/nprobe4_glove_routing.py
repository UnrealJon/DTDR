import numpy as np
import math
import hnswlib
from pathlib import Path

# ============================================================
# Hadamard (DTDR)
# ============================================================

def fwht_inplace(x):
    d = x.shape[-1]
    h = 1
    while h < d:
        x_ = x.reshape(*x.shape[:-1], -1, 2*h)
        a = x_[..., :h]
        b = x_[..., h:2*h]
        x_[..., :h] = a + b
        x_[..., h:2*h] = a - b
        h *= 2

def dtdr(x):
    y = x.astype(np.float32, copy=True)
    fwht_inplace(y)
    y /= math.sqrt(y.shape[-1])
    return y

# ============================================================
# Utilities
# ============================================================

def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def pad_to_pow2(x):
    """Pad vectors to next power-of-two dimension for Hadamard transform"""
    d = x.shape[1]
    p = 1 << (d-1).bit_length()
    if p == d:
        return x
    padded = np.zeros((x.shape[0], p), dtype=x.dtype)
    padded[:, :d] = x
    return padded

# ============================================================
# Load GloVe
# ============================================================

def load_glove(path, max_vectors=200000):
    vecs = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            vec = np.array(parts[1:], dtype=np.float32)
            vecs.append(vec)
            if len(vecs) >= max_vectors:
                break
    return np.vstack(vecs)

# ============================================================
# Ground truth
# ============================================================

def brute_topk(X, q, k):
    sims = X @ q
    return np.argsort(-sims)[:k]

# ============================================================
# Simple k-means IVF
# ============================================================

def kmeans(X, k, iters=8):
    rng = np.random.default_rng(0)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:,None,:]-C[None,:,:])**2).sum(-1)
        labels = d.argmin(1)
        for i in range(k):
            pts = X[labels==i]
            if len(pts):
                C[i]=pts.mean(0)
    return C

# ============================================================
# Dilution routing score
# ============================================================

def dilution_scores(q, X, lists):
    scores = np.zeros(len(lists))
    for i,ids in enumerate(lists):
        if len(ids)==0:
            continue
        sample = ids[:min(512,len(ids))]
        scores[i]= (X[sample]@q).max()
    return scores

# ============================================================
# MAIN
# ============================================================

DATA = Path("G:/train_jw/data/glove/glove.6B.50d.txt")

print("Loading GloVe...")
X = load_glove(DATA, 200000)

print("Normalizing...")
X = normalize(X)

print("Padding to power-of-two...")
X = pad_to_pow2(X)

print("Applying DTDR...")
X = dtdr(X)

print("Building queries...")
rng=np.random.default_rng(0)
Q=X[rng.choice(len(X),500,replace=False)]

print("Building IVF...")
nlist=512
C=kmeans(X,nlist)
d=((X[:,None,:]-C[None,:,:])**2).sum(-1)
labels=d.argmin(1)
lists=[np.where(labels==i)[0] for i in range(nlist)]

print("Building HNSW per list...")
hnsw={}
for i,ids in enumerate(lists):
    if len(ids)==0:
        continue
    idx=hnswlib.Index(space='cosine',dim=X.shape[1])
    idx.init_index(len(ids),M=16,ef_construction=200)
    idx.add_items(X[ids],ids)
    idx.set_ef(128)
    hnsw[i]=idx

kvals=[10,20,40,80,160,320]

print("\n k_local  recall@10  hit@1")

for K in kvals:

    hit1=0
    rec=0

    for q in Q:

        truth=brute_topk(X,q,10)
        truth1=truth[0]

        ranked=np.argsort(-dilution_scores(q,X,lists))
        probe=ranked[:2]

        cand=[]
        for li in probe:
            if li not in hnsw:
                continue
            k=min(K,len(lists[li]))
            if k<=0:
                continue
            f,_=hnsw[li].knn_query(q,k=k)
            cand.extend(f[0])

        cand=set(cand)

        if truth1 in cand:
            hit1+=1

        if len(cand)==0:
            continue

        final=sorted(cand,key=lambda i:-(X[i]@q))[:10]
        rec+=len(set(final)&set(truth))/10

    print(f"{K:8d} {rec/len(Q):10.4f} {hit1/len(Q):8.3f}")