"""
U(rho) Experiment: Semantic Utility vs Completeness Ratio — v2 FIXED
=====================================================================
Measures how semantic utility degrades as a function of coefficient
completeness ratio rho, for DTDR-stored embeddings.

Two datasets: SIFT1M (geometric) and GloVe-100d (semantic)
Two dropout modes: random and block (shard-loss simulation)
Two metrics: cosine similarity and recall@10

KEY FIX from v1:
  Forward transform:  C = X @ T.T
  Inverse transform:  X = C @ T
  (T is orthogonal so T^-1 = T.T, but since we operate on row vectors
   the inverse of (X @ T.T) is (C @ T), not (C @ T.T))

A pipeline sanity check is printed first — cosine should be ~1.0.
"""

import numpy as np
import time
import json

# -----------------------------------------------------------------------
# PATHS — adjust to match your setup
# -----------------------------------------------------------------------
SIFT_BASE_NPY   = r"G:\train_jw\datasets\sift1m\sift_base.npy"
SIFT_QUERY_NPY  = r"G:\train_jw\datasets\sift1m\sift_query.npy"
SIFT_GT_IVECS   = r"G:\train_jw\datasets\sift1m\sift_groundtruth.ivecs"
GLOVE_TXT       = r"G:\train_jw\data\glove\glove.6B.100d.txt"

# -----------------------------------------------------------------------
# EXPERIMENT PARAMETERS
# -----------------------------------------------------------------------
RHO_VALUES      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
N_BASE_SAMPLE   = 10_000
N_QUERIES       = 500
K_RECALL        = 10
N_GLOVE_SAMPLE  = 10_000
RANDOM_SEED     = 42

np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------------------
# TRANSFORM
# -----------------------------------------------------------------------
def make_orthogonal_transform(n, seed=0):
    """Random orthogonal matrix via QR decomposition of Gaussian matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def forward_transform(X, T):
    """Row-vector convention: C = X @ T.T"""
    return X.astype(np.float32) @ T.T


def inverse_transform(C, T):
    """Row-vector convention: X = C @ T  (since T^-1 = T.T, so (X@T.T)@T = X)"""
    return C @ T


def quantise_int8(C):
    """Per-vector symmetric INT8 quantisation, returns dequantised float32."""
    scale = np.abs(C).max(axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(C / scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale


def pipeline_sanity_check(T, d, label=""):
    """Full round-trip at rho=1.0 should give cosine ~1.0."""
    rng = np.random.default_rng(99)
    X   = rng.standard_normal((20, d)).astype(np.float32)
    C   = quantise_int8(forward_transform(X, T))
    X_r = inverse_transform(C, T)
    A = X   / (np.linalg.norm(X,   axis=1, keepdims=True) + 1e-9)
    B = X_r / (np.linalg.norm(X_r, axis=1, keepdims=True) + 1e-9)
    cos = float(np.mean(np.sum(A * B, axis=1)))
    status = "OK" if cos > 0.99 else "*** FAIL ***"
    print(f"  [SANITY {status}] {label}  cosine(orig, recon) at rho=1.0 = {cos:.6f}")
    if cos < 0.99:
        raise RuntimeError("Pipeline sanity check failed.")
    return True


# -----------------------------------------------------------------------
# PARTIAL RECONSTRUCTION
# -----------------------------------------------------------------------
def partial_reconstruct(C_stored, rho, T, mode='random'):
    """
    Zero out (1-rho) fraction of coefficients then reconstruct.
    mode='random': random coefficient dropout
    mode='block' : keep first n_keep coefficients (contiguous shard loss)
    """
    n_coeff = C_stored.shape[1]
    n_keep  = max(1, int(round(rho * n_coeff)))
    mask    = np.zeros(n_coeff, dtype=np.float32)
    if mode == 'random':
        idx = np.random.choice(n_coeff, n_keep, replace=False)
        mask[idx] = 1.0
    else:  # block
        mask[:n_keep] = 1.0
    return inverse_transform(C_stored * mask[np.newaxis, :], T)


# -----------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------
def cosine_mean(A, B):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return float(np.mean(np.sum(A_n * B_n, axis=1)))


def recall_at_k(query_recons, base_vecs, gt_ids, k=10):
    """
    query_recons : dict {rho: (Nq,d) array}
    base_vecs    : (Nb,d) original base (search target unchanged)
    gt_ids       : (Nq,k) true top-k indices
    """
    recalls = {}
    Nq = gt_ids.shape[0]
    for rho, q_r in query_recons.items():
        scores   = q_r @ base_vecs.T
        pred     = np.argsort(-scores, axis=1)[:, :k]
        hits     = sum(len(set(gt_ids[i]) & set(pred[i])) for i in range(Nq))
        recalls[rho] = hits / (Nq * k)
    return recalls


# -----------------------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------------------
def load_ivecs(path):
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    n = len(data) // (d + 1)
    return data.reshape(n, d + 1)[:, 1:]


def load_glove(path, max_vectors=10_000):
    vecs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_vectors:
                break
            parts = line.split()
            vecs.append(np.array(parts[1:], dtype=np.float32))
    return np.stack(vecs)


# -----------------------------------------------------------------------
# MAIN EXPERIMENT
# -----------------------------------------------------------------------
def run_experiment(name, base_vecs, query_vecs, ground_truth):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  base={base_vecs.shape}  queries={query_vecs.shape}")
    print(f"{'='*60}")

    d = base_vecs.shape[1]
    T = make_orthogonal_transform(d)
    pipeline_sanity_check(T, d, label=name)

    rng      = np.random.default_rng(RANDOM_SEED)
    idx_base = rng.choice(len(base_vecs), min(N_BASE_SAMPLE, len(base_vecs)), replace=False)
    X_sample = base_vecs[idx_base].astype(np.float32)
    C_base   = quantise_int8(forward_transform(X_sample, T))

    idx_q    = rng.choice(len(query_vecs), min(N_QUERIES, len(query_vecs)), replace=False)
    X_query  = query_vecs[idx_q].astype(np.float32)
    C_query  = quantise_int8(forward_transform(X_query, T))
    gt_sub   = ground_truth[idx_q, :K_RECALL]

    base_full = base_vecs[:min(100_000, len(base_vecs))].astype(np.float32)

    results          = {k: {} for k in ['cosine_random','cosine_block',
                                         'recall_random','recall_block']}
    q_recon_random   = {}
    q_recon_block    = {}

    print(f"\n  {'rho':>5}  {'cos(rand)':>10}  {'cos(blk)':>10}")
    for rho in RHO_VALUES:
        r_r = partial_reconstruct(C_base, rho, T, 'random')
        r_b = partial_reconstruct(C_base, rho, T, 'block')
        c_r = cosine_mean(X_sample, r_r)
        c_b = cosine_mean(X_sample, r_b)
        results['cosine_random'][rho] = c_r
        results['cosine_block'][rho]  = c_b
        print(f"  {rho:>5.2f}  {c_r:>10.4f}  {c_b:>10.4f}")
        q_recon_random[rho] = partial_reconstruct(C_query, rho, T, 'random')
        q_recon_block[rho]  = partial_reconstruct(C_query, rho, T, 'block')

    print(f"\n  Computing recall@{K_RECALL} (searching {len(base_full):,} vectors)...")
    rec_r = recall_at_k(q_recon_random, base_full, gt_sub, K_RECALL)
    rec_b = recall_at_k(q_recon_block,  base_full, gt_sub, K_RECALL)
    results['recall_random'] = rec_r
    results['recall_block']  = rec_b

    print(f"\n  {'rho':>5}  {'rec(rand)':>10}  {'rec(blk)':>10}")
    for rho in RHO_VALUES:
        print(f"  {rho:>5.2f}  {rec_r[rho]:>10.4f}  {rec_b[rho]:>10.4f}")

    return results


def save_results(name, results):
    fname = f"urho_{name.lower().replace(' ','_').replace('-','_')}.json"
    out = {k: {str(rho): v for rho, v in d.items()} for k, d in results.items()}
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {fname}")


def print_table(name, results):
    print(f"\n--- {name} ---")
    print(f"{'rho':>6}  {'cos(rand)':>10}  {'cos(blk)':>10}  "
          f"{'rec(rand)':>10}  {'rec(blk)':>10}")
    for rho in RHO_VALUES:
        print(f"{rho:>6.2f}  "
              f"{results['cosine_random'].get(rho,float('nan')):>10.4f}  "
              f"{results['cosine_block'].get(rho,float('nan')):>10.4f}  "
              f"{results['recall_random'].get(rho,float('nan')):>10.4f}  "
              f"{results['recall_block'].get(rho,float('nan')):>10.4f}")


# -----------------------------------------------------------------------
if __name__ == '__main__':

    print("\nLoading SIFT1M...")
    t0 = time.time()
    sift_base  = np.load(SIFT_BASE_NPY).astype(np.float32)
    sift_query = np.load(SIFT_QUERY_NPY).astype(np.float32)
    sift_gt    = load_ivecs(SIFT_GT_IVECS)
    print(f"Loaded in {time.time()-t0:.1f}s")

    r = run_experiment("SIFT1M", sift_base, sift_query, sift_gt)
    print_table("SIFT1M", r)
    save_results("SIFT1M", r)

    print("\nLoading GloVe 100d...")
    t0 = time.time()
    glove = load_glove(GLOVE_TXT, N_GLOVE_SAMPLE)
    print(f"Loaded {glove.shape[0]} vectors in {time.time()-t0:.1f}s")

    split      = int(0.8 * len(glove))
    g_base     = glove[:split]
    g_query    = glove[split:]
    print("Computing GloVe ground truth...")
    g_gt       = np.argsort(-(g_query @ g_base.T), axis=1)[:, :K_RECALL]

    r = run_experiment("GloVe-100d", g_base, g_query, g_gt)
    print_table("GloVe-100d", r)
    save_results("GloVe-100d", r)

    print("\nDone.")
