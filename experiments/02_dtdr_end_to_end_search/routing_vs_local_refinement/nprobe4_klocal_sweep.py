import argparse
import math
import numpy as np
import hnswlib
import csv

# ---------------------------
# Hadamard (DTDR domain)
# ---------------------------
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

def dtdr_hadamard(x):
    y = x.astype(np.float32, copy=True)
    fwht_inplace(y)
    y /= math.sqrt(y.shape[-1])
    return y

def l2_normalize(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)

# ---------------------------
# Simple kmeans IVF
# ---------------------------
def kmeans_simple(X, k, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    C = X[rng.choice(N, k, replace=False)].copy()
    for _ in range(iters):
        dots = X @ C.T
        d2 = ((X*X).sum(1, keepdims=True) + (C*C).sum(1) - 2*dots)
        labels = d2.argmin(1)
        for i in range(k):
            pts = X[labels == i]
            if len(pts) > 0:
                C[i] = pts.mean(0)
    return C

def ivf_assign(X, C):
    dots = X @ C.T
    d2 = ((X*X).sum(1, keepdims=True) + (C*C).sum(1) - 2*dots)
    return d2.argmin(1)

# ---------------------------
# Dilution ranking
# ---------------------------
def dilution_evidence_for_lists(q, X, list_ids, block_sizes, topk_per_layer):
    qn = q / np.linalg.norm(q)
    scores = np.zeros(len(list_ids), np.float32)

    for B in block_sizes:
        layer = np.full(len(list_ids), -1.0, np.float32)
        for i, ids in enumerate(list_ids):
            if len(ids) == 0:
                continue
            sample = ids[:min(len(ids), B*topk_per_layer)]
            sims = X[sample] @ qn
            layer[i] = sims.max()

        valid = layer > -0.5
        mu = layer[valid].mean()
        sd = max(layer[valid].std(), 1e-6)
        scores += (layer - mu) / sd

    return scores

# ---------------------------
# Brute truth
# ---------------------------
def brute_topk(X, q, k):
    sims = X @ q
    return np.argsort(-sims)[:k]

def recall(found, truth):
    return len(set(found) & set(truth)) / len(truth)

# ============================================================
# MAIN
# ============================================================
def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--q", type=int, default=500)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("\nBuilding dataset...")

    # Hard clustered dataset
    n_topics = 128
    topic_strength = 1.0
    db_noise = 1.0
    q_noise = 1.6

    topics = l2_normalize(rng.standard_normal((n_topics, args.d)).astype(np.float32))

    topic_ids_db = rng.integers(0, n_topics, size=args.n)
    X_fp = topic_strength * topics[topic_ids_db] + db_noise * rng.standard_normal((args.n, args.d)).astype(np.float32)

    topic_ids_q = rng.integers(0, n_topics, size=args.q)
    Q_fp = topic_strength * topics[topic_ids_q] + q_noise * rng.standard_normal((args.q, args.d)).astype(np.float32)

    X = dtdr_hadamard(l2_normalize(X_fp))
    Q = dtdr_hadamard(l2_normalize(Q_fp))

    print("Building IVF...")
    C = kmeans_simple(X, args.nlist)
    labels = ivf_assign(X, C)
    list_ids = [np.where(labels == i)[0] for i in range(args.nlist)]

    print("Building HNSW per list...")
    hnsw = {}
    for i, ids in enumerate(list_ids):
        if len(ids) == 0:
            continue
        idx = hnswlib.Index(space='cosine', dim=args.d)
        idx.init_index(len(ids), M=16, ef_construction=200)
        idx.add_items(X[ids], ids)
        idx.set_ef(128)
        hnsw[i] = idx

    k_values = [10, 20, 40, 80, 160, 320]
    results = []

    print("\nRunning sweep...\n")
    print(f"{'k_local':>8} {'recall@10':>10} {'lists':>8} {'candidates':>12}")

    for K_LOCAL in k_values:

        recalls = []
        probed = []
        cand_counts = []

        for qi in range(args.q):

            qv = Q[qi]
            truth = brute_topk(X, qv, 10)

            ranked = np.argsort(-dilution_evidence_for_lists(qv, X, list_ids, [64,256,1024], 16))
            probe = ranked[:2]  # fixed routing observation

            probed.append(len(probe))

            candidates = []
            for li in probe:
                ids = list_ids[li]
                if len(ids) == 0 or li not in hnsw:
                    continue

                k_local = min(K_LOCAL, len(ids))
                if k_local <= 0:
                    continue

                # robust HNSW query (reduces k until query succeeds)
                k_try = k_local
                while k_try > 0:
                    try:
                        found, _ = hnsw[li].knn_query(qv, k=k_try)
                        candidates.extend(found[0])
                        break
                    except RuntimeError:
                        k_try //= 2
                        break
                

            cand_counts.append(len(candidates))

            if len(candidates) == 0:
                recalls.append(0)
                continue

            candidates = np.unique(candidates)
            sims = X[candidates] @ qv
            found = candidates[np.argsort(-sims)[:10]]

            recalls.append(recall(found, truth))

        r = float(np.mean(recalls))
        l = float(np.mean(probed))
        c = float(np.mean(cand_counts))

        print(f"{K_LOCAL:8d} {r:10.4f} {l:8.2f} {c:12.1f}")

        results.append((K_LOCAL, r, l, c))

    with open("klocal_sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_local","recall@10","mean_lists","mean_candidates"])
        writer.writerows(results)

    print("\nSaved klocal_sweep_results.csv")

if __name__ == "__main__":
    main()