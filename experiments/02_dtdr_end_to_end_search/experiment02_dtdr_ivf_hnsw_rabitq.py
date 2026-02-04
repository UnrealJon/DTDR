import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import hnswlib
except ImportError as e:
    raise SystemExit(
        "Missing dependency: hnswlib. Install with: pip install hnswlib"
    ) from e


# -----------------------------
# DTDR: fast Walsh–Hadamard
# -----------------------------
def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fwht_inplace(x: np.ndarray) -> None:
    """
    In-place Fast Walsh–Hadamard Transform (FWHT) on last dimension.
    x: float32/float64 array shape (..., d) where d is power of two
    """
    d = x.shape[-1]
    if not _is_power_of_two(d):
        raise ValueError(f"FWHT requires power-of-two dim, got d={d}")

    h = 1
    while h < d:
        # reshape into blocks of size 2h: (..., d/(2h), 2h)
        x_ = x.reshape(*x.shape[:-1], -1, 2 * h)
        a = x_[..., :h]
        b = x_[..., h:2 * h]
        x_[..., :h] = a + b
        x_[..., h:2 * h] = a - b
        h *= 2


def dtdr_hadamard(x: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    DTDR transform using Hadamard (orthogonal up to scaling).
    Returns float32 transformed vectors.
    """
    y = x.astype(np.float32, copy=True)
    fwht_inplace(y)
    if normalize:
        y /= math.sqrt(y.shape[-1])
    return y


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


# -----------------------------
# "RaBitQ-like" 1-bit coding
# (random hyperplanes + Hamming)
# -----------------------------
@dataclass
class BinaryCoder:
    R: np.ndarray  # shape (d, nbits), float32
    nbits: int

    @staticmethod
    def make(d: int, nbits: int, seed: int = 0) -> "BinaryCoder":
        rng = np.random.default_rng(seed)
        # random gaussian hyperplanes
        R = rng.standard_normal((d, nbits), dtype=np.float32)
        return BinaryCoder(R=R, nbits=nbits)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode float32 X (N, d) into packed bits (N, nbytes).
        """
        proj = (X @ self.R)  # (N, nbits)
        bits = (proj >= 0).astype(np.uint8)  # 0/1
        # pack bits into bytes
        nbytes = (self.nbits + 7) // 8
        out = np.zeros((X.shape[0], nbytes), dtype=np.uint8)
        for b in range(self.nbits):
            out[:, b // 8] |= (bits[:, b] << (b % 8))
        return out

    @staticmethod
    def hamming_distance_packed(a: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Hamming distance between one packed code a (nbytes,)
        and many packed codes B (M, nbytes). Returns int32 (M,).
        """
        # XOR then popcount
        x = np.bitwise_xor(B, a[None, :])
        # popcount per byte via lookup
        lut = _POPCOUNT_LUT
        return lut[x].sum(axis=1).astype(np.int32)


_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# -----------------------------
# IVF (k-means) in DTDR domain
# -----------------------------
def kmeans_simple(X: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    """
    Lightweight k-means (L2) returning centroids.
    X: (N, d) float32
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape

    # kmeans++-ish init (cheap)
    centroids = np.empty((k, d), dtype=np.float32)
    centroids[0] = X[rng.integers(0, N)]
    dist = np.full((N,), np.inf, dtype=np.float32)
    for i in range(1, k):
        # update distance to nearest centroid
        diff = X - centroids[i - 1][None, :]
        dist = np.minimum(dist, (diff * diff).sum(axis=1))
        probs = dist / np.maximum(dist.sum(), 1e-12)
        centroids[i] = X[rng.choice(N, p=probs)]

    # Lloyd iterations
    for _ in range(iters):
        # assign
        # (N, k) distances would be big; do chunked
        labels = np.empty((N,), dtype=np.int32)
        for start in range(0, N, 8192):
            end = min(N, start + 8192)
            chunk = X[start:end]
            # compute (chunk, k)
            # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
            x2 = (chunk * chunk).sum(axis=1, keepdims=True)
            c2 = (centroids * centroids).sum(axis=1)[None, :]
            dots = chunk @ centroids.T
            d2 = x2 + c2 - 2.0 * dots
            labels[start:end] = np.argmin(d2, axis=1).astype(np.int32)

        # update
        newC = np.zeros_like(centroids)
        counts = np.zeros((k,), dtype=np.int32)
        np.add.at(newC, labels, X)
        np.add.at(counts, labels, 1)
        for j in range(k):
            if counts[j] > 0:
                newC[j] /= counts[j]
            else:
                newC[j] = X[rng.integers(0, N)]
        centroids = newC

    return centroids


def ivf_assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each vector to nearest centroid (L2).
    """
    N = X.shape[0]
    k = centroids.shape[0]
    labels = np.empty((N,), dtype=np.int32)
    for start in range(0, N, 8192):
        end = min(N, start + 8192)
        chunk = X[start:end]
        x2 = (chunk * chunk).sum(axis=1, keepdims=True)
        c2 = (centroids * centroids).sum(axis=1)[None, :]
        dots = chunk @ centroids.T
        d2 = x2 + c2 - 2.0 * dots
        labels[start:end] = np.argmin(d2, axis=1).astype(np.int32)
    return labels


def top_nprobe_lists(q: np.ndarray, centroids: np.ndarray, nprobe: int) -> np.ndarray:
    """
    Choose nprobe nearest IVF centroids for query q (L2).
    """
    # same L2 trick
    q2 = float((q * q).sum())
    c2 = (centroids * centroids).sum(axis=1)
    dots = centroids @ q
    d2 = q2 + c2 - 2.0 * dots
    return np.argsort(d2)[:nprobe].astype(np.int32)


# -----------------------------
# DTDR-domain "dilution ladder"
# (optional: coarse list scoring)
# -----------------------------
def dilution_evidence_for_lists(
    q: np.ndarray,
    X: np.ndarray,
    list_ids: List[np.ndarray],
    block_sizes: List[int],
    topk_per_layer: int = 32,
) -> np.ndarray:
    """
    Simple, practical evidence score per IVF list:
    For each list, concatenate in blocks of B and take max cosine over slots.
    Returns evidence scores array (nlist,).
    """
    nlist = len(list_ids)
    qn = q / max(np.linalg.norm(q), 1e-12)

    evidence = np.zeros((nlist,), dtype=np.float32)

    for B in block_sizes:
        layer_scores = np.full((nlist,), -1.0, dtype=np.float32)
        for li in range(nlist):
            ids = list_ids[li]
            if ids.size == 0:
                continue
            # sample at most B*topk_per_layer vectors to keep it cheap
            take = min(ids.size, B * topk_per_layer)
            sample = ids[:take]
            # compute cosine scores for each sampled vector against q
            V = X[sample]
            Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
            s = (Vn @ qn).max()
            layer_scores[li] = s

        # z-score layer and add
        mu = float(layer_scores[layer_scores > -0.5].mean()) if np.any(layer_scores > -0.5) else 0.0
        sd = float(layer_scores[layer_scores > -0.5].std()) if np.any(layer_scores > -0.5) else 1.0
        sd = max(sd, 1e-6)
        z = (layer_scores - mu) / sd
        evidence += z

    return evidence


# -----------------------------
# Ground truth / evaluation
# -----------------------------
def brute_force_topk_cosine(X: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    qn = q / max(np.linalg.norm(q), 1e-12)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sims = Xn @ qn
    return np.argsort(-sims)[:k].astype(np.int32)


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    return float(len(set(found.tolist()) & set(truth.tolist()))) / float(len(truth))


# -----------------------------
# Main experiment
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="database size")
    ap.add_argument("--q", type=int, default=200, help="number of queries")
    ap.add_argument("--d", type=int, default=256, help="dimension (power of two for Hadamard)")
    ap.add_argument("--nlist", type=int, default=256, help="IVF lists")
    ap.add_argument("--nprobe", type=int, default=8, help="lists probed per query")
    ap.add_argument("--ef", type=int, default=128, help="HNSW efSearch")
    ap.add_argument("--M", type=int, default=16, help="HNSW M")
    ap.add_argument("--topk", type=int, default=10, help="top-k retrieval")
    ap.add_argument("--cand", type=int, default=2000, help="candidate pool size before rerank")
    ap.add_argument("--nbits", type=int, default=256, help="binary code bits for RaBitQ-like rerank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_dilution_evidence", action="store_true",
                    help="use DTDR dilution ladder to rank IVF lists before nprobe")
    args = ap.parse_args()

    if not _is_power_of_two(args.d):
        raise SystemExit("d must be a power of two for Hadamard DTDR (e.g., 128, 256, 512).")

    rng = np.random.default_rng(args.seed)

    # 1) Make synthetic dataset (swap this out for real embeddings later)
    X_fp = rng.standard_normal((args.n, args.d), dtype=np.float32)
    Q_fp = rng.standard_normal((args.q, args.d), dtype=np.float32)

    # 2) DTDR transform (domain for EVERYTHING)
    t0 = time.time()
    X = dtdr_hadamard(X_fp, normalize=True)
    Q = dtdr_hadamard(Q_fp, normalize=True)
    X = l2_normalize(X)
    Q = l2_normalize(Q)
    t_dtdr = time.time() - t0

    # 3) Build IVF in DTDR domain
    t0 = time.time()
    centroids = kmeans_simple(X, k=args.nlist, iters=10, seed=args.seed)
    labels = ivf_assign(X, centroids)
    list_ids: List[np.ndarray] = []
    for li in range(args.nlist):
        ids = np.where(labels == li)[0].astype(np.int32)
        list_ids.append(ids)
    t_ivf = time.time() - t0

    # 4) Build HNSW per list (DTDR domain) + store binary codes per list
    #    This keeps the demo simple and clearly "IVF + HNSW".
    coder = BinaryCoder.make(d=args.d, nbits=args.nbits, seed=args.seed)

    hnsw_by_list: Dict[int, hnswlib.Index] = {}
    codes_by_list: Dict[int, np.ndarray] = {}
    ids_by_list: Dict[int, np.ndarray] = {}

    t0 = time.time()
    for li in range(args.nlist):
        ids = list_ids[li]
        if ids.size == 0:
            continue
        vecs = X[ids]

        idx = hnswlib.Index(space="cosine", dim=args.d)
        idx.init_index(max_elements=ids.size, ef_construction=200, M=args.M)
        idx.add_items(vecs, ids)
        idx.set_ef(args.ef)

        hnsw_by_list[li] = idx
        ids_by_list[li] = ids

        # Precompute binary codes for RaBitQ-like reranking
        codes_by_list[li] = coder.encode(vecs)
    t_hnsw = time.time() - t0

    # 5) Evaluate queries
    recalls = []
    t_query_total = 0.0

    for qi in range(args.q):
        q = Q[qi]

        # ground truth entirely in DTDR domain (brute force)
        gt = brute_force_topk_cosine(X, q, args.topk)

        # Choose lists to probe
        if args.use_dilution_evidence:
            # Evidence ranks lists; then take top nprobe
            evidence = dilution_evidence_for_lists(
                q=q,
                X=X,
                list_ids=list_ids,
                block_sizes=[64, 256, 1024],
                topk_per_layer=16,
            )
            probe_lists = np.argsort(-evidence)[:args.nprobe].astype(np.int32)
        else:
            # vanilla IVF: nearest centroids (still DTDR domain)
            probe_lists = top_nprobe_lists(q, centroids, args.nprobe)

        # ANN retrieval in DTDR domain: gather candidates via HNSW across probed lists
        t0 = time.time()
        cand_ids: List[int] = []
        cand_scores: List[float] = []

        per_list_budget = max(10, args.cand // max(1, len(probe_lists)))
        for li in probe_lists:
            if int(li) not in hnsw_by_list:
                continue
            idx = hnsw_by_list[int(li)]
            # query k=per_list_budget candidates
            ids, dists = idx.knn_query(q, k=min(per_list_budget, ids_by_list[int(li)].size))
            ids = ids.reshape(-1).astype(np.int32)
            dists = dists.reshape(-1).astype(np.float32)
            # cosine space in hnswlib: "distance" = 1 - cosine_similarity
            sims = 1.0 - dists

            cand_ids.extend(ids.tolist())
            cand_scores.extend(sims.tolist())

        # de-dup
        if len(cand_ids) == 0:
            recalls.append(0.0)
            t_query_total += (time.time() - t0)
            continue

        unique = {}
        for i, cid in enumerate(cand_ids):
            s = cand_scores[i]
            if cid not in unique or s > unique[cid]:
                unique[cid] = s

        cand_ids_u = np.array(list(unique.keys()), dtype=np.int32)

        # RaBitQ-like rerank: Hamming distance in DTDR domain
        q_code = coder.encode(q[None, :])[0]  # packed
        # Need codes for candidates; fetch by list membership:
        # easiest is direct encode candidates (still DTDR domain, small set)
        cand_vecs = X[cand_ids_u]
        cand_codes = coder.encode(cand_vecs)
        ham = BinaryCoder.hamming_distance_packed(q_code, cand_codes)

        # Take best by smallest Hamming distance, then final exact cosine rerank on that shortlist (DTDR domain)
        # (This mirrors "RaBitQ estimate -> refine")
        shortlist = cand_ids_u[np.argsort(ham)[:min(args.topk * 20, cand_ids_u.size)]]
        # final exact DTDR cosine rerank
        final = shortlist[np.argsort(-(X[shortlist] @ q))[:args.topk]]

        t_query_total += (time.time() - t0)

        rec = recall_at_k(final, gt)
        recalls.append(rec)

    print("============================================================")
    print("Experiment 02: DTDR-native IVF + HNSW + RaBitQ-like rerank")
    print("============================================================")
    print(f"N={args.n}, Q={args.q}, d={args.d}")
    print(f"nlist={args.nlist}, nprobe={args.nprobe}, HNSW(M={args.M}, ef={args.ef}), nbits={args.nbits}")
    print(f"Use dilution evidence: {args.use_dilution_evidence}")
    print("------------------------------------------------------------")
    print(f"DTDR transform time: {t_dtdr:.3f}s")
    print(f"IVF build time:      {t_ivf:.3f}s")
    print(f"HNSW build time:     {t_hnsw:.3f}s (per-list indices)")
    print(f"Mean query time:     {t_query_total / max(1, args.q):.6f}s")
    print(f"Recall@{args.topk}:          {np.mean(recalls):.4f}")
    print("============================================================")


if __name__ == "__main__":
    main()
