#!/usr/bin/env python
"""
DTDR_RAG_double_transform_demo.py
---------------------------------
A clean, self-contained DTDR RAG retrieval demonstration intended to run in a local .venv.

What it shows:
  1) "DTDR as a compute-capable representation": retrieval scores are computed directly in DTDR space
     (transform-domain int8 coefficients + per-block scales), without reconstructing float embeddings.
  2) "Double / composite DTDR": composing orthogonal transforms (Hadamard then DCT) still supports
     high-quality retrieval.
  3) A negative control: "no-transform INT8" (raw int8 quantization without an orthogonal transform)
     typically degrades retrieval and robustness more.

Dependencies:
  - numpy
  - sentence-transformers
Optional:
  - scipy (NOT required): if installed, we will use scipy's DCT for speed; otherwise we use a small
    orthonormal DCT implementation in numpy.

Run:
  python DTDR_RAG_double_transform_demo.py

Notes:
  - Downloads a few Project Gutenberg books on first run into ./books/
  - Uses a lightweight sentence-transformer by default.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.fft import dct as scipy_dct  # type: ignore
    _HAVE_SCIPY_DCT = True
except Exception:
    _HAVE_SCIPY_DCT = False

from sentence_transformers import SentenceTransformer


# ----------------------------- Reproducibility -----------------------------

SEED = 123
random.seed(SEED)
np.random.seed(SEED)

BOOKS_DIR = "books"

GUTENBERG: List[Tuple[str, str]] = [
    ("alice_in_wonderland.txt", "https://www.gutenberg.org/cache/epub/11/pg11.txt"),
    ("pride_and_prejudice.txt", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("frankenstein.txt", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("moby_dick.txt", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("sherlock_holmes.txt", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
    ("dracula.txt", "https://www.gutenberg.org/cache/epub/345/pg345.txt"),
]


# ----------------------------- Data handling ------------------------------

def download(path: str, url: str, timeout: int = 40) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    print(f"[download] {os.path.basename(path)}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)


def read_text_file(path: str) -> str:
    # Gutenberg is usually UTF-8 but sometimes has oddities; be permissive.
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def clean_gutenberg(text: str) -> str:
    """
    Roughly strip Gutenberg headers/footers. This doesn't need to be perfect.
    """
    # Remove CRs
    text = text.replace("\r", "")
    # Try typical markers
    start_markers = [
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]
    end_markers = [
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]
    for pat in start_markers:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = text[m.end():]
            break
    for pat in end_markers:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = text[:m.start()]
            break

    # Whitespace normalize
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150, min_len: int = 250) -> List[str]:
    chunks: List[str] = []
    step = max(1, chunk_chars - overlap)
    for i in range(0, len(text), step):
        ch = text[i:i + chunk_chars].strip()
        if len(ch) >= min_len:
            chunks.append(ch)
    return chunks


@dataclass
class Passage:
    book: str
    idx: int
    text: str


def build_passages(chunk_chars: int, overlap: int, per_book_cap: int | None) -> Tuple[List[Passage], List[str]]:
    os.makedirs(BOOKS_DIR, exist_ok=True)
    passages: List[Passage] = []
    books: List[str] = []
    for fname, url in GUTENBERG:
        path = os.path.join(BOOKS_DIR, fname)
        download(path, url)
        raw = read_text_file(path)
        clean = clean_gutenberg(raw)
        chunks = chunk_text(clean, chunk_chars=chunk_chars, overlap=overlap)
        if per_book_cap is not None:
            chunks = chunks[: int(per_book_cap)]
        for j, ch in enumerate(chunks):
            passages.append(Passage(book=fname, idx=j, text=ch))
            books.append(fname)
        print(f"[chunk] {fname}: {len(chunks)} chunks")
    return passages, books


# ----------------------------- Transforms ---------------------------------

def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def fwht_inplace(x: np.ndarray) -> None:
    """
    In-place Fast Walsh–Hadamard Transform (FWHT) for length power-of-two vectors.
    Produces an unnormalised Hadamard transform; we normalise separately so it's orthonormal.
    """
    h = 1
    n = x.shape[0]
    while h < n:
        for i in range(0, n, h * 2):
            a = x[i:i + h]
            b = x[i + h:i + 2 * h]
            x[i:i + h] = a + b
            x[i + h:i + 2 * h] = a - b
        h *= 2


def hadamard_orthonormal_vec(v: np.ndarray) -> np.ndarray:
    out = v.astype(np.float32, copy=True)
    fwht_inplace(out)
    out *= (1.0 / math.sqrt(out.shape[0]))
    return out


def hadamard_orthonormal_mat(X: np.ndarray) -> np.ndarray:
    # Apply FWHT row-wise
    out = X.astype(np.float32, copy=True)
    n = out.shape[1]
    scale = 1.0 / math.sqrt(n)
    for i in range(out.shape[0]):
        fwht_inplace(out[i])
        out[i] *= scale
    return out


def dct_orthonormal_vec(v: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY_DCT:
        return scipy_dct(v, type=2, norm="ortho").astype(np.float32)
    # Numpy fallback: orthonormal DCT-II via matrix multiply (fine for D<=1024)
    n = v.shape[0]
    k = np.arange(n, dtype=np.float32)[:, None]
    m = np.arange(n, dtype=np.float32)[None, :]
    C = np.cos(math.pi / n * (m + 0.5) * k)
    # Orthonormal scaling: first row differs
    C[0, :] *= math.sqrt(1.0 / n)
    C[1:, :] *= math.sqrt(2.0 / n)
    return (C @ v.astype(np.float32)).astype(np.float32)


def dct_orthonormal_mat(X: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY_DCT:
        return scipy_dct(X, type=2, norm="ortho", axis=1).astype(np.float32)
    n = X.shape[1]
    k = np.arange(n, dtype=np.float32)[:, None]
    m = np.arange(n, dtype=np.float32)[None, :]
    C = np.cos(math.pi / n * (m + 0.5) * k)
    C[0, :] *= math.sqrt(1.0 / n)
    C[1:, :] *= math.sqrt(2.0 / n)
    return (X.astype(np.float32) @ C.T).astype(np.float32)


def apply_transforms_vec(v: np.ndarray, transforms: Tuple[str, ...]) -> np.ndarray:
    out = v
    for t in transforms:
        if t == "hadamard":
            out = hadamard_orthonormal_vec(out)
        elif t == "dct":
            out = dct_orthonormal_vec(out)
        else:
            raise ValueError(f"Unknown transform: {t}")
    return out


def apply_transforms_mat(X: np.ndarray, transforms: Tuple[str, ...]) -> np.ndarray:
    out = X
    for t in transforms:
        if t == "hadamard":
            out = hadamard_orthonormal_mat(out)
        elif t == "dct":
            out = dct_orthonormal_mat(out)
        else:
            raise ValueError(f"Unknown transform: {t}")
    return out


# ----------------------------- DTDR build ---------------------------------

def quantize_blockwise(X: np.ndarray, block: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-row, per-block symmetric int8 quantization:
      q = round(x / s), with s = max(abs(block))/127.

    Returns:
      q: int8 array shape [N, D]
      scales: float32 array shape [N, nblocks]
    """
    assert X.dtype in (np.float32, np.float64)
    N, D = X.shape
    nblocks = (D + block - 1) // block
    qmax = 127

    q = np.zeros((N, D), dtype=np.int8)
    scales = np.zeros((N, nblocks), dtype=np.float32)

    for b in range(nblocks):
        j0, j1 = b * block, min(D, (b + 1) * block)
        chunk = X[:, j0:j1]
        amax = np.max(np.abs(chunk), axis=1) + 1e-12
        s = amax / qmax
        scales[:, b] = s.astype(np.float32)
        q[:, j0:j1] = np.clip(np.round(chunk / s[:, None]), -qmax, qmax).astype(np.int8)

    return q, scales


def build_dtdr(emb: np.ndarray, transforms: Tuple[str, ...], block: int = 16) -> Dict[str, object]:
    """
    Build DTDR representation for passage embeddings.

    Inputs:
      emb: float32 [N, d_in] normalised embeddings (unit vectors)
      transforms: e.g. ("hadamard",) or ("hadamard","dct")
    """
    t0 = time.time()
    N, d_in = emb.shape
    d_h = next_pow2(d_in)

    # pad to power-of-two for hadamard/fwht
    X = np.zeros((N, d_h), dtype=np.float32)
    X[:, :d_in] = emb.astype(np.float32)

    # transform(s)
    X_t = apply_transforms_mat(X, transforms)

    # int8 + scales
    q, s = quantize_blockwise(X_t, block=block)

    dt = time.time() - t0
    print(f"[build] DTDR build complete in {dt:.2f}s  (transforms={transforms}, d_in={d_in}, d_h={d_h}, block={block})")
    return {"q": q, "scales": s, "d_in": d_in, "d_h": d_h, "transforms": transforms, "block": block}


def dt_scores(query_pad: np.ndarray, rep: Dict[str, object]) -> np.ndarray:
    """
    Compute similarity scores in DTDR domain:
      score_i = sum_b ( <q_i_block, q_query_block_float> * scale_i_block )

    query_pad is float32 padded to d_h.
    """
    qcoef = rep["q"]  # type: ignore
    scales = rep["scales"]  # type: ignore
    transforms = rep["transforms"]  # type: ignore
    block = int(rep["block"])  # type: ignore

    v = apply_transforms_vec(query_pad.astype(np.float32), transforms)

    qf = qcoef.astype(np.float32)
    sf = scales.astype(np.float32)

    out = np.zeros((qf.shape[0],), dtype=np.float32)
    nblocks = sf.shape[1]
    for b in range(nblocks):
        j0, j1 = b * block, min(qf.shape[1], (b + 1) * block)
        out += (qf[:, j0:j1] * v[j0:j1]).sum(axis=1) * sf[:, b]
    return out


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


# ----------------------------- Evaluation ---------------------------------

RAG_QUERIES: List[Tuple[str, str, List[str]]] = [
    ("Who is the White Rabbit and what is he doing?", "alice_in_wonderland.txt", ["white rabbit", "rabbit"]),
    ("What does Elizabeth think of Mr. Darcy early on?", "pride_and_prejudice.txt", ["darcy", "elizabeth"]),
    ("Who created the creature and what was the consequence?", "frankenstein.txt", ["frankenstein", "creature"]),
    ("What is Captain Ahab obsessed with?", "moby_dick.txt", ["ahab", "whale"]),
    ("What is Sherlock Holmes known for in solving mysteries?", "sherlock_holmes.txt", ["holmes", "watson"]),
    ("Who is Count Dracula and what is his nature?", "dracula.txt", ["dracula", "count"]),
]


def float_scores(query_emb: np.ndarray, emb: np.ndarray) -> np.ndarray:
    # emb is already normalised; query_emb is normalised
    return (emb @ query_emb.astype(np.float32)).astype(np.float32)


def eval_rag(rep: Dict[str, object], emb: np.ndarray, books: List[str], norm_texts: List[str],
             model: SentenceTransformer, K: int = 8) -> Dict[str, float]:
    overlaps, bookhits, anchorhits = [], [], []
    d_in = int(rep["d_in"])  # type: ignore
    d_h = int(rep["d_h"])    # type: ignore

    for q, exp_book, anchors in RAG_QUERIES:
        qemb = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)

        base = topk(float_scores(qemb, emb), K)

        qpad = np.zeros((d_h,), dtype=np.float32)
        qpad[:d_in] = qemb
        dt = topk(dt_scores(qpad, rep), K)

        overlaps.append(len(set(base.tolist()).intersection(set(dt.tolist()))) / len(base))
        bookhits.append(int(any(books[i] == exp_book for i in dt)))
        anchors_l = [a.lower() for a in anchors]
        anchorhits.append(int(any(any(a in norm_texts[i] for a in anchors_l) for i in dt)))

    return {
        "K": float(K),
        "mean_overlap_vs_float": float(np.mean(overlaps)),
        "book_hit_rate@K": float(np.mean(bookhits)),
        "anchor_hit_rate@K": float(np.mean(anchorhits)),
    }


# ----------------------------- Corruption ---------------------------------

def corrupt_dropout(q: np.ndarray, drop_frac: float, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = q.copy()
    mask = rng.random(out.shape) < float(drop_frac)
    out[mask] = 0
    return out


def corrupt_block_loss(q: np.ndarray, block: int, block_frac: float, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = q.copy()
    Nn, D = out.shape
    nblocks = (D + block - 1) // block
    if block_frac <= 0:
        return out
    n_drop = max(1, int(math.ceil(float(block_frac) * nblocks)))
    drop_blocks = rng.choice(nblocks, size=min(nblocks, n_drop), replace=False)
    for b in drop_blocks:
        j0, j1 = b * block, min(D, (b + 1) * block)
        out[:, j0:j1] = 0
    return out


# ----------------------------- No-transform INT8 (negative control) -------

def quantize_blockwise_raw(X: np.ndarray, block: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    return quantize_blockwise(X.astype(np.float32), block=block)


def scores_quantized_no_transform(qpad: np.ndarray, qcoef: np.ndarray, scales: np.ndarray, block: int) -> np.ndarray:
    qf = qcoef.astype(np.float32)
    sf = scales.astype(np.float32)
    out = np.zeros((qf.shape[0],), dtype=np.float32)
    nblocks = sf.shape[1]
    for b in range(nblocks):
        j0, j1 = b * block, min(qf.shape[1], (b + 1) * block)
        out += (qf[:, j0:j1] * qpad[j0:j1]).sum(axis=1) * sf[:, b]
    return out


def eval_rag_no_transform(qcoef: np.ndarray, scales: np.ndarray,
                          emb: np.ndarray, d_in: int, d_h: int,
                          books: List[str], norm_texts: List[str],
                          model: SentenceTransformer, K: int = 8, block: int = 16) -> Dict[str, float]:
    overlaps, bookhits, anchorhits = [], [], []
    for q, exp_book, anchors in RAG_QUERIES:
        qemb = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)
        base = topk(float_scores(qemb, emb), K)

        qpad = np.zeros((d_h,), dtype=np.float32)
        qpad[:d_in] = qemb
        dt = topk(scores_quantized_no_transform(qpad, qcoef, scales, block), K)

        overlaps.append(len(set(base.tolist()).intersection(set(dt.tolist()))) / len(base))
        bookhits.append(int(any(books[i] == exp_book for i in dt)))
        anchors_l = [a.lower() for a in anchors]
        anchorhits.append(int(any(any(a in norm_texts[i] for a in anchors_l) for i in dt)))

    return {
        "K": float(K),
        "mean_overlap_vs_float": float(np.mean(overlaps)),
        "book_hit_rate@K": float(np.mean(bookhits)),
        "anchor_hit_rate@K": float(np.mean(anchorhits)),
    }


# ----------------------------- Example printing ---------------------------

def show_example(question: str, expected_book: str, anchors: List[str],
                 rep: Dict[str, object], passages: List[Passage],
                 model: SentenceTransformer, K: int = 5) -> None:
    d_in = int(rep["d_in"])  # type: ignore
    d_h = int(rep["d_h"])    # type: ignore
    qemb = model.encode([question], normalize_embeddings=True)[0].astype(np.float32)
    qpad = np.zeros((d_h,), dtype=np.float32)
    qpad[:d_in] = qemb
    idx = topk(dt_scores(qpad, rep), K)

    print("\n" + "=" * 92)
    print(f"DTDR example (transforms={rep['transforms']})")
    print("QUESTION:", question)
    print("Expected:", expected_book, "| Anchors:", anchors)
    print("-" * 92)
    for r, i in enumerate(idx, 1):
        ps = passages[int(i)]
        print(f"{r:>2}. {ps.book} [chunk {ps.idx}]")
        print(textwrap.shorten(ps.text.replace("\n", " "), width=260, placeholder=" …"))
        print()

    context = "\n\n".join([passages[int(i)].text for i in idx[: min(3, len(idx))]])
    print("RAG PROMPT (example):")
    print("-" * 92)
    print("Question:", question)
    print("Context:\n", textwrap.shorten(context.replace("\n", " "), width=650, placeholder=" …"))


# ----------------------------- Main ---------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--chunk-chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--per-book-cap", type=int, default=140, help="Cap chunks per book for speed (set 0 to disable)")
    ap.add_argument("--block", type=int, default=16, help="Quantization block size")
    ap.add_argument("--K", type=int, default=8, help="Top-K for metrics")
    args = ap.parse_args()

    per_book_cap = None if int(args.per_book_cap) <= 0 else int(args.per_book_cap)

    # Build corpus
    passages, books = build_passages(args.chunk_chars, args.overlap, per_book_cap)
    texts = [p.text for p in passages]
    norm_texts = [re.sub(r"\s+", " ", t.lower()) for t in texts]

    # Embed
    print(f"[embed] Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    t0 = time.time()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    print(f"[embed] Done: emb={emb.shape} in {time.time() - t0:.2f}s")

    # DTDR builds
    BLOCK = int(args.block)
    single = build_dtdr(emb, transforms=("hadamard",), block=BLOCK)
    double = build_dtdr(emb, transforms=("hadamard", "dct"), block=BLOCK)
    if not _HAVE_SCIPY_DCT:
        print("[note] scipy not found: using numpy DCT fallback (slower, but fine for this demo).")

    # Clean metrics
    print("\nRAG metrics (clean) — SINGLE Hadamard DTDR:")
    print(eval_rag(single, emb, books, norm_texts, model, K=args.K))

    print("\nRAG metrics (clean) — DOUBLE (Hadamard + DCT) DTDR:")
    print(eval_rag(double, emb, books, norm_texts, model, K=args.K))

    # Corruption sweeps
    levels = [0.0, 0.01, 0.05, 0.10, 0.20]
    print("\nCorruption sweep (dropout) — SINGLE vs DOUBLE:")
    for lvl in levels:
        s_rep = dict(single)
        d_rep = dict(double)
        s_rep["q"] = corrupt_dropout(single["q"], lvl, seed=SEED)  # type: ignore
        d_rep["q"] = corrupt_dropout(double["q"], lvl, seed=SEED)  # type: ignore
        print(lvl,
              "single", eval_rag(s_rep, emb, books, norm_texts, model, K=args.K),
              "double", eval_rag(d_rep, emb, books, norm_texts, model, K=args.K))

    print("\nCorruption sweep (block loss) — SINGLE vs DOUBLE:")
    for lvl in levels:
        s_rep = dict(single)
        d_rep = dict(double)
        s_rep["q"] = corrupt_block_loss(single["q"], BLOCK, lvl, seed=SEED)  # type: ignore
        d_rep["q"] = corrupt_block_loss(double["q"], BLOCK, lvl, seed=SEED)  # type: ignore
        print(lvl,
              "single", eval_rag(s_rep, emb, books, norm_texts, model, K=args.K),
              "double", eval_rag(d_rep, emb, books, norm_texts, model, K=args.K))

    # Negative control: no-transform INT8
    d_in = int(single["d_in"])  # type: ignore
    d_h = int(single["d_h"])    # type: ignore
    X_raw = np.zeros((emb.shape[0], d_h), dtype=np.float32)
    X_raw[:, :d_in] = emb.astype(np.float32)
    q_raw, s_raw = quantize_blockwise_raw(X_raw, block=BLOCK)
    print(f"\n[ablation] NO-TRANSFORM INT8 built: q={q_raw.shape} scales={s_raw.shape} blocks={s_raw.shape[1]}")

    print("\nRAG metrics (clean) — No-transform INT8:")
    print(eval_rag_no_transform(q_raw, s_raw, emb, d_in, d_h, books, norm_texts, model, K=args.K, block=BLOCK))

    print("\nCorruption sweep (dropout) — No-transform INT8:")
    for lvl in levels:
        qr = corrupt_dropout(q_raw, lvl, seed=SEED)
        print(lvl, eval_rag_no_transform(qr, s_raw, emb, d_in, d_h, books, norm_texts, model, K=args.K, block=BLOCK))

    print("\nCorruption sweep (block loss) — No-transform INT8:")
    for lvl in levels:
        qr = corrupt_block_loss(q_raw, BLOCK, lvl, seed=SEED)
        print(lvl, eval_rag_no_transform(qr, s_raw, emb, d_in, d_h, books, norm_texts, model, K=args.K, block=BLOCK))

    # Show one example prompt for single and double
    q0, exp0, anc0 = RAG_QUERIES[0]
    show_example(q0, exp0, anc0, single, passages, model, K=5)
    show_example(q0, exp0, anc0, double, passages, model, K=5)

    print("\nDone.")


if __name__ == "__main__":
    main()
