# DTDR Spectral Trajectory ANN

Distributed Transform-Domain Representation (DTDR) enables a hierarchical routing signal inside approximate nearest-neighbour (ANN) indices. This repository demonstrates that signal on structured embeddings and the SIFT1M benchmark.

The goal is not to replace IVF or HNSW, but to show that DTDR's internal transform-domain structure enables principled trajectory routing within IVF lists, reducing candidate evaluations while maintaining recall.

---

## Key Result

On SIFT1M (1,000,000 vectors, 10,000 queries):

| Method | Candidates evaluated | Recall@10 |
|---|---|---|
| Flat IVF1024, nprobe=8 (published baseline) | ~7,812 | ~0.57 |
| Trajectory router, nprobe=8 | 899 | 0.580 |
| Trajectory router, nprobe=16 | 1,796 | 0.657 |
| Trajectory router, nprobe=32 | 3,577 | 0.698 |

At nprobe=8: **8.7× fewer candidate evaluations** at equivalent recall to the flat IVF baseline.

Candidate counts scale predictably as approximately:

```
nprobe × 112
```

making the system deterministic — an operational advantage in latency-sensitive environments where flat IVF candidate counts vary with list size distribution.

---

## Repository Structure

```
.
├── 01_transform_domain_similarity.ipynb
├── 02_spectral_tree_routing_poc.ipynb
├── 03_glove_spectral_ann_demo.ipynb
├── 04_sift1m_full_trajectory_ann.ipynb
├── dtdr_index_io.py
├── convert_sift_to_dtdr.py
└── README.md
```

Each notebook builds on the previous, progressing from theoretical demonstration to a complete benchmarked ANN pipeline.

---

## Notebooks

### 01 — Transform-Domain Similarity

Demonstrates that nearest-neighbour identity can be recovered from partial Hadamard-domain measurements and that a genuine spectral trajectory signal exists in DTDR space.

Key observations:

- Progressive emergence of top-1 identity from spectral coefficients
- Strong correlation between similarity margin and stability depth
- L2 certification bound tighter than L1
- Sublinear scaling of stable depth with structured embeddings

### 02 — Spectral Tree Routing (Proof of Concept)

Implements hierarchical mean-of-segment routing inside fixed-size vector bags.

Features:

- Binary tree decomposition (5 levels, bag size = 32)
- Beam descent
- Deterministic candidate budgeting
- L2-consistent node scoring:

```
score = 2·q·mean − ‖mean‖²
```

### 03 — GloVe Spectral ANN Demonstration

Tests routing behaviour on GloVe-300 embeddings (50,000 vectors).

Example results (200 queries):

| top_bags | beam | recall@1 | time (s) |
|---|---|---|---|
| 64 | 2 | 0.825 | 26.3 |
| 128 | 2 | 0.910 | 37.5 |

### 04 — SIFT1M Full End-to-End ANN

Full benchmark on SIFT1M (1M vectors, 10,000 queries) using IVF1024 coarse routing with trajectory routing within lists.

**Setup**

- Dataset: SIFT1M (128-dimensional, L2 metric)
- IVF: 1,024 centroids (MiniBatchKMeans)
- Bag size: 32
- Tree levels: 5
- Beam width: 2
- Evaluation: recall@10

**Example Sweep (2,000 queries)**

| nprobe | top_bags | recall@10 | mean candidates |
|---|---|---|---|
| 4 | 32 | 0.435 | 452 |
| 8 | 32 | 0.536 | 901 |
| 8 | 64 | 0.574 | 901 |
| 16 | 32 | 0.615 | 1,798 |
| 16 | 64 | 0.658 | 1,798 |
| 32 | 64 | 0.700 | 3,578 |

Increasing `top_bags` improves recall without increasing candidate count — confirming genuine discriminative routing rather than brute widening.

---

## Optional: DTDR Persistent Storage Mode (SIFT1M)

In addition to float32 `.npy` arrays, SIFT1M base vectors can be stored persistently in quantised structured orthogonal transform-domain form using `.dtdr`.

This validates that the ANN routing pipeline remains functionally equivalent when built from DTDR storage.

### Conversion

From this directory:

```bash
python convert_sift_to_dtdr.py
```

This converts:

```
sift_base.npy  (~488 MB)
```

to:

```
sift1m_base.dtdr  (~122 MB)
```

### Storage Comparison

| Format | Size |
|---|---|
| float32 .npy | 488 MB |
| float32 zipped | 134 MB |
| .dtdr (int8 Hadamard) | 122 MB |
| .dtdr zipped | 87 MB |

DTDR therefore achieves:

- ~4× reduction vs float32 raw
- ~1.5× reduction vs zipped float32
- ~5.6× total reduction vs raw float32 when zipped

### Running Notebook 04 Using DTDR

Replace:

```python
X = np.load(BASE_NPY, mmap_mode="r")[:BASE_LIMIT].astype(np.float32)
```

with:

```python
from dtdr_index_io import read_dtdr_index

X_full = read_dtdr_index("sift1m_base.dtdr")
X = X_full[:BASE_LIMIT].astype(np.float32)
```

All routing code remains unchanged.

### Observed Behaviour

Using DTDR storage:

| nprobe | top_bags | beam | recall@10 |
|---|---|---|---|
| 8 | 32 | 2 | 0.889 |
| 16 | 32 | 2 | 0.922 |
| 16 | 64 | 2 | 0.931 |

Routing behaviour and candidate counts remain unchanged.

This demonstrates:

- Hierarchical routing survives structured orthogonal transform + quantisation
- The ANN index can be persisted entirely in transform-domain form
- DTDR retains residual lossless compressibility
- No algorithmic modification is required

---

## How It Works

### Pipeline

```
Query
  ↓
IVF centroid shortlist (nprobe lists)
  ↓
Stage A: score bags by L2-proxy to root mean
Stage B: beam descent through tree
  ↓
Leaf candidates
  ↓
Exact L2 rerank
```

Each tree node stores a segment mean. The L2-proxy:

```
score = 2·q·mean − ‖mean‖²
```

is geometrically principled and computationally cheap.

### What This Demonstrates

- DTDR-compatible hierarchical routing produces measurable signal on a standard ANN benchmark
- Equivalent recall to flat IVF with ~8–9× fewer distance computations
- Deterministic candidate budgeting
- Persistent transform-domain storage compatible with ANN infrastructure

---

## Requirements

```
python >= 3.10
numpy
scikit-learn
jupyter
```

**Data:**

- SIFT1M: http://corpus-texmex.irisa.fr/
- GloVe 300d: https://nlp.stanford.edu/projects/glove/

Convert `.fvecs` to `.npy` before running notebook 04.

---

## Background

DTDR (Distributed Transform-Domain Representation) is a computational memory architecture based on the Walsh–Hadamard transform.

By distributing signal energy across orthogonal coefficients and enabling structured quantisation, DTDR supports:

- Persistent storage of high-dimensional semantic data
- Hierarchical routing signals
- Model parameter compression
- Residual lossless compressibility

Related work includes DTDR applied to LLM weight compression and retrieval-augmented generation.

UK Patent Application GB2602157.6 (filed January 2026, Green Channel).

---

## Citation

```bibtex
@misc{west2026dtdr,
  author = {West, Jonathan},
  title  = {DTDR Spectral Trajectory ANN},
  year   = {2026},
  url    = {https://github.com/[your-repo]}
}
```
