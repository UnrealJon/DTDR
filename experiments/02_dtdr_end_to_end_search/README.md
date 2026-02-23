# DTDR Spectral Trajectory ANN

**Distributed Transform-Domain Representation (DTDR)** enables a hierarchical routing signal inside approximate nearest-neighbour (ANN) indices. This repository demonstrates that signal on structured embeddings and the SIFT1M benchmark.

The goal is not to replace IVF or HNSW, but to show that DTDR's internal transform-domain structure enables *principled trajectory routing within IVF lists*, reducing candidate evaluations while maintaining recall.

---

## Key Result

On SIFT1M (1,000,000 vectors, 10,000 queries):

| Method | Candidates evaluated | Recall@10 |
|---|---|---|
| Flat IVF1024, nprobe=8 (published baseline) | ~7,812 | ~0.57 |
| **Trajectory router, nprobe=8** | **899** | **0.580** |
| Trajectory router, nprobe=16 | 1,796 | 0.657 |
| Trajectory router, nprobe=32 | 3,577 | 0.698 |

At nprobe=8: **8.7× fewer candidate evaluations** at equivalent recall to the flat IVF baseline.

Candidate counts scale predictably with routing parameters — approximately `nprobe × 112` — making the system deterministic and suitable for latency-sensitive production environments.

---

## Repository Structure

```
.
├── 01_transform_domain_similarity.ipynb   # Spectral similarity emergence
├── 02_spectral_tree_routing_poc.ipynb     # Hierarchical routing proof of concept
├── 03_glove_spectral_ann_demo.ipynb       # Structured embeddings (GloVe300)
├── 04_sift1m_full_trajectory_ann.ipynb    # Full SIFT1M end-to-end evaluation
└── README.md
```

Each notebook builds on the previous, progressing from theoretical demonstration to a complete benchmarked ANN pipeline.

---

## Notebooks

### 01 — Transform-Domain Similarity

Demonstrates that nearest-neighbour identity can be recovered from partial Hadamard-domain measurements, and that a genuine spectral trajectory signal exists in DTDR space.

Key observations:
- Progressive emergence of top-1 identity from spectral coefficients
- Strong correlation between similarity margin and stability depth
- L2 certification bound significantly tighter than L1
- Sublinear scaling of stable depth with structured embeddings

### 02 — Spectral Tree Routing (Proof of Concept)

Implements hierarchical mean-of-segment routing inside fixed-size vector bags. Bridges the spectral similarity theory from notebook 01 to the routing mechanics used in the full pipeline.

Features:
- Binary tree decomposition of bag vectors across 5 levels (bag size = 32)
- Beam descent with controllable width
- Deterministic candidate budgeting
- L2-consistent node scoring: `score = 2·q·mean − ‖mean‖²`

### 03 — GloVe Spectral ANN Demonstration

Tests routing behaviour on GloVe-300 embeddings (50,000 vectors). Shows that the trajectory signal is stronger on real structured embeddings than on random vectors.

Key results (200 queries):

| top_bags | beam | recall@1 | time (s) |
|---|---|---|---|
| 64 | 2 | 0.825 | 26.3 |
| 128 | 2 | 0.910 | 37.5 |

### 04 — SIFT1M Full End-to-End ANN

Full benchmark on SIFT1M (1M vectors, 10,000 queries) using IVF1024 coarse routing with trajectory routing within lists.

**Setup:**
- Dataset: SIFT1M (128-dimensional, L2 metric)
- IVF: 1,024 centroids, MiniBatchKMeans
- Bag size: 32 vectors
- Tree levels: 5 (nodes: 1, 2, 4, 8, 16 per level)
- Beam width: 2
- Evaluation: standard recall@10, unfiltered ground truth

**Full sweep (2,000 queries):**

| nprobe | top_bags | recall@10 | mean candidates |
|---|---|---|---|
| 4 | 32 | 0.435 | 452 |
| 8 | 32 | 0.536 | 901 |
| 8 | 64 | 0.574 | 901 |
| 16 | 32 | 0.615 | 1,798 |
| 16 | 64 | 0.658 | 1,798 |
| 32 | 64 | 0.700 | 3,578 |

Note: top_bags=32 and top_bags=64 produce **identical candidate counts** at each nprobe level. The recall difference arises from the beam descent selecting different candidates — not from evaluating more of them. This confirms the tree routing is making genuine discriminative decisions.

---

## How It Works

### Pipeline

```
Query
  │
  ▼
IVF centroid shortlist (nprobe lists)
  │
  ▼  for each probed list:
  │
  ├─ Stage A: score all bags by L2-proxy to root mean
  │           score = 2·q·mean − ‖mean‖²
  │           keep top_bags bags
  │
  └─ Stage B: beam descent through tree levels 1→4
              at each level, expand beam-width best children
              collect leaf candidates from final level
  │
  ▼
Deduplicate and rerank by exact L2
  │
  ▼
Top-k results
```

### Why Mean-of-Segment Scoring Works

Each tree node stores the mean of a contiguous segment of vectors within a bag. The L2-proxy score `2·q·mean − ‖mean‖²` is a computationally cheap but geometrically principled filter: a segment whose mean is close to the query is likely to contain individual vectors close to the query. At each level of the tree, the segment size halves, progressively localising the search to the most promising leaves.

This is the *trajectory* signal: query-to-segment-mean similarity that increases monotonically as segment size decreases, converging on a true near-neighbour. The rate of convergence is a function of DTDR's energy distribution across transform-domain coefficients.

### Node scoring precomputation

At index build time, `‖mean‖²` is computed and stored for every tree node. At query time, scoring any node requires only a single dot product `q·mean`. The full L2 proxy is then recovered without computing `‖q‖²` (constant across all nodes for a given query).

---

## What This Demonstrates

- DTDR-compatible hierarchical routing produces a real, measurable signal on a standard ANN benchmark
- The signal localises promising regions *within* IVF lists without exhaustive evaluation
- Candidate evaluations scale predictably with routing parameters
- Equivalent recall to flat IVF is achievable with ~8-9× fewer distance computations

## What This Does Not Claim

- It does not replace HNSW or state-of-the-art ANN systems
- It does not claim top performance on ANN benchmarks
- It does not rely on heuristic pruning — the routing criterion is geometrically principled

---

## Requirements

```
python >= 3.10
numpy
scikit-learn
jupyter
```

**Data:**
- SIFT1M: http://corpus-texmex.irisa.fr/ (notebooks 01, 02, 04)
- GloVe 300d: https://nlp.stanford.edu/projects/glove/ (notebook 03)

Pre-convert SIFT `.fvecs` files to `.npy` format before running notebook 04.

---

## Background

DTDR (Distributed Transform-Domain Representation) is a computational memory architecture based on the Walsh-Hadamard transform. By distributing signal energy uniformly across transform-domain coefficients, DTDR enables structured approximation with certifiable error bounds.

These experiments form part of a broader research programme exploring DTDR as a general-purpose memory architecture for machine learning systems. Related work includes DTDR applied to LLM weight compression (2× GPU speedup, 5× DRAM traffic reduction on Mistral-7B) and to retrieval-augmented generation with graceful degradation under coefficient loss.

UK Patent Application GB2602157.6 (filed January 2026, Green Channel).

---

## Citation

If you find this work useful:

```bibtex
@misc{west2026dtdr,
  author = {West, Jonathan},
  title  = {DTDR Spectral Trajectory ANN},
  year   = {2026},
  url    = {https://github.com/[your-repo]}
}
```
