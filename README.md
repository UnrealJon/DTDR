# DTDR — Distributed Transform-Domain Representation

DTDR is a persistent numerical representation for machine-learning data — including model parameters and vector embeddings — stored directly in a distributed transform domain.

In DTDR, the stored form is itself compute-capable.  
Inference, similarity search, and approximate nearest-neighbour (ANN) traversal can operate directly on the stored representation without reconstructing full-precision floating-point weights.

Unlike conventional parameter storage, DTDR does not primarily store independently meaningful weights.  
Instead, it stores a globally distributed system of constraints whose solution corresponds to the model.

Because behaviour depends on global consistency rather than precision of individual parameters, DTDR exhibits distinctive operational properties:

- Corruption produces gradual degradation
- Truncation produces functional thresholds
- Compatible priors can restore behaviour
- Computation occurs directly in the transform domain

DTDR functions as a persistent computational representation, not a compression codec.

---

## TL;DR

- 2–4× storage reduction for large models and embeddings
- Compute-capable INT8 representation
- Substantial residual lossless compressibility (ZIP)
- Graceful degradation under corruption
- End-to-end ANN search in transform domain
- Hierarchical trajectory routing: ~8–9× candidate reduction at equivalent recall on SIFT1M (IVF1024 baseline)

---

## 1. Model Storage & Inference

DTDR-compressed model parameters can be reconstructed to numerically working precision sufficient for standard inference.

| Model | FP16 | DTDR-INT8 | Compression | Cosine Similarity |
|-------|------|-----------|-------------|-------------------|
| Mistral-7B | ~14.5 GB | ~6.7 GB | ~2.2× | 0.9998 |

Inference throughput remains comparable to FP16 baselines.

See: `experiments/01_model_inference/`

---

## 2. Residual Lossless Compressibility (Audited Run)

DTDR representations retain structured statistical regularity in the transform domain after quantisation.

Measured on Mistral-7B INT8 artefacts using Python `zipfile` (ZIP64 + DEFLATE, single-file archives):

| Representation | Stored Size (bytes) | ZIP Size (bytes) | Residual Reduction |
|----------------|--------------------|------------------|-------------------|
| GGUF Q8_0 | 7,695,857,952 | 7,411,219,447 | 3.70% |
| **DTDR INT8** | **7,248,464,396** | **5,180,785,451** | **28.52%** |

Residual Reduction is defined as:

(Stored − ZIP) / Stored

Compression was performed using standard ZIP (Deflate) with identical settings across artefacts. No transform-aware compression was applied.

The DTDR representation exhibits substantial residual lossless compressibility beyond quantised storage, indicating preserved structured redundancy in the transform domain.

Secondary ZIP compression is optional and orthogonal to DTDR.

See: `experiments/05_storage_accounting/`

---

## 3. Storage Robustness

DTDR was evaluated under identical random byte corruption compared to FP16 safetensors.

| Representation | Corruption Behaviour |
|----------------|----------------------|
| FP16 | Catastrophic numerical failure at small corruption levels |
| DTDR | Smooth statistical degradation over orders of magnitude greater corruption |

DTDR redistributes damage across coefficients rather than localising it.

See: `experiments/04_graceful_degradation/`

---

## 4. End-to-End ANN in DTDR Domain

DTDR supports ANN pipelines operating entirely within the transform domain, integrating IVF partitioning, HNSW per-list search, binary reranking, and transform-domain scoring — without reconstructing full-precision vectors.

See: `experiments/02_dtdr_end_to_end_search/`

---

## 5. Hierarchical Trajectory Routing

DTDR's transform-domain structure enables a hierarchical routing signal inside IVF lists. Rather than evaluating all vectors in each probed list, a binary tree of segment means guides beam-search descent to geometrically promising leaf candidates.

### How It Works

For each IVF list, vectors are grouped into bags of 32. A binary tree is precomputed across 5 levels, storing the mean of progressively smaller segments at each node. At query time:

1. All bags are scored cheaply by L2-proxy distance to the root mean: `score = 2·q·mean − ‖mean‖²`
2. The top-scoring bags are selected
3. Beam descent through the tree levels localises candidates to the most promising leaves
4. Leaf candidates are deduplicated and reranked by exact L2

Node squared norms are precomputed at index build time, so each node evaluation costs a single dot product at query time.

### Results on SIFT1M (10,000 queries, full 1M index)

| Method | Candidates evaluated | Recall@10 |
|--------|---------------------|-----------|
| Flat IVF1024, nprobe=8 (published baseline) | ~7,812 | ~0.57 |
| **Trajectory router, nprobe=8** | **899** | **0.580** |
| Trajectory router, nprobe=16 | 1,796 | 0.657 |
| Trajectory router, nprobe=32 | 3,577 | 0.698 |

At nprobe=8: **8.7× fewer candidate evaluations** at equivalent recall to the flat IVF baseline.

Candidate counts scale predictably as approximately `nprobe × 112`, making the system deterministic — an operational advantage in latency-sensitive environments where flat IVF candidate counts vary with list size distribution.

A further finding: increasing `top_bags` from 32 to 64 produces identical candidate counts but measurably higher recall at each nprobe level. The beam descent is selecting *different* candidates, not more of them — confirming the tree routing is making genuine discriminative decisions rather than simply widening the search.

See: `experiments/06_trajectory_routing/`

---

## 6. Repository Structure

```text
experiments/
├── 01_model_inference/
├── 02_dtdr_end_to_end_search/
├── 03_embedding_search/
├── 04_graceful_degradation/
│   └── dtdr_disk_corruption/
├── 05_storage_accounting/
├── 06_trajectory_routing/
DTDR_RAG_double_transform_demo.ipynb

Patent & Commercial Licensing

UK patent application under accelerated examination (Green Channel)
UK Patent Application No. GB2602157.6

This repository is provided for research and evaluation purposes.

For commercial licensing, strategic partnerships, or IP inquiries:
Contact: dtdr@multiverse1.com

See LICENSE_NOTICE.md for evaluation terms.