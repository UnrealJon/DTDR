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
- Graceful degradation under corruption
- End-to-end ANN search in transform domain
- Hierarchical trajectory routing: ~8–9× candidate reduction at equivalent recall on SIFT1M (IVF1024 baseline)

> **Note on residual ZIP compressibility:** an earlier version of this section reported a residual compressibility advantage for DTDR over conventional INT8. Further internal testing found that the specific artefact used to produce that figure had not actually undergone the DTDR transform. See the correction in Section 2 below.

---

## 1. Model Storage & Inference

DTDR-compressed model parameters can be reconstructed to numerically working precision sufficient for standard inference.

| Model | FP16 | DTDR-INT8 | Compression | Cosine Similarity |
|-------|------|-----------|-------------|-------------------|
| Mistral-7B | ~14.5 GB | ~6.7 GB | ~2.2× | 0.9998 |

Inference throughput remains comparable to FP16 baselines.

See: `experiments/01_model_inference/`

---

## 2. Residual Lossless Compressibility — CORRECTED

> **Correction (added following further internal testing):** the table and figures originally published in this section were measured on a build artefact (`dtdr_int8_clean.pkl`) that was subsequently found **not to have undergone the DTDR structured-transform step**. It was plain blockwise-quantised data in the original parameter domain. This was confirmed directly: the artefact's decompressed output matches the untransformed source weights closely (cosine ≈ 0.997) and does not match a genuinely transformed version of the same weights (cosine ≈ 0.0002, i.e. no meaningful relationship). The 28.52% figure below is real but does not demonstrate what it was described as demonstrating, and should not be relied on as evidence of a transform-attributable compressibility benefit.

**Original (uncorrected) figures, retained here for transparency:**

| Representation | Stored Size (bytes) | ZIP Size (bytes) | Residual Reduction |
|----------------|--------------------|------------------|-------------------|
| GGUF Q8_0 | 7,695,857,952 | 7,411,219,447 | 3.70% |
| DTDR INT8 (artefact not actually transformed — see correction above) | 7,248,464,396 | 5,180,785,451 | 28.52% |

**Corrected findings, from re-testing with the transform genuinely applied to real weight data, using the same blockwise INT8 quantisation scheme, measured byte-weighted across all 2D parameter tensors:**

| Configuration | Residual Reduction |
|---|---|
| Blockwise INT8, no transform applied | ~28.5% |
| Blockwise INT8, DTDR transform genuinely applied | ~17.3% |

Applying the transform **reduces** residual ZIP compressibility relative to the same quantisation scheme without it. This is mechanistically expected rather than anomalous: the transform's function is to distribute/decorrelate information across coefficients, which is the opposite of the redundancy a generic compressor like DEFLATE depends on to find anything to compress. The original 28.52% figure is better explained by the coarse block granularity used in that quantisation pass (256 rows sharing one scale factor) than by anything specific to DTDR's transform — a plain blockwise-quantised, untransformed representation using the same coarse blocks reproduces a comparable figure on its own.

**Conclusion:** DTDR does not currently demonstrate a residual-lossless-compressibility advantage attributable to the transform step. This claim is withdrawn pending any further evidence. Compression behaviour is primarily a function of quantisation block granularity, independent of whether a structured transform is applied.

See: `experiments/05_storage_accounting/`

---

## 3. Storage Robustness

DTDR was evaluated under identical random byte corruption compared to FP16 safetensors.

| Representation | Corruption Behaviour |
|----------------|----------------------|
| FP16 | Catastrophic numerical failure at small corruption levels |
| DTDR | Smooth statistical degradation over orders of magnitude greater corruption |

DTDR redistributes damage across coefficients rather than localising it.

This specific comparison (transform-plus-quantisation vs. uncompressed FP16) has not been contradicted by later testing. Later, more extensive testing did find that the degradation-robustness *benefit specifically attributable to the transform step* (as opposed to quantisation alone, and as opposed to FP16) is real and reproducible on real model weights. That same later testing found other quantisation schemes in the wider field can exceed DTDR's robustness at matched corruption levels; a full comparative picture is still being assembled and is not reflected in the single comparison above.

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
```

## Patent & Commercial Licensing

UK patent application under accelerated examination (Green Channel)
UK Patent Application No. GB2602157.6

This repository is provided for research and evaluation purposes.

For commercial licensing, strategic partnerships, or IP inquiries:
Contact: dtdr@multiverse1.com

See LICENSE_NOTICE.md for evaluation terms.
