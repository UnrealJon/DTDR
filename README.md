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
- Transform-domain micro-aggregation signals for routing

---

## 1. Model Storage & Inference

DTDR-compressed model parameters can be reconstructed to numerically working precision sufficient for standard inference.

| Model | FP16 | DTDR-INT8 | Compression | Cosine Similarity |
|------|------|-----------|-------------|------------------|
| Mistral-7B | ~14.5 GB | ~6.7 GB | ~2.2× | 0.9998 |

Inference throughput remains comparable to FP16 baselines.

See: `experiments/01_model_inference/`

---

## 2. Residual Lossless Compression

DTDR representations retain structured regularity in the transform domain.

Measured on INT8 Mistral-7B:

| Representation | Stored Size | ZIP Size | Additional Reduction |
|---------------|-------------|----------|----------------------|
| FP16 | ~14.5 GB | ~14.4 GB | ~0–1% |
| INT8 GGUF | ~8.2 GB | ~7.9 GB | ~3–4% |
| **INT8 DTDR** | **~6.7 GB** | **~4.4–4.7 GB** | **~30–35%** |

Secondary ZIP compression is optional and orthogonal to DTDR.

---

## 3. Storage Robustness

DTDR was evaluated under identical random byte corruption compared to FP16 safetensors.

Observed behaviour:

| Representation | Corruption Behaviour |
|---------------|----------------------|
| FP16 | Catastrophic numerical failure at small corruption levels |
| DTDR | Smooth statistical degradation over orders of magnitude greater corruption |

DTDR redistributes damage across coefficients rather than localising it.

See: `experiments/04_graceful_degradation/`

---

## 4. End-to-End ANN in DTDR Domain

DTDR supports ANN pipelines entirely within the transform domain.

Experiment 02 integrates:

- IVF partitioning
- HNSW per-list search
- Binary reranking
- Transform-domain scoring

All without reconstructing full-precision vectors.

See: `experiments/02_dtdr_end_to_end_search/`

---

## 5. Micro-Dilution Routing Signal

Earlier versions of this repository reported large recall gains from a “dilution evidence” heuristic.  
Subsequent corrections revealed those gains were overstated due to timing scope and evaluation inconsistencies.

The revised approach evaluates a level-1 transform-domain aggregation:

For each IVF list, pairwise sums of vectors are precomputed:

u_k = normalize(x_{2k} + x_{2k+1})

At query time, lists are scored using:

s(list) = max_k (q · u_k)  or  top-k mean

On SIFT1M (200k subset, cosine-consistent GT, nprobe=1):

| Routing Method | List Hit-Rate |
|----------------|---------------|
| Centroid only | 0.4865 |
| + Micro-dilution (level-1) | 0.4915 |

This represents a small but measurable improvement (~+1%) at ~+2ms/query CPU overhead in the current prototype.

Key observations:

- Deep hierarchical dilution does not improve routing.
- Signal is concentrated in shallow (level-1) aggregation.
- Micro-aggregation behaves as a secondary routing feature.
- The effect is incremental, not transformative.

The result is reproducible and provided transparently.

---

## 6. What Dilution Is — and Is Not

Dilution is not a navigable tree search structure.

Deep aggregation collapses discriminative signal.

However, shallow transform-domain mixing produces a measurable list-level containment signal.

This signal can:

- Slightly improve IVF routing
- Act as a secondary ranking feature
- Operate entirely in transform domain

It does not replace centroids or graph traversal.

---

## 7. Repository Structure


---

## Repository Structure

```text
experiments/
├── 01_model_inference/          # DTDR model storage and inference reconstruction
├── 02_dtdr_end_to_end_search/   # IVF + HNSW + RaBitQ-style ANN in DTDR domain
├── 03_embedding_search/         # DTDR embeddings and similarity search
├── 04_graceful_degradation/     # Quantisation and corruption robustness
│   └── dtdr_disk_corruption/    # On-disk corruption study (FP16 vs DTDR)
├── 05_storage_accounting/       # Storage sizing and residual compressibility
DTDR_RAG_double_transform_demo.ipynb

Patent & Commercial Licensing

UK patent application under accelerated examination (Green Channel)

UK Patent Application No. GB2602157.6

This repository is provided for research and evaluation purposes.

For commercial licensing, strategic partnerships, or IP inquiries:

Contact: dtdr@multiverse1.com

See LICENSE_NOTICE.md for evaluation terms.