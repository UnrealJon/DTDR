# DTDR  Distributed Transform-Domain Representation

DTDR is a method for representing numerical data, including machine-learning model parameters and vector embeddings, in a distributed transform domain that preserves computational functionality while reducing memory footprint and bandwidth pressure. It has been designed as a persistent transform-domain representation, with an associated on-disk file format for storing such data, and in many cases offers advantages over conventional floating-point representations. For example, unlike conventional compression, DTDR maintains a compute-capable representation: inference, similarity search, and approximate nearest-neighbour (ANN) traversal can be performed directly in the transformed domain, without reconstructing full-precision data.
This repository contains **reference implementations and experiments** demonstrating these properties.

---

## TL;DR (Why this repository exists)

- **3–4× storage reduction** for large models and embeddings  
- **End-to-end ANN search entirely in DTDR** (IVF + HNSW + binary reranking)  
- **Novel ANN signal (“dilution evidence”)** enabling recall–latency trade-offs unavailable in standard representations  
- **Graceful degradation** under quantisation *and* on-disk corruption  
- **Substantial residual lossless compression** (ZIP), indicating retained structure  

DTDR is not a codec: it is a **numerical representation**.

---

## Key Results

### Model Storage & Inference

DTDR-compressed model parameters can be reconstructed to a *working numerical precision*
sufficient for standard inference, without specialised kernels.

| Model | FP16 | DTDR-INT8 | Compression | Similarity |
|------|------|-----------|-------------|------------|
| Mistral-7B | ~14.5 GB | ~6.7 GB | ~2.2× | 0.9998 |

*Inference throughput comparable to FP16 baseline.*

See **Experiment 01** for details.

---

### Residual Lossless Compression (ZIP)

DTDR representations retain structured regularity in the transform domain.
As a result, DTDR-stored model parameters exhibit substantial **additional
lossless compression** under standard tools such as ZIP.

Measured on INT8 representations of **Mistral-7B**:

| Representation | Stored Size | ZIP Size | Additional Reduction |
|---------------|-------------|----------|----------------------|
| FP16 | ~14.5 GB | ~14.4 GB | ~0–1% |
| INT8 GGUF | ~8.2 GB | ~7.9 GB | ~3–4% |
| **INT8 DTDR** | **~6.7 GB** | **~4.4–4.7 GB** | **~30–35%** |

This secondary compression is **optional and orthogonal** to DTDR.
All DTDR storage reductions are achieved *prior* to ZIP compression.

---

### Storage Robustness Under On-Disk Corruption

DTDR was evaluated as a **storage representation** under random on-disk corruption.

Identical random byte corruption was applied to:
- FP16 safetensors (baseline)
- DTDR Hadamard-transformed artefacts

Reconstruction fidelity was measured using cosine similarity and relative L2 error.

**Result:**  
FP16 exhibits catastrophic numerical failure at extremely small corruption levels.  
DTDR redistributes damage and degrades **smoothly and statistically**, preserving numerical
validity over orders of magnitude greater corruption.

See:  
`experiments/04_graceful_degradation/dtdr_disk_corruption/`

---

### End-to-End ANN Search (DTDR Domain)

DTDR can act as a **unified numerical domain** for ANN pipelines.

In **Experiment 02**, we demonstrate an end-to-end ANN search operating *entirely* on DTDR vectors,
integrating IVF, HNSW, and binary distance estimation — with no full-precision reconstruction.

| Configuration | Recall@10 | Mean latency |
|--------------|-----------|--------------|
| DTDR-IVF-HNSW-Binary (`nprobe=4`) | 0.63 | 2.9 ms |
| **+ DTDR dilution evidence (`nprobe=4`)** | **0.78** | **3.1 ms** |
| DTDR-IVF-HNSW-Binary (`nprobe=8`) | 0.80 | 4.9 ms |

DTDR’s **multi-resolution dilution evidence** recovers ~15 percentage points of recall
at low probe counts, approaching higher-cost baselines with ~40% lower latency.

This signal does **not exist** in conventional vector representations.

---

## What Is “Dilution Evidence”?

DTDR distributes information across coefficients at multiple aggregation scales.
By examining how similarity signals persist under progressive aggregation (or dilution),
it is possible to infer which regions of the database are likely to contain relevant
neighbours *before* probing them.

This provides a coarse localisation signal that is:
- orthogonal to centroids,
- complementary to graph traversal,
- inexpensive to compute,
- unavailable in localised representations.

---
## A structural property: routing vs search

Most vector search systems scale by searching more of the database.

DTDR-based retrieval behaves differently.

In our end-to-end ANN experiments we repeatedly observe:

> The number of partitions that must be searched remains approximately constant as dataset size grows.

Accuracy is instead controlled by *local refinement* within the routed partitions.

This converts nearest-neighbour retrieval from a global exploration problem into a local ranking problem.

We isolate this behaviour in a minimal reproducible experiment:

➡️ **[Routing vs Local Refinement experiment](end_to_end_ann_search/routing_vs_local_refinement/README.md)**

The experiment shows:

| Dataset size | Partitions searched |
|------------|----------------|
| 50k | ~2 |
| 200k | ~2 |
| 500k | ~2 |

while recall increases only with local candidate depth.

This behaviour is unusual for ANN systems and has potential implications for large-scale search infrastructure, where cross-partition fan-out dominates latency and energy cost.

---

## Why DTDR Is Not Just Compression

Compression optimises **reconstruction fidelity**.  
DTDR optimises **functional equivalence**.

Key differences:

- Information is deliberately **distributed**, not localised  
- Partial corruption leads to **graceful degradation**, not failure  
- Exact reconstruction is optional  
- Computation and search occur **directly in the transform domain**  
- DTDR exposes **new usable structure**, not just smaller files  

DTDR should be viewed as a **numerical representation**, not a storage format.

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

DTDR is the subject of a filed UK patent application:

UK Patent Application No. GB2602157.6

This repository is provided for research and evaluation purposes.

For commercial licensing, strategic partnerships, or IP inquiries:

Contact: dtdr@multiverse1.com

See LICENSE_NOTICE.md for evaluation terms.