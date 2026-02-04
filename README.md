# DTDR — Distributed Transform-Domain Representation

DTDR is a method for representing numerical data — including machine-learning model parameters
and vector embeddings — in a **distributed transform domain** that preserves computational
functionality while substantially reducing memory footprint and bandwidth pressure.

Unlike conventional compression, DTDR produces a **search- and compute-capable representation**
that can be used directly for inference, similarity search, and approximate nearest-neighbour (ANN)
pipelines without reconstructing full-precision data.

This repository contains **reference implementations and experiments** demonstrating these
properties.

---

## Why DTDR?

Most systems today follow the pattern:

> full-precision representation → compression → reconstruction → computation

DTDR enables an alternative:

> **compressed-but-structured representation → native computation**

Key properties:

- Information is **distributed across coefficients**, not localised
- Partial corruption degrades functionality **gracefully**
- The representation remains compatible with existing kernels and distance metrics
- Computation can occur **directly in the transformed domain**

These properties make DTDR especially relevant for **memory-bound workloads**.

---

## Key Results (Summary)

### 1. Model Inference from DTDR Storage

DTDR-compressed model parameters can be reconstructed to a *working numerical precision*
sufficient for standard inference, without specialised kernels.

In a reference implementation using a 7-billion-parameter language model:

- FP16 baseline: ~14.5 GB
- DTDR-INT8 stored representation: ~6.7–6.8 GB
- Inference throughput: comparable to or faster than FP16
- Robustness under controlled corruption

See **Experiment 01** for details.

---

### 2. End-to-End ANN Search Entirely in the DTDR Domain

DTDR can act as a **unified numerical domain** for approximate nearest-neighbour search.

In **Experiment 02**, we demonstrate an end-to-end ANN pipeline operating *entirely* on DTDR
vectors, integrating:

- IVF (inverted file indexing)
- HNSW (graph-based ANN traversal)
- Binary distance estimation (RaBitQ-like)
- A DTDR-specific multi-resolution “dilution evidence” signal for coarse localisation

No full-precision reconstruction is required at any stage.

#### Representative results (50k vectors, CPU, Python):

| Configuration | Recall@10 | Mean query time |
|--------------|-----------|-----------------|
| DTDR-IVF-HNSW-Binary (`nprobe=4`) | 0.63 | 2.9 ms |
| **+ DTDR dilution evidence (`nprobe=4`)** | **0.78** | **3.1 ms** |
| DTDR-IVF-HNSW-Binary (`nprobe=8`) | 0.80 | 4.9 ms |

DTDR dilution evidence recovers ~15 percentage points of absolute recall in the low-`nprobe`
regime, approaching the recall of a higher-cost baseline with substantially lower latency.

This behaviour is smooth, interpretable, and consistent with established ANN theory.

---

## Why This Is Not Just Compression

DTDR differs fundamentally from conventional codecs:

- It is **not optimised for reconstruction fidelity alone**
- It preserves **computational semantics**
- It supports **direct similarity, traversal, and inference**
- It exposes additional signals (e.g. multi-resolution persistence) that do not exist in
  standard representations

DTDR should be viewed as a **numerical representation**, not a storage format.

---

## Repository Structure

```text
experiments/
├── 01_model_inference/
│   └── DTDR-compressed model inference benchmarks
└── 02_dtdr_end_to_end_search/
    └── End-to-end ANN search entirely in the DTDR domain
