# DTDR — Distributed Transform-Domain Representation

DTDR is a method for representing numerical data — including machine-learning model parameters
and vector embeddings — in a **distributed transform domain** that preserves computational
functionality while reducing memory footprint and bandwidth pressure.

Unlike conventional compression, DTDR maintains a **compute-capable representation**:
inference, similarity search, and approximate nearest-neighbour (ANN) traversal can be performed
*directly in the transformed domain*, without reconstructing full-precision data.

This repository contains **reference implementations and experiments** demonstrating these properties.

---

## TL;DR (What’s new here)

- **3–4× storage reduction** for large models and embeddings  
- **End-to-end ANN search entirely in DTDR** (IVF + HNSW + binary reranking)  
- **Novel ANN signal (“dilution evidence”)** that recovers large fractions of recall at low probe counts  
- Improvements are **orthogonal** to existing ANN optimisations — DTDR adds *new capability*, not just compression  

---

## Key Results

### Model Storage & Inference

DTDR-compressed model parameters can be reconstructed to a *working numerical precision*
sufficient for standard inference, without specialised kernels.

| Model | FP16 | DTDR-INT8 | Compression | Similarity |
|------|------|-----------|-------------|------------|
| Mistral-7B | 14.5 GB | 6.7 GB | 2.2× | 0.9998 |

*Inference throughput comparable to FP16 baseline.*

See **Experiment 01** for details.

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

**Interpretation:**

DTDR’s multi-resolution dilution evidence recovers ~15 percentage points of absolute recall
at low probe counts, approaching the recall of a higher-cost baseline with ~37–40% lower latency.

This behaviour is smooth, interpretable, and consistent with ANN theory — but the signal itself
does **not exist** in standard vector representations.

See **Experiment 02** for full methodology and results.

---

## What Is “Dilution Evidence”?

DTDR distributes information across coefficients at multiple aggregation scales.
By examining how similarity signals persist under progressive aggregation (or dilution),
it is possible to infer *which regions of the database are likely to contain relevant neighbours*
before probing them.

This provides a **coarse localisation signal** that is:

- independent of centroids,
- orthogonal to graph traversal,
- inexpensive to compute,
- and unavailable in representations where information is localised.

In practice, this allows ANN systems to operate effectively at lower probe counts,
where conventional IVF methods discard the most information.

---

## Why DTDR Is Not Just Compression

Compression schemes optimise **reconstruction fidelity**.

DTDR optimises **functional equivalence**.

Key differences:

- Information is deliberately **distributed**, not localised
- Partial corruption leads to **graceful degradation**, not artefacts
- Exact reconstruction is optional — sometimes undesirable
- Computation, similarity, and traversal can occur **directly in the transformed domain**
- DTDR exposes **new usable structure** (e.g. multi-resolution persistence)

DTDR should therefore be viewed as a **numerical representation**, not a storage format.

---

## Repository Structure

```text
experiments/
├── 01_model_inference/
│   └── DTDR-compressed model inference benchmarks
└── 02_dtdr_end_to_end_search/
    └── End-to-end ANN search entirely in the DTDR domain


Patent & Commercial Licensing

DTDR is the subject of a filed UK patent application:

UK Patent Application No. GB2602157.6

This repository is provided for research and evaluation purposes.

For commercial licensing, strategic partnerships, or IP inquiries:

Contact: dtdr@multiverse1.com

See LICENSE-NOTICE.md  https://github.com/UnrealJon/DTDR/blob/main/LICENSE_NOTICE.md
 for evaluation terms.

