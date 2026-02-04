# Experiment 02 — End-to-End ANN Search Entirely in the DTDR Domain

## Overview

This experiment demonstrates an **end-to-end approximate nearest-neighbour (ANN) search pipeline**
operating *entirely* in a **Distributed Transform-Domain Representation (DTDR)**, without reconstructing
full-precision vectors at any intermediate stage.

The pipeline integrates four standard ANN components:

- **IVF (Inverted File Index)** for coarse partitioning  
- **HNSW (Hierarchical Navigable Small World graphs)** for approximate traversal  
- **Binary distance estimation (RaBitQ-like)** for fast re-ranking  
- **DTDR multi-resolution “dilution evidence”** for enhanced coarse localisation  

All stages operate directly on DTDR-transformed vectors.

The experiment evaluates recall–latency trade-offs and, in particular, the impact of DTDR-specific
dilution evidence in the low-`nprobe` regime where classical IVF probing is weakest.

---

## Motivation

Conventional ANN systems typically follow the pattern:

> compressed storage → partial or full reconstruction → search

DTDR enables a different approach:

> compressed-but-structured storage → native search in the transformed domain

This experiment tests whether DTDR can act as a **unified numerical domain** supporting:

- coarse partitioning (IVF),
- graph-based ANN traversal (HNSW),
- binary distance estimation,
- and additional localisation signals unavailable in standard vector representations.

---

## Pipeline Summary

All steps below operate on DTDR vectors only.

1. **DTDR Transform**  
   Input vectors are transformed using an orthogonal Walsh–Hadamard transform and L2-normalised.

2. **DTDR-Native IVF Construction**  
   - k-means clustering is performed directly in DTDR space.
   - Each vector is assigned to one of `nlist` IVF lists.

3. **Optional DTDR Dilution Evidence (Coarse Localisation)**  
   - For each IVF list, vectors are examined across multiple aggregation scales.
   - A list-level “evidence score” is computed based on the persistence of similarity across scales.
   - Lists are ranked using this evidence prior to probing.
   - This step is optional and introduces minimal overhead.

4. **DTDR-Native HNSW Traversal**  
   - HNSW indices are built per IVF list.
   - Graph traversal and distance comparisons use DTDR cosine similarity.

5. **Binary Re-Ranking (RaBitQ-like)**  
   - DTDR vectors are encoded using random hyperplane 1-bit projections.
   - Hamming distance provides a fast distance estimate.
   - A small shortlist is optionally re-ranked using exact DTDR cosine similarity.

6. **Evaluation**  
   - Ground truth is computed via brute-force cosine similarity in DTDR space.
   - Recall@10 and mean query latency are reported.

At no point is full-precision reconstruction required.

---

## Experimental Setup

- Dataset size: 50,000 vectors  
- Dimensionality: 256  
- Queries: 200  
- IVF lists (`nlist`): 256  
- HNSW parameters: `M = 16`, `ef = 128`  
- Binary code length: 256 bits  
- Hardware: CPU (Python implementation)

---

## Results

### Baseline DTDR-IVF-HNSW-Binary Pipeline

| nprobe | Recall@10 | Mean query time |
|-------:|:----------|:----------------|
| 4 | 0.6285 | 2.87 ms |
| 8 | 0.7950 | 4.91 ms |
| 16 | 0.8805 | 6.53 ms |
| 32 | 0.9020 | 7.20 ms |

These results show smooth, expected recall–latency scaling, indicating that IVF and HNSW behave
normally when operating directly in the DTDR domain.

---

### Effect of DTDR Dilution Evidence (Low-nprobe Regime)

| nprobe | Dilution evidence | Recall@10 | Mean query time |
|-------:|:------------------|:----------|:----------------|
| 2 | Yes | 0.4950 | 2.36 ms |
| 4 | No | 0.6285 | 2.87 ms |
| 4 | **Yes** | **0.7760** | **3.10 ms** |
| 8 | No | 0.7950 | 4.91 ms |

**Key observation:**

At `nprobe = 4`, DTDR dilution evidence recovers approximately **15 percentage points of absolute recall**
with only ~0.2 ms additional latency, approaching the recall of an `nprobe = 8` baseline at
substantially lower cost.

---

## Interpretation

- DTDR supports **native ANN search** using established techniques (IVF, HNSW, binary distance estimation).
- DTDR’s distributive structure enables an **additional coarse localisation signal** that is:
  - independent of centroids,
  - orthogonal to graph traversal,
  - inexpensive to compute.
- This signal is most effective in the **low-`nprobe` regime**, where conventional IVF probing discards
  the most information.
- Performance degrades **smoothly and predictably** under extreme pruning (`nprobe = 2`), indicating
  that the method respects fundamental information limits rather than introducing artefacts.

---

## Why This Is Not Just Compression

Unlike conventional compression schemes:

- DTDR distributes information across coefficients rather than localising it.
- The transformed representation remains **searchable and composable**.
- Additional signals (such as dilution evidence) can be extracted *because* the information is distributed.

This allows DTDR to function not merely as a storage format, but as an **active numerical domain for
inference and search**.

---

## How to Run

Baseline:

```powershell
python experiment02_dtdr_ivf_hnsw_rabitq.py --n 50000 --q 200 --d 256 --nlist 256 --nprobe 8
