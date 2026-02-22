# DTDR Experiments

A structured sequence of runnable experiments demonstrating the behavioural, computational, and storage properties of **Distributed Transform-Domain Representation (DTDR)**.

DTDR is treated throughout as a **persistent, compute-capable numerical representation**: models, embeddings, and vectors are stored in transform-domain form and operated on directly, without reconstructing full-precision floating-point data.

The experiments progress from minimal mathematical demonstrations to full-scale model and ANN pipeline tests.

---

## System Requirements

These experiments have been developed and tested on a consumer workstation:

| Component | Specification |
|-----------|--------------|
| CPU | Intel i9-12900F |
| RAM | 64GB DDR5 |
| GPU | NVIDIA RTX 3090 (24GB VRAM) |
| OS | Windows 11 |

> **Notes:**
> - Experiment 00 runs on **CPU only**
> - Experiments 01–05 vary in requirements (see individual sections below)
> - GPU is required for large-model experiments involving Mistral 7B
> - Significant disk space is required for model downloads
> - All experiments are written in **Python**

---

## Experiment Overview

> **Recommended execution order:** 00 → 01 → 03 → 04 → 02 → 05

---

### 00 — Start Here: DTDR Geometry & Degradation Demo

**Location:** `00_startDTDR_demo/`  
**Files:** `.ipynb` and `.py` versions available

The minimal, self-contained introduction to DTDR. This is the **recommended first step for all readers**.

**Demonstrates:**
- Similarity preservation under orthogonal transforms
- Direct computation in transform-domain representations
- Composite transform constructions
- Graceful degradation under coefficient loss
- Comparison with a no-transform INT8 baseline

**Requirements:** No external models · No GPU · Public-domain text downloaded automatically

---

### 01 — Functional Reconstruction of a DTDR-Stored Model

**Location:** `01_model_inference/`

Demonstrates that a model stored persistently in DTDR form can be reconstructed to a numerically different but functionally equivalent representation for inference.

**Focus:**
- Persistent on-disk DTDR storage
- Functional equivalence vs numerical identity
- Robustness under perturbation

**Requirements:** HuggingFace model download required (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) · GPU strongly recommended · Significant disk space required

---

### 02 — End-to-End ANN Search in the DTDR Domain

**Location:** `02_dtdr_end_to_end_search/`

Implements a complete approximate nearest-neighbour (ANN) pipeline operating entirely in the DTDR domain.

**Includes:**
- IVF (inverted file indexing)
- HNSW graph traversal
- Binary distance approximation (RaBitQ-like)
- DTDR-specific multi-resolution dilution signal

**Focus:**
- DTDR as a primary computational domain
- Recall / latency trade-offs
- Additional routing signals enabled by distributed representations

**Requirements:** GloVe embeddings (downloaded automatically or manually) · Moderate RAM · CPU sufficient, GPU optional

---

### 03 — Embedding Similarity Search in DTDR

**Location:** `03_embedding_search/`

Demonstrates similarity search over embedding vectors stored and queried directly in transform-domain form.

**Focus:**
- Similarity preservation under orthogonal transforms
- Retrieval equivalence in DTDR domain

**Requirements:** CPU sufficient

---

### 04 — Graceful Degradation Under Perturbation

**Location:** `04_graceful_degradation/`

Systematic evaluation of DTDR behaviour under adverse conditions.

**Covers:**
- Coefficient dropout
- Block loss
- File corruption simulations
- Comparison against FP16 baselines

**Focus:**
- Smooth degradation vs catastrophic failure
- Structural robustness of distributed representations

**Requirements:** CPU sufficient · GPU optional for larger models

---

### 05 — Storage Accounting & Residual Compressibility

**Location:** `05_storage_accounting/`

Quantitative storage analysis comparing representation formats.

**Compares:**
- FP16
- INT8
- DTDR-INT8
- Secondary lossless compression (ZIP)

**Focus:**
- Persistent storage footprint
- Evidence that DTDR is not entropy-saturated compression
- Residual structural compressibility

**Requirements:** Large model downloads required for full demonstration

---

## External Downloads

Some experiments require external resources that are **not included in the repository**:

- HuggingFace model downloads (e.g. `mistralai/Mistral-7B-Instruct-v0.3`)
- GloVe embedding datasets
- Public-domain texts (downloaded automatically)

Ensure you have adequate disk space, sufficient RAM, and a stable internet connection before running these experiments.

---

## Purpose of This Directory

This folder is **not intended as a production framework**. It is a structured collection of controlled experiments designed to:

- Test DTDR hypotheses
- Demonstrate compute-capable transform-domain representations
- Explore robustness and degradation behaviour
- Evaluate storage properties
- Investigate ANN routing behaviour

Each experiment includes its own `README` explaining the hypothesis, methodology, measured outcomes, and interpretation.
