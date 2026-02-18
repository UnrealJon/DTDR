# DTDR — Distributed Transform-Domain Representation

DTDR is a method for representing numerical data — including machine-learning
model parameters and vector embeddings — in a distributed transform domain that
preserves *computational structure* rather than merely preserving values.

Unlike compression formats, DTDR is a persistent numerical representation:
computation, search, and inference can operate directly on the stored form.

This repository contains reference implementations and experiments demonstrating
the properties of this representation.

---

## The Core Idea

Most numerical formats store information locally:

> individual values contain individual meaning

DTDR stores information collectively:

> meaning exists only in the consistency of the whole representation

As a result, two independent properties appear:

| Property | What survives partial information |
|--------|------|
Relational structure | survives |
Functional behaviour | does not |

In simple terms:

> You may lose the ability to reconstruct the answer before you lose the ability to find where the answer is.

The experiments in this repository demonstrate this principle repeatedly.

---

## TL;DR

- 3–4× storage reduction for large models and embeddings
- ANN search performed entirely in DTDR domain
- Novel routing signal (“dilution evidence”)
- Robust to corruption
- Additional lossless compressibility
- Behaviour emerges only when representation becomes globally consistent

DTDR is **not a codec** — it is a computational representation.

---

## Key Experimental Results

---

### 1. Model Storage & Inference Reconstruction

DTDR-compressed parameters reconstruct to a numerically working model using
standard kernels.

| Model | FP16 | DTDR-INT8 | Compression | Similarity |
|------|------|-----------|-------------|------------|
| Mistral-7B | ~14.5 GB | ~6.7 GB | ~2.2× | 0.9998 |

Inference behaviour matches the FP baseline once reconstruction is complete.

See `experiments/01_model_inference/`

---

### 2. Emergence of Behaviour (Critical Completeness)

Partial reconstruction does **not** produce a weaker model.

Instead:

| Fraction present | Behaviour |
|------|------|
Low | no language |
Medium | unstable fragments |
High | sudden coherent inference |

Inference appears only after sufficient global constraints exist.

This shows DTDR stores behaviour as a *constraint closure* rather than a layered approximation.

See `experiments/06_emergence_from_distribution/`

---

### 3. Graceful Degradation Under Corruption

Random on-disk corruption was applied to:

- FP16 safetensors
- DTDR transformed parameters

Result:

FP16 → catastrophic numerical failure  
DTDR → smooth statistical degradation

DTDR redistributes error rather than concentrating it.

See `experiments/04_graceful_degradation/`

---

### 4. End-to-End ANN Search in DTDR Domain

ANN search can operate entirely within DTDR vectors without reconstruction.

| Configuration | Recall@10 | Latency |
|--------------|-----------|------|
DTDR-IVF-HNSW | 0.63 | 2.9 ms |
+ dilution evidence | 0.78 | 3.1 ms |

DTDR introduces a routing signal unavailable in conventional embeddings.

See `experiments/02_dtdr_end_to_end_search/`

---

### 5. Routing Reliability on Real Embeddings

On real GloVe embeddings:

| Local candidates | True NN retained |
|------|------|
80 | 85% |
160 | 96% |
320 | 100% |

The correct neighbour survives routing even when most of the database is ignored.

Meaning:

> Location survives before identity.

---

### 6. Residual Lossless Compression

DTDR artefacts compress further under ZIP:

| Representation | Additional reduction |
|------|------|
FP16 | ~0–1% |
INT8 | ~3–4% |
DTDR-INT8 | ~30–35% |

Indicates preserved transform-domain structure.

See `experiments/05_storage_accounting/`

---

## What “Dilution Evidence” Means

DTDR distributes signal across multiple aggregation scales.

By observing how similarity persists under progressive aggregation,
the system predicts *where relevant data will be* before accessing it.

This converts search from global exploration to local verification.

---

## The Unifying Interpretation

All experiments demonstrate the same structural property:

| Experiment | What remains stable |
|------|------|
Corruption | approximate values |
Routing | geometric relations |
Truncation | nothing (until closure) |

DTDR behaves like a constraint system:

- insufficient constraints → no function
- sufficient constraints → stable function

But relational structure exists earlier.

This separates two concepts normally tied together:

> identity vs similarity

---

## Why This Matters

Traditional representations optimise reconstruction fidelity.

DTDR optimises computational usefulness.

This enables:

- routing before reconstruction
- search without full decoding
- robust storage
- progressive computation
- large-scale retrieval efficiency

DTDR should therefore be viewed as a numerical coordinate system for computation.

---

## Repository Structure



```text
experiments/
├── 01_model_inference/
├── 02_dtdr_end_to_end_search/
├── 03_embedding_search/
├── 04_graceful_degradation/
├── 05_storage_accounting/
├── 06_emergence_from_distribution/

DTDR_RAG_double_transform_demo.ipynb

Patent & Commercial Licensing

DTDR is the subject of a filed UK patent application:

UK Patent Application No. GB2602157.6

This repository is provided for research and evaluation purposes.

For commercial licensing, strategic partnerships, or IP inquiries:

Contact: dtdr@multiverse1.com


See LICENSE_NOTICE.md for evaluation terms.
