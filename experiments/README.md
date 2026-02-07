# DTDR Experiments

This directory contains a curated set of **claim-aligned, runnable experiments**
demonstrating the properties of the **Distributed Transform-Domain Representation (DTDR)**.

DTDR is treated throughout as a **persistent, on-disk numerical representation**
that remains *compute-capable*: inference, similarity search, and approximate
nearest-neighbour (ANN) traversal can be performed directly in the transform domain,
without reconstructing full-precision floating-point data.

Each experiment is self-contained and accompanied by a focused README describing:
- purpose and hypothesis,
- methodology,
- and key results.

Readers are encouraged to begin with the **Primary Demonstration** below.

---

## Primary Demonstration (Start Here)

### DTDR RAG Demo with Composite Transforms

**File:** `DTDR_RAG_double_transform_demo.ipynb`  
**Location:** this directory

This notebook provides a **fully self-contained, human-verifiable demonstration**
of DTDR behaviour using retrieval-style (RAG-like) similarity search over
public-domain texts (*Alice in Wonderland*, *Sherlock Holmes*).

It demonstrates:

- DTDR as a **persistent, compute-capable file representation**
- direct similarity computation in the DTDR domain
- single and composite DTDR constructions (Hadamard, Hadamard + DCT)
- graceful degradation of retrieval quality under simulated corruption
- comparison against a no-transform INT8 baseline

The notebook:
- downloads required data automatically,
- runs end-to-end in a clean Python environment,
- produces both tabular results and degradation plots.

This is the recommended **first point of contact** for understanding DTDR.

---

## Experiment Index

### Experiment 01 — Functional Reconstruction of a DTDR-Stored Model

**Location:** `01_model_inference/`

Demonstrates that a numerical model stored persistently in DTDR form can be
reconstructed to a numerically different but **functionally equivalent**
representation suitable for inference.

Focus:
- DTDR as a persistent on-disk representation
- functional equivalence vs numerical fidelity
- robustness under controlled coefficient perturbation

See: `01_model_inference/README.md`

---

### Experiment 02 — End-to-End ANN Search in the DTDR Domain

**Location:** `02_dtdr_end_to_end_search/`

Demonstrates a complete approximate nearest-neighbour (ANN) pipeline operating
*entirely* in the DTDR domain, integrating:

- IVF (inverted file indexing),
- HNSW graph traversal,
- binary distance estimation (RaBitQ-like),
- DTDR-specific **multi-resolution dilution evidence** for coarse localisation.

Focus:
- DTDR as a native computational domain
- recall / latency trade-offs
- novel ANN signals enabled by distributed representations

See: `02_dtdr_end_to_end_search/README.md`

---

### Experiment 03 — Embedding Similarity Search in the DTDR Domain

**Location:** `03_embedding_search/`

Demonstrates similarity search over embedding vectors represented, stored,
and queried directly in the DTDR domain.

Focus:
- DTDR representations of embedding spaces
- similarity preservation under orthogonal transforms
- functional equivalence of DTDR-domain search results

See: `03_embedding_search/README.md`

---

### Experiment 04 — Graceful Degradation Under Perturbation

**Location:** `04_graceful_degradation/`

Explores DTDR behaviour under controlled perturbation, partial coefficient loss,
and on-disk corruption.

Includes:
- coefficient dropout experiments,
- block-loss simulations,
- file-on-disk corruption sweeps (DTDR vs FP16).

Focus:
- robustness to partial data loss
- smooth degradation vs catastrophic failure
- contrast with localised floating-point representations

See: `04_graceful_degradation/README.md`

---

### Experiment 05 — Storage Accounting and Residual Compressibility

**Location:** `05_storage_accounting/`

Provides transparent accounting of DTDR persistent storage footprint relative to
conventional floating-point and quantised formats.

Includes analysis of **residual lossless compressibility** (e.g. ZIP), highlighting
structural differences between DTDR representations and terminal encodings
such as GGUF.

Focus:
- DTDR storage size
- secondary lossless compression potential
- evidence that DTDR is not entropy-saturated compression

See: `05_storage_accounting/README.md`

---

## Suggested Reading Order

1. **Primary Demonstration**  
   Concrete, human-verifiable illustration of DTDR behaviour.

2. **Experiment 01**  
   Establishes DTDR as a persistent representation that preserves computation.

3. **Experiment 02**  
   Shows consequences of treating DTDR as a primary computational domain.

4. **Experiments 03 & 04**  
   Explore similarity search and robustness under degradation.

5. **Experiment 05**  
   Quantifies storage characteristics and residual structure.

Additional experiments may be added as the framework evolves.
