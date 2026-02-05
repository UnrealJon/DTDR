# DTDR Experiments

This directory contains a set of **focused, claim-aligned experiments and demonstrations**
illustrating the properties and capabilities of the Distributed Transform-Domain Representation (DTDR).

Each numbered subdirectory contains:
- a runnable reference implementation, and
- a self-contained README describing purpose, methodology, and results.

The files in this directory are intended to be read in the order suggested below.

---

## Primary Demonstration

The script: DTDR_RAG_double_transform_demo.py


located in this directory provides an end-to-end demonstration of DTDR as a
**functional numerical representation**, integrating:

- composite (double) DTDR constructions,
- direct similarity search in the transform domain,
- retrieval-style (RAG-like) workflows.

This script is intended as a **conceptual and functional demonstration**, rather than a
benchmarked experiment. Readers looking for a single, self-contained illustration of
DTDR behaviour may wish to begin here before exploring the numbered experiments below.

---

## Experiment Index

### Experiment 01 — Functional Reconstruction of a DTDR-Stored Model

**Location:** `experiments/01_model_inference/`

Demonstrates that a numerical model stored in DTDR form can be reconstructed to a
numerically different but **functionally equivalent** representation suitable for inference.

Focus:
- DTDR as a *persistent stored representation*
- functional equivalence versus reconstruction fidelity
- robustness under controlled coefficient perturbation

See: `experiments/01_model_inference/README.md`

---

### Experiment 02 — End-to-End ANN Search in the DTDR Domain

**Location:** `experiments/02_dtdr_end_to_end_search/`

Demonstrates an end-to-end approximate nearest-neighbour (ANN) search pipeline
operating *entirely* in the DTDR domain, integrating:

- IVF (inverted file indexing),
- HNSW graph traversal,
- binary distance estimation (RaBitQ-like),
- DTDR-specific multi-resolution dilution evidence for coarse localisation.

Focus:
- DTDR as a native computational domain
- ANN recall/latency trade-offs
- novel localisation signals enabled by distributed representations

See: `experiments/02_dtdr_end_to_end_search/README.md`

---

### Experiment 03 — Embedding Similarity Search in the DTDR Domain

**Location:** `experiments/03_embedding_search/`

Demonstrates similarity search over embedding vectors represented and queried
directly in the DTDR domain.

Focus:
- DTDR representations of embedding spaces
- cosine / similarity preservation under transform
- functional equivalence of DTDR-domain search results

See: `experiments/03_embedding_search/README.md`

---

### Experiment 04 — Graceful Degradation Under Perturbation

**Location:** `experiments/04_graceful_degradation/`

Explores the behaviour of DTDR representations under controlled perturbation,
corruption, or partial loss of transform-domain coefficients.

Focus:
- robustness to noise and partial data loss
- smooth degradation of functional behaviour
- contrast with localised or brittle representations

See: `experiments/04_graceful_degradation/README.md`

---

### Experiment 05 — Storage Accounting and Residual Compressibility

**Location:** `experiments/05_storage_accounting/`

Provides transparent accounting of DTDR persistent storage footprint relative
to conventional floating-point and quantised formats, including analysis of
residual lossless compressibility.

Focus:
- DTDR persistent storage size
- optional secondary lossless compression (ZIP)
- distinction between structured representations and terminal encodings

See: `experiments/05_storage_accounting/README.md`

---

## Conceptual Ordering

The experiments are ordered deliberately:

1. **Primary Demonstration**  
   Introduces DTDR as a functional and composable numerical representation.

2. **Experiment 01**  
   Establishes DTDR as a persistent representation that preserves computational functionality.

3. **Experiment 02**  
   Demonstrates the consequences of treating DTDR as a primary computational domain,
   including new ANN search strategies and signals.

4. **Experiments 03 & 04**  
   Explore functional behaviour in embedding spaces and robustness under perturbation.

5. **Experiment 05**  
   Quantifies persistent storage characteristics and residual structure.

Additional experiments may be added in future as the framework evolves.
