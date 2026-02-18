# DTDR — Distributed Transform-Domain Representation

DTDR is a computational memory architecture.
It maintains high-dimensional data in quantised transform-domain form
as the **primary working representation**, not as a compressed archive
that must be decompressed before use.

This distinction matters. Conventional compression reduces storage cost
but adds a decode step before every operation. DTDR eliminates that step:
computation happens directly on the stored form.

**UK Patent Application GB2602157.6**

---

## What DTDR does

A Hadamard (or composite) transform distributes each vector's information
across all coefficients. The transformed coefficients are then quantised
and stored. From that point on, distance calculations, nearest-neighbour
search, and neural network inference operate on the quantised
transform-domain data without reconstruction.

Three properties follow from this design:

1. **Direct computation** — dot products and cosine similarity are
   evaluated in the transform domain. No decompression path exists
   because none is needed.

2. **Graceful degradation** — corrupting stored coefficients degrades
   output smoothly rather than causing catastrophic failure, because
   no single coefficient carries isolated meaning.

3. **Emergence under truncation** — removing coefficients does not
   produce a weaker model. Instead, behaviour is absent until a
   critical fraction is present, then appears rapidly. The representation
   stores distributed constraints, not localised features.

These properties are not theoretical claims. Each is demonstrated
experimentally in this repository.

---

## Key results

### Inference acceleration (Mistral-7B)

DTDR-compressed weights running on a 7-billion-parameter language model:

| Metric | Result |
| ------ | ------ |
| GPU throughput | 2× faster than INT8 quantisation |
| DRAM traffic | 5× reduction vs INT8 |
| Output quality | Functionally equivalent |

### Retrieval routing (GloVe, 200k vectors)

DTDR used as a routing layer ahead of conventional ANN search:

| Metric | Result |
| ------ | ------ |
| Distance evaluations | 10×–40× fewer than conventional IVF |
| Shard fan-out | 2 partitions vs 20–100 |
| Memory bandwidth | 5×–50× reduction (scales with dimension) |
| hit@1 | Approaches 1.0 with local refinement only |

Recall improves by increasing local compute, not by searching
more partitions. This converts distributed search into a local
ranking problem.

### Emergence threshold (Mistral-7B under truncation)

Progressive reconstruction of a DTDR-stored model reveals a phase
transition rather than gradual improvement:

| Coefficients present | Cosine similarity to full model |
| -------------------- | ------------------------------- |
| < 10% | ≈ −0.3 (anti-correlated) |
| 20% | 0.32 |
| 60% | 0.59 |
| 80% | 0.74 |
| 100% | 1.00 |

![Emergence curve](experiments/06_emergence/emergence_curve.png)

Behaviour does not degrade proportionally. It is absent, then present.
This is consistent with a distributed constraint system —
analogous to GPS triangulation, where position does not exist
until enough satellites are available.

### Dilution evidence (novel ANN signal)

DTDR generates a signal not available in raw embedding space:
the rate at which similarity concentrates or disperses under
progressive aggregation of transform-domain coefficients.
This signal is **orthogonal** to existing ANN optimisations and
can be used to improve routing decisions without modifying the
underlying search index.

---

## Why DTDR is not compression

Compression optimises **reconstruction fidelity** — the goal is to
recover the original data as closely as possible.

DTDR optimises **functional equivalence** — the goal is to preserve
computational outcomes (distances, rankings, inference results)
while the data remains in its stored form.

This is a different objective with different engineering consequences.
A compressed model must be decompressed to run. A DTDR-stored model
runs as stored.

---

## Repository structure

```
experiments/
  01_mistral_7b/       — Inference acceleration benchmarks
  02_dilution/         — Novel ANN signal discovery
  03_routing/          — IVF routing with constant fan-out
  04_corruption/       — Graceful degradation under bit-level damage
  05_storage/          — Storage accounting and residual compression
  06_emergence/        — Emergence threshold under truncation
```

Each experiment folder contains its own README, code, and results.

---

## Running the experiments

Requirements vary by experiment. Most need Python, NumPy, and PyTorch.
The Mistral-7B experiments require a CUDA GPU and ~16 GB VRAM.
See individual experiment READMEs for specific instructions.

---

## Commercial applications

DTDR is applicable wherever high-dimensional vectors are stored,
transmitted, or compared at scale:

- **Vector databases** — routing layer reducing shard fan-out
  by an order of magnitude
- **LLM inference** — weight storage enabling direct computation
  with reduced memory traffic
- **Edge deployment** — smaller memory footprint without
  decompression overhead
- **Retrieval-augmented generation** — faster candidate selection
  in the ANN stage of RAG pipelines

The improvements are orthogonal to existing optimisations
(quantisation, pruning, knowledge distillation) and can be
composed with them.

---

## IP and licensing

DTDR is the subject of UK Patent Application GB2602157.6.

This repository is published for evaluation and research purposes.
Production use requires a commercial licence.

For licensing, strategic partnerships, or technical discussion:
**dtdr@multiverse1.com**

---

## Citation

```bibtex
@misc{west2024dtdr,
  title   = {Distributed Transform-Domain Representation for
             Neural Network Storage and Computation},
  author  = {West, Jonathan},
  year    = {2024},
  note    = {UK Patent Application GB2602157.6},
  url     = {https://github.com/UnrealJon/DTDR}
}
```

---

## Author

**Jonathan West**
Independent researcher. Formerly NHS Consultant in Obstetrics
and Gynaecology, Royal Devon & Exeter NHS Foundation Trust.
Cambridge physiology background; self-taught in machine learning.
