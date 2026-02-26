# 06 – Behaviour Under Partial Completeness

This folder characterises how DTDR representations behave when only a fraction of transform coefficients is available.

It demonstrates that DTDR exhibits **two distinct utility regimes** under progressive coefficient incompleteness:

1. **Geometric regime** – similarity search over embeddings  
2. **Functional regime** – generative inference using model weights  

These regimes behave differently because they arise from different mathematical structures (linear projection vs nonlinear constraint satisfaction).

---

# Overview

Let:

\[
\rho = \frac{\text{number of available coefficients}}{\text{total coefficients}}
\]

denote the **coefficient completeness ratio**.

We progressively restrict access to coefficients and measure semantic utility.

---

# Regime A — Geometric Utility (Embeddings)

**File:** `urho_experiment.py`  
**Datasets:**  
- SIFT1M (geometric benchmark)  
- GloVe-100d (semantic benchmark)

**Metrics:**
- Cosine similarity to original vectors
- Recall@10

## Result

Across both datasets and dropout modes:

\[
U(\rho) \approx \rho
\]

Semantic utility scales approximately linearly with coefficient completeness.

This holds for:
- Random coefficient dropout
- Block (contiguous shard-loss) dropout

There is **no catastrophic collapse** for similarity tasks.

This is the **geometric regime**.

---

# Regime B — Functional Emergence (Model Weights)

**Files:**
- `progressive_emergence_demo.py`
- `progressive_emergence_demo_v2.py`

These experiments apply DTDR-style partial reconstruction to stored neural network weights.

## Result

Inference quality exhibits nonlinear emergence behaviour:

- Low ρ → incoherent output
- Intermediate ρ → unstable output
- Above threshold → coherent generative inference

This is not linear degradation. It reflects nonlinear constraint satisfaction in deep networks.

This is the **functional regime**.

---

# Interpretation

DTDR induces two mathematically distinct behaviours:

| Property | Geometric Regime | Functional Regime |
|-----------|-----------------|------------------|
| Operation type | Linear | Nonlinear |
| Task | Similarity search | Generative inference |
| Utility scaling | Proportional | Emergence-like |
| Collapse threshold | No | Yes |

These behaviours are not contradictory.

Similarity preservation is governed by linear projection scaling.

Generative coherence depends on nonlinear multi-layer constraint dynamics.

---

# Why This Matters

The geometric regime enables:

- Proportional confidentiality
- Cost-scaling exfiltration resistance
- Quantitative completeness gating
- Secure vector database search without homomorphic overhead

The functional regime suggests:

- Model weight extraction may resist partial exposure more strongly
- Emergence dynamics arise naturally in nonlinear inference systems

---

# Running the Experiments

## U(ρ) Experiment

```bash
python urho_experiment.py

Outputs:

JSON result files

Console tables

Plot-ready data

Requires:

SIFT1M dataset

GloVe 100d embeddings

Adjust paths at top of script.

python progressive_emergence_demo_v2.py
Requires:

Pre-compressed model weights (see repo root)

HuggingFace cache directory



