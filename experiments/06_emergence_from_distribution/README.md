# Experiment 06 — Emergence vs Graceful Degradation in DTDR

## Purpose

This experiment investigates how neural network behaviour changes when a model
stored in **DTDR (Distributed Transform Domain Representation)** is only partially available.

Two separate properties are studied:

1. **Graceful degradation under corruption**
2. **Emergence of inference under truncation**

At first glance these appear contradictory.

They actually reveal the defining characteristic of DTDR:

> The model’s function is stored as a distributed constraint system rather than as localised parameters.

---

## Background

Typical neural network storage behaves locally:

- Removing parameters weakens behaviour
- Some weights matter more than others
- A smaller model behaves like a weaker model

DTDR changes this.

Instead of storing meaning in specific weights, the representation distributes
semantic information across many mutually dependent coefficients.

The question becomes:

> What happens if we *damage* the information vs *remove* the information?

---

## The Two Experiments

### 1) Corruption Sweep — Graceful Degradation

We progressively corrupt stored DTDR coefficients while keeping all of them present.

This simulates:

- storage damage
- quantisation
- transmission noise

| Damage | Behaviour |
|------|------|
Small | Slightly worse output |
Medium | Noticeably degraded output |
Large | Very poor output |
Extreme | Failure |

**Behaviour declines smoothly.**

The model still “knows what it is trying to say”, but imperfectly.

---

### 2) Truncation — Emergence Threshold

We reconstruct only a fraction of coefficients.

This simulates:

- partial download
- missing shards
- incomplete reconstruction

| Fraction present | Behaviour |
|------|------|
< 20% | No language at all |
20–60% | Unstable fragments |
60–80% | Partial linguistic structure |
> 80% | Rapid stabilisation |
100% | Correct model |

**Behaviour does NOT degrade smoothly — it appears suddenly.**

---

## Why These Results Are Different

The two tests modify different properties of the representation.

| Experiment | Mathematical effect | Result |
|------|------|------|
Corruption | Perturb constraints | Degraded solution |
Truncation | Remove constraints | No solution → Solution |

---

## Intuition: GPS Triangulation

Corruption experiment:
> Move satellites slightly → position estimate drifts

Truncation experiment:
> Remove satellites → no position exists until enough satellites are available

DTDR behaves like triangulation — language exists only once enough global constraints are present.

---

## What This Shows About DTDR

DTDR is **not a hierarchical encoding** (like JPEG).

Hierarchical encoding:
> Coarse meaning first, details later

DTDR encoding:
> Meaning exists only collectively

No subset of coefficients independently stores interpretable behaviour.

---

## Practical Interpretation

| Representation | Partial Data Behaviour |
|------|------|
Standard weights | Weaker model |
Compressed | Degraded model |
DTDR | No model → Model transition |

DTDR stores **consistency conditions**, not individual features.

---

## Running the Experiment

### Requirements

- Python
- PyTorch
- Transformers
- matplotlib

### Command

From this folder:

python progressive_emergence_demo.py --pkl ../compressed_mistral_7b.pkl --cache-dir ../hf_cache



### Outputs
emergence_curve.csv
emergence_curve.png



The plot shows cosine similarity to the full model.

Expected: an S-curve showing sudden emergence of inference.

---

## What the Curve Means

Low fractions:
> Insufficient constraints → unstable system

Critical region:
> Constraint closure begins

High fractions:
> Stable attractor (language) forms

The model does not gradually improve — it becomes *possible*.

---

## Relationship to Graceful Degradation

These results are complementary, not contradictory.

| Property | What it tests | Meaning |
|------|------|------|
Graceful degradation | Stability of solution | Robustness |
Emergence threshold | Existence of solution | Distributed encoding |

Together they indicate:

> DTDR stores function globally and reconstructs behaviour only when consistency conditions are satisfied.

---

## Significance

This experiment demonstrates that DTDR does not store behaviour in identifiable components.

Instead:

- Individual coefficients are not meaningful
- Behaviour emerges from collective consistency
- Incomplete representations cannot act as weaker models

This distinguishes DTDR from compression, pruning, or quantisation.

---

## Summary

**Graceful degradation:**  
The model still knows what to say, but says it badly.

**Emergence threshold:**  
The model does not know what to say until enough of itself exists.

DTDR therefore represents models as coherent distributed constraint systems rather than collections of functional parts.






