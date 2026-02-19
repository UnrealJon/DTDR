# Emergence & Guided Reconstruction Experiments

This folder contains experiments demonstrating how DTDR-stored models behave when **partially known**.

The central question is:

> Does a DTDR model fail because information is missing, or because the reconstruction problem is under-constrained?

The experiments progressively reconstruct a DTDR-compressed model using only a fraction of its stored coefficients and compare different ways of filling the missing portions.

---

## Concept

Three completion strategies are tested:

| Method      | Meaning               |
| ----------- | --------------------- |
| Zero fill   | No prior knowledge    |
| Random fill | Incorrect prior       |
| Guided fill | Weak compatible prior |

Observed behaviour:

| Condition            | Behaviour                            |
| -------------------- | ------------------------------------ |
| Too few coefficients | No coherent output                   |
| Random completion    | Stable nonsense                      |
| Weak aligned prior   | Gradual recovery of correct language |

This demonstrates:

> DTDR models behave like constraint systems rather than parameter lists.

The model is not merely degraded when incomplete — it is **underdetermined**.
Providing even a weak compatible prior stabilises reconstruction.

---

## Files

### `progressive_emergence_demo.py`

Original emergence experiment.

Shows the sharp functional threshold when reconstructing a DTDR model using only a fraction of stored coefficients.

Purpose:

* Demonstrates emergence behaviour
* Establishes difference between truncation and corruption

---

### `progressive_emergence_demo_v2.py`

Guided reconstruction experiment.

Extends the first experiment by introducing a weak aligned prior (slightly corrupted model) to guide reconstruction.

Purpose:

* Tests whether failure is caused by missing data or missing constraints
* Demonstrates stabilisation from compatible priors

---

## Requirements

* Python ≥ 3.10
* PyTorch
* Transformers
* safetensors
* A DTDR-compressed Mistral-7B `.pkl` file

The first run will automatically download the base Mistral tokenizer/model into a HuggingFace cache directory.

---

## Running the Experiments

### 1) Baseline — Emergence Threshold

Collapse when insufficient coefficients exist.

```bash
python progressive_emergence_demo_v2.py \
  --pkl "mistral_7b_compressed_2x.pkl" \
  --mode zero
```

---

### 2) Random Completion — Incorrect Constraints

Produces stable but meaningless output.

```bash
python progressive_emergence_demo_v2.py \
  --pkl "mistral_7b_compressed_2x.pkl" \
  --mode random
```

---

### 3) Guided Reconstruction — Weak Prior

Gradual recovery of coherent language.

```bash
python progressive_emergence_demo_v2.py \
  --pkl "mistral_7b_compressed_2x.pkl" \
  --anchor-pkl "compressed_mistral_7b_f1e-06_s0.pkl" \
  --mode guided
```

---

## Output

The script produces:

* Console text samples at each reconstruction fraction
* Cosine similarity measurements
* Top-token overlap metrics
* Emergence curve plots (`.png` + `.csv`)

---

## Interpretation

These experiments show a fundamental distinction:

| Representation type  | Behaviour when incomplete              |
| -------------------- | -------------------------------------- |
| Conventional weights | degraded but functional                |
| DTDR                 | non-functional until constraints close |

Adding a weak compatible prior converts the problem from:

> unsolvable → solvable

This indicates DTDR stores a **solution manifold**, not independent parameters.

---

## Why This Matters

This behaviour explains multiple DTDR properties:

* graceful degradation under corruption
* sharp emergence threshold
* recovery using weak priors
* stability under noise

The experiments therefore test the reconstruction dynamics of the representation, not compression performance.
