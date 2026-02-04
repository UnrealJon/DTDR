# Experiment 01 — Functional Reconstruction of a DTDR-Stored Model

## Purpose

This experiment demonstrates that a numerical model stored in a **Distributed Transform-Domain Representation (DTDR)** can be reconstructed to a *numerically different but functionally equivalent* representation suitable for inference and further computation.

The objective is to establish that DTDR constitutes a **persistent numerical representation**, rather than a conventional compression scheme.

---

## Method Overview

1. A reference floating-point tensor representation is taken as the baseline.
2. The same numerical data are stored in a DTDR form using a structured orthogonal transform and quantisation.
3. The DTDR representation is reconstructed to a working numerical precision.
4. Functional equivalence between the baseline and reconstructed representations is evaluated using similarity metrics.
5. A controlled perturbation is applied to the DTDR coefficients to assess robustness.

All comparisons are performed under identical shapes and execution conditions.

---

## Observations

The reconstructed DTDR representation exhibits:

- Cosine similarity extremely close to unity relative to the floating-point baseline.
- Small relative L2 error, consistent with numerical reconstruction rather than exact equality.
- Only marginal degradation under partial coefficient corruption, indicating that information is distributed across coefficients rather than localised.

These observations are consistent with a representation that preserves computationally relevant structure while permitting reduced-precision storage.

---

## Execution Context

Inference is performed on reconstructed FP16 weights loaded into the
standard Hugging Face model architecture.

- Execution device: CPU  
- Environment: Python virtual environment (`.venv`)
- Script: `mistral7b_reconstruction_inference.py`

This experiment evaluates **functional equivalence**, not inference throughput. Performance benchmarking and GPU execution are addressed separately.

---

## Scope and Limitations

- This experiment does not claim bit-wise equivalence.
- No specialised inference kernels or custom execution frameworks are required.
- The DTDR representation is reconstructed prior to evaluation; direct computation in the transformed domain is addressed in other experiments.

---

## Conclusion

This experiment demonstrates that DTDR enables persistent storage of numerical model parameters while allowing reconstruction to a numerically different but functionally equivalent representation. The results support the claim that DTDR preserves computational functionality and exhibits robustness consistent with a distributed representation.

