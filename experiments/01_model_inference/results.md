# Experiment 01 — Results Summary  
Functional Reconstruction from DTDR

## Overview

This experiment evaluates whether a numerical model representation stored
in a **Distributed Transform-Domain Representation (DTDR)** can be
reconstructed to a *numerically different but functionally equivalent*
working precision suitable for inference.

The results reported here correspond to execution of
`DTDR_inference_demo.py` using a DTDR-INT8 persistent representation and
a floating-point baseline.

---

## Numerical Equivalence Metrics

The following metrics compare outputs produced using:

- a floating-point baseline representation, and
- a DTDR-stored representation reconstructed to working precision.

Cosine similarity (DTDR vs FP): 0.9999980926513672
Relative L2 error (DTDR vs FP): 0.002108924090862274


These values indicate extremely close functional agreement despite
numerical non-identity.

---

## Robustness Under Partial Corruption

A controlled perturbation was applied to a small fraction of DTDR
coefficients prior to reconstruction.

### Corruption parameters
- Fraction of coefficients perturbed: 0.001 (0.1%)

### Resulting metrics
Cosine similarity (corrupt vs FP): 0.999997615814209
Relative L2 error (corrupt vs FP): 0.0023368862457573414


The results show only marginal degradation relative to the uncorrupted
DTDR case.

---

## Observations

- DTDR reconstruction yields a representation that is numerically
  different from the floating-point baseline but functionally equivalent.
- Similarity metrics remain extremely high.
- Partial corruption produces smooth, proportional degradation rather
  than catastrophic failure.

These behaviours are consistent with a representation in which
information is distributed across coefficients rather than localised.

---

## Notes on Storage and Execution

- The DTDR representation is stored persistently in quantised form.
- Reconstruction to working numerical precision is performed prior to
  inference.
- Reconstruction is not part of the steady-state inference loop.
- No specialised execution kernels are required following reconstruction.

Persistent storage accounting is documented separately in
`04_storage_accounting`.

---

## Scope

This results summary focuses on **functional equivalence and robustness**.
Inference throughput, GPU execution, and system-level performance are
addressed in separate experiments and demonstrations.






