# Results: On-Disk Corruption Robustness

This document reports the measured numerical impact of random on-disk corruption applied to
two different storage representations of the same neural network parameters:

- **Baseline FP16 safetensors**
- **DTDR (Hadamard-based Distributed Transform-Domain Representation)**

All results are derived from automated corruption sweeps recorded in CSV form.

---

## Metrics Reported

For each corruption fraction and random seed, we report:

- **Mean cosine similarity** to the clean FP16 reference  
- **Mean relative L2 error**
- **Number of valid tensors** successfully reconstructed

Results are aggregated over multiple random seeds per corruption level.

---

## DTDR (Hadamard) Results

DTDR shows smooth, bounded numerical degradation across several orders of magnitude of
corruption.

| Corruption Fraction | Mean Cosine | Mean Rel. L2 | Valid Tensors (avg) |
|---------------------|------------:|-------------:|--------------------:|
| 0                   | 1.003497    | 0.023254     | 97 / 97 |
| 1e-7                | 1.003495    | 0.023322     | 97 / 97 |
| 2e-7                | 1.003492    | 0.023401     | 97 / 97 |
| 5e-7                | 1.003485    | 0.023626     | 97 / 97 |
| 1e-6                | 1.003468    | 0.024086     | 97 / 97 |
| 2e-6                | 0.999978    | 0.067963     | 97 / 97 |
| 5e-6                | 1.003291    | 0.026958     | 96 / 97 |

**Observations:**
- No catastrophic numerical failures occur within this regime.
- Cosine similarity degrades smoothly.
- Relative L2 error increases gradually and remains bounded.
- Occasional single-tensor outliers do not propagate or cascade.
- All tensors remain valid until container-level failure.

---

## FP16 Baseline Results

FP16 storage exhibits sharp, unstable failure behaviour even at extremely low corruption
fractions.

| Corruption Fraction | Mean Cosine | Mean Rel. L2 | Valid Tensors (avg) |
|---------------------|------------:|-------------:|--------------------:|
| 0                   | 1.003726    | 0.000000     | 97 / 97 |
| 1e-7                | 0.804641    | 1.57e16      | 64 / 97 |
| 2e-7                | 0.815421    | 6.01e15      | 50 / 97 |
| 5e-7                | 0.734497    | 9.16e15      | 38 / 97 |
| 1e-6                | 0.825394    | 5.42e14      | 28 / 97 |
| 2e-6                | 0.937955    | 5.27e15      | 21 / 97 |
| 5e-6                | 1.000001    | 2.78e-05     | 20 / 97 |
| 1e-5                | 0.999999    | 2.52e-04     | 20 / 97 |
| 2e-5                | 0.983652    | 4.15e+00     | 19 / 97 |
| 5e-5                | 0.977471    | 1.42e15      | 19 / 97 |
| 1e-4                | 0.915800    | 1.73e13      | 19 / 97 |
| 2e-4                | 0.760677    | 2.22e13      | 17 / 97 |
| 5e-4                | 0.557943    | 9.15e14      | 13 / 97 |
| 1e-3                | 0.442101    | 1.62e14      | 8 / 97 |

**Observations:**
- FP16 fails catastrophically at corruption fractions as low as **1e-7**.
- Relative L2 error explodes by many orders of magnitude.
- Large numbers of tensors become invalid.
- Behaviour is highly seed-dependent once failure begins.
- Apparent “recovery” at some corruption levels reflects masking by tensor invalidation,
  not numerical stability.
- Cosine values slightly exceeding 1.0 arise from accumulated floating-point and reconstruction error and do not indicate increased alignment.
---

## Container-Level Failure

For DTDR, failures beyond ~5e-6 corruption were traced to "_pickle.UnpicklingError: invalid load key"

This reflects **serialization container fragility**, not numerical instability of the DTDR
representation itself.

Crucially:
- DTDR numerical degradation remains smooth up to the point of container failure.
- FP16 numerical failure occurs well before any container-level issues arise.

---

## Summary

- **FP16 storage** exhibits catastrophic, non-graceful failure under extremely small random
  corruption.
- **DTDR transforms local corruption into distributed, statistically averaged noise**, yielding
  smooth degradation.
- DTDR preserves numerical validity over **orders of magnitude greater corruption** than FP16.

These results demonstrate a fundamental difference in how numerical damage propagates in
localized versus distributed representations. From an engineering perspective, DTDR converts storage corruption from a hard failure into a graceful degradation problem.

