# Experiment 05 — DTDR Storage Accounting

## Purpose

This experiment provides transparent accounting of the **persistent storage footprint**
of numerical data stored in a **Distributed Transform-Domain Representation (DTDR)**,
relative to conventional floating-point and quantised representations.

The objective is to quantify storage reduction while clarifying the role of DTDR as a
**persistent stored representation**, distinct from both runtime inference formats and
conventional compression schemes.

---

## Method Overview

1. A reference model is stored in a conventional floating-point format (FP16).
2. The same numerical parameters are stored in DTDR form using a structured orthogonal
   transform and INT8 quantisation.
3. Persistent storage sizes are compared.
4. Residual compressibility under standard lossless compression (ZIP) is evaluated.
5. The reconstruction pathway to working numerical precision is described qualitatively.

No runtime performance benchmarking or inference timing is performed in this experiment.

---

## Observations

- DTDR enables a substantial reduction in persistent storage relative to FP16
  representations.
- DTDR representations are stored as the **primary persistent format**, not as a
  transient preprocessing artefact.
- Reconstruction to working numerical precision is performed once at initialisation
  and does not require specialised execution kernels.
- After reconstruction, standard inference infrastructure may be used unchanged.

These observations support the claim that DTDR decouples **persistent storage format**
from **working computational representation**.

---

## Secondary Lossless Compression

DTDR representations retain structured statistical regularities following transform
and quantisation. As a result, DTDR-stored parameter files exhibit substantial
additional size reduction when subjected to conventional **lossless compression**
tools such as ZIP.

By contrast, conventional FP16 and INT8 formats (e.g. GGUF-style representations)
are typically close to entropy-saturated and exhibit little residual compressibility.

Importantly:

- DTDR does **not** rely on entropy coding.
- Lossless compression is **optional and orthogonal** to DTDR.
- All reported DTDR storage reductions are achieved **prior to** any ZIP compression.

Where applied, secondary compression further reduces persistent storage and
distribution size without affecting reconstruction or computation.

---

## Scope and Limitations

- This experiment does not attempt to optimise secondary compression.
- Runtime performance and inference throughput are addressed in other experiments.
- Storage figures are representative and depend on model architecture, transform
  choice, and quantisation parameters.

---

## Conclusion

This experiment demonstrates that DTDR provides reduced-footprint **persistent storage**
of numerical model parameters while allowing reconstruction to a working numerical
precision suitable for standard inference workflows.

In addition, DTDR representations retain residual structure that enables substantial
optional lossless compression, further reducing storage and transmission size.
