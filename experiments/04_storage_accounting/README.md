# Experiment 04 — DTDR Storage Accounting

## Purpose

This experiment provides transparent accounting of the **persistent
storage footprint** of numerical data stored in a
**Distributed Transform-Domain Representation (DTDR)**, relative to
conventional floating-point representations.

The objective is to quantify storage reduction while clarifying the role
and cost of reconstruction to working numerical precision.

---

## Method Overview

1. A reference model is stored in a conventional floating-point format
   (FP16).
2. The same numerical parameters are stored in a DTDR form using a
   structured orthogonal transform and quantisation.
3. Persistent storage sizes are compared.
4. The reconstruction pathway is described qualitatively.

No performance benchmarking or inference timing is performed in this
experiment.

---

## Observations

- DTDR enables a substantial reduction in persistent storage relative to
  FP16 representations.
- Reconstruction to working numerical precision is performed once at
  initialisation and does not require specialised execution kernels.
- After reconstruction, standard inference infrastructure may be used
  unchanged.

These observations support the claim that DTDR decouples *persistent
storage format* from *working computational representation*.

---

## Scope and Limitations

- This experiment does not optimise storage beyond the demonstrated
  configuration.
- Runtime performance and throughput are addressed in other experiments.
- Storage figures are representative and depend on model architecture
  and quantisation choice.

---

## Conclusion

This experiment demonstrates that DTDR provides reduced-footprint
persistent storage of numerical model parameters while allowing
reconstruction to a working numerical precision suitable for standard
inference workflows.
