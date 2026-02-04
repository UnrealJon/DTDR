Experiment 03 — Graceful Degradation of DTDR-Stored Model Parameters
Purpose

This experiment evaluates the behaviour of a model whose parameters are stored in a Distributed Transform-Domain Representation (DTDR) when the stored representation is subjected to controlled partial corruption.

The objective is to demonstrate that DTDR exhibits graceful degradation: a smooth, proportional reduction in functional fidelity as corruption increases, rather than abrupt or catastrophic failure.

This experiment provides direct evidence of robustness at the model-parameter level, not merely at the level of embeddings or activations.

Method Summary

A reference floating-point model is instantiated and evaluated to establish a baseline output.

The model parameters are converted into a DTDR representation using a structured orthogonal transform and INT8 quantisation.

Controlled bit-level corruption is applied to a specified fraction of the stored DTDR coefficients.

Inference is performed using the corrupted DTDR representation via the standard DTDR execution pathway.

Output similarity between the corrupted DTDR model and the floating-point baseline is measured using cosine similarity and relative L2 error.

Corruption levels are swept over multiple fractions (0%–2%) to observe the progression of degradation.

Observations

Outputs produced from partially corrupted DTDR representations remain closely aligned with the floating-point baseline at low corruption levels.

Both cosine similarity and relative L2 error degrade smoothly and monotonically as the corruption fraction increases.

No abrupt loss of functionality is observed, even when a non-trivial fraction of DTDR coefficients is corrupted.

These observations are consistent with a representation in which information is distributed across coefficients rather than localised.

Scope and Limitations

This experiment focuses exclusively on robustness to partial corruption of stored model parameters.

It does not measure inference throughput, latency, or hardware-level performance.

The model architecture is intentionally minimal and serves only as a controlled testbed for storage-level robustness.

Conclusion

This experiment demonstrates that DTDR-encoded model parameters exhibit graceful degradation under controlled corruption of their transform-domain representation. The results support the use of DTDR as a robust persistent memory format for numerical model parameters, with predictable and proportional degradation characteristics under data loss or corruption.