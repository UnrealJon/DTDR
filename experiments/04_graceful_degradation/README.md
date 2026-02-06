Experiment 04 — Graceful Degradation of DTDR-Stored Model Parameters
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

How to Run

From a Python environment with PyTorch installed:

python dtdr\_corruption\_sweep.py



The script is fully self-contained and requires no additional configuration.

On completion, it will:

print similarity metrics for each corruption level to the console

write a CSV file containing the results:

dtdr\_corruption\_sweep.csv



\### Disk Corruption Robustness



The subfolder `dtdr\_disk\_corruption/` contains a storage-level robustness study

examining the effect of random on-disk corruption on FP16 versus DTDR model

representations.



This experiment demonstrates that DTDR degrades smoothly under corruption,

whereas FP16 exhibits catastrophic numerical failure at extremely small

corruption fractions.



See `dtdr\_disk\_corruption/README.md` and `RESULTS.md` for details.



