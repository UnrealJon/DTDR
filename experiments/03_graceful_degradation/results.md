Results — Experiment 03: Graceful Degradation

Summary



This document records the results obtained from Experiment 03, which evaluates the robustness of DTDR-encoded model parameters under controlled partial corruption of their stored transform-domain representation.



The experiment measures functional similarity between a floating-point baseline model and a DTDR-stored model subjected to increasing levels of bit-level corruption.



Experimental Conditions



Model architecture: minimal transformer testbed



Parameter storage: DTDR using a structured orthogonal transform with INT8 quantisation



Corruption model: random bit-level flips applied to a specified fraction of stored DTDR coefficients



Inference: DTDR execution pathway (no retraining, no error correction)



Metrics:



Cosine similarity between outputs



Relative L2 error with respect to the floating-point baseline



Each corruption level is evaluated independently, with the DTDR representation re-initialised prior to corruption.



Results Table

Corruption Fraction	Cosine Similarity	Relative L2 Error

0.0%	0.999998	0.002108

0.1%	0.999826	0.018658

0.5%	0.999144	0.041426

1.0%	0.998085	0.062023

2.0%	0.996241	0.087020



The full numerical output is also saved in machine-readable form as:



dtdr\_corruption\_sweep.csv



Observations



Both cosine similarity and relative L2 error degrade smoothly and monotonically as the corruption fraction increases.



No abrupt or catastrophic loss of functional similarity is observed, even when a non-trivial fraction of stored DTDR coefficients is corrupted.



At 2% random corruption of stored INT8 DTDR weights, output cosine similarity remains above 0.996 relative to the floating-point baseline.



These results are consistent with a representation in which information is distributed across coefficients, such that localised corruption produces proportional rather than catastrophic effects.



Interpretation



The observed behaviour supports the claim that DTDR-encoded model parameters exhibit graceful degradation under data corruption. In contrast to representations where parameters are stored in localised or block-structured form, DTDR distributes information across the transformed domain, resulting in predictable and proportional loss of fidelity.



Importantly, this robustness is demonstrated at the model-parameter level, without retraining, redundancy, or specialised error-correction mechanisms.



Notes



Absolute metric values depend on model size and architecture; the monotonic trend is the primary observation of interest.



This experiment is not intended to benchmark accuracy on downstream tasks, but rather to characterise storage-level robustness.

