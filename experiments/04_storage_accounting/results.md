# Experiment 04 — Storage Accounting Results

## Model

- Model: Mistral-7B
- Baseline representation: FP16
- DTDR representation: structured orthogonal transform + INT8 quantisation

## Persistent Storage Footprint

- FP16 baseline size: approximately 14.5 GB
- DTDR persistent size: approximately 6.7–6.8 GB

This corresponds to a reduction in persistent storage of approximately
50–55% relative to the FP16 baseline.

## Reconstruction

- DTDR representations are reconstructed to working numerical precision
  prior to inference or computation.
- Reconstruction is performed once at model initialisation and is not
  part of the steady-state inference loop.
- No specialised execution kernels or custom inference frameworks are
  required following reconstruction.

## Notes

This experiment focuses on *persistent storage accounting* rather than
runtime performance or throughput optimisation.
