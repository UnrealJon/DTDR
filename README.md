# Experiment 06 — Emergence from Distributed Representation

## Purpose

This experiment investigates how inference capability appears as increasing fractions
of DTDR coefficients are reconstructed.

Unlike hierarchical compression (e.g. JPEG), DTDR does not preserve meaning at low fractions.
Instead, semantic behaviour emerges collectively once a sufficient portion of the distributed
representation is present.

## What this demonstrates

DTDR behaves as a **globally coherent representation** rather than a coarse-to-fine approximation.

Small subsets of coefficients do not contain partial meaning.
However, behaviour appears abruptly once enough coefficients accumulate.

This is characteristic of distributed associative memory systems.

## Running

python progressive_emergence_demo.py --pkl compressed_mistral_7b.pkl --cache-dir ./hf_cache


Outputs:

- emergence_curve.csv
- emergence_curve.png

## Expected Result

An S-shaped curve:
- near-zero similarity at low fractions
- rapid transition
- convergence near full reconstruction

This shows inference is a collective property of the representation.
