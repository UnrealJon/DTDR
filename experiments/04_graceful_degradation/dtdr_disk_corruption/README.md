# DTDR Disk Corruption Robustness Experiment

This experiment investigates the robustness of different numerical representations of large
neural network parameters under **random on-disk corruption**.

Specifically, it compares:

- **Baseline FP16 storage** (safetensors)
- **DTDR (Distributed Transform-Domain Representation)** using a Hadamard transform

The focus of the experiment is **storage-level numerical robustness**, not inference accuracy.

---

## 1. Purpose

The aim of this experiment is to isolate and measure how numerical damage propagates when
stored model parameters are subjected to random local corruption on disk.

The central hypothesis is:

> **DTDR redistributes information across coefficients such that local corruption produces
graceful, statistically averaged degradation, whereas FP16 storage exhibits catastrophic,
non-graceful failure.**

This experiment is designed to test that hypothesis directly.

---

## 2. Conceptual Framing

We explicitly separate three layers:

1. **Storage container**  
   How bytes are serialized on disk (e.g. safetensors, pickle)

2. **Numerical representation**  
   FP16 vs DTDR (Hadamard-transformed, blockwise-quantised)

3. **Functional reconstruction**  
   Recovery of FP16 tensors for use by downstream systems

By separating these concerns, the experiment answers a precise question:

> *Given identical random corruption of stored bytes, how does numerical damage propagate
when the data is reconstructed?*

Inference accuracy is deliberately out of scope.

---

## 3. Model and Artefacts

- **Model:** Mistral-7B
- **Reference:** Clean FP16 safetensors shard
- **DTDR artefact:** Hadamard-transformed, blockwise-quantised DTDR representation  
  (previously validated for inference equivalence)

The same model parameters are used in both cases.

---

## 4. Corruption Model

Random byte-level corruption is applied directly to the stored files on disk.

- Corruption fraction swept from **1e-7 to 1e-3**
- Multiple random seeds per corruption level
- Identical corruption machinery used for both representations

Corruption is applied *after* serialization and *before* reconstruction.

---

## 5. Measurement Pipeline

For each corruption fraction and seed:

1. Corrupt the stored file on disk
2. Reconstruct FP16 tensors
   - Direct load (FP16 baseline)
   - DTDR → FP16 reconstruction
3. Compare reconstructed tensors against clean FP16 reference
4. Record numerical metrics

The same comparison code and metrics are used in both cases.

---

## 6. Metrics

The following metrics are reported:

### Cosine Similarity
- Measures directional alignment in high-dimensional parameter space
- Sensitive to distributed distortion
- Robust to uniform rescaling

### Relative L2 Error
- Measures magnitude distortion
- Highly sensitive to extreme-value explosions
- Particularly revealing for FP16 exponent corruption

### Valid Tensor Count
- Tracks whether tensors remain numerically meaningful
- Captures catastrophic vs graceful failure modes
- NaN / Inf tensors are rejected

---

## 7. Failure Modes

Two distinct failure modes are observed:

### Numerical Degradation
- Gradual distortion of values
- Smooth change in cosine and L2 metrics
- Characteristic of DTDR

### Container Failure
- Serialization-level errors (e.g. `_pickle.UnpicklingError`)
- Caused by opcode corruption, not numerical instability
- Observed only after DTDR has already demonstrated smooth degradation

These modes are intentionally distinguished.

---

## 8. Results Summary

Quantitative results are provided in:

