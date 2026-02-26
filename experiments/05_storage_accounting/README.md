# Experiment: Residual Lossless Compressibility of DTDR-INT8 vs GGUF Q8_0 (Mistral-7B)

## Purpose

This experiment measures **residual lossless compressibility** after INT8 quantisation, comparing:

- a conventional INT8 baseline representation (**GGUF Q8_0**), and
- a **DTDR-INT8** (Hadamard-transform + INT8 quantised) persistent transform-domain artefact.

The hypothesis is that DTDR produces a **non-terminal** representation at the quantisation stage, retaining enough statistical structure to permit **substantially greater additional lossless compression** (e.g. ZIP/DEFLATE) than the baseline.

This is relevant to:
- model distribution and storage,
- transport bandwidth reduction,
- and the claim that DTDR does not saturate entropy at the quantisation stage.

---

## Summary of Result (this run)

| Representation | Raw Size (GB) | Zipped Size (GB) | Additional Lossless Reduction |
|---|---:|---:|---:|
| GGUF Q8_0 | 7.167 | 6.902 | 3.7% |
| DTDR INT8 | 6.751 | 4.825 | 28.5% |

Raw sizes and compressed sizes were measured in bytes on disk. ZIP was produced using **Python `zipfile` with ZIP64 enabled** and **DEFLATE** compression.

---

## Artefacts Used

### Baseline (GGUF)
- `mistral-7b-instruct-v0.2.Q8_0.gguf`
- Example path used in this run:
  - `G:\AI_demo\zip_demo_mistral\int8_baseline\mistral-7b-instruct-v0.2.Q8_0.gguf`

### DTDR (Hadamard + INT8) artefact
- `compressed_mistral_7b.pkl` (DTDR persistent transform-domain, INT8)
- Example path used in this run:
  - `G:\train_jw\compressed_mistral_7b.pkl`
- Copied into experiment folder as:
  - `G:\AI_demo\zip_demo_mistral\dtdr_int8_clean.pkl`

> Note: The DTDR artefact is the **stored transform-domain INT8 representation** (not reconstructed FP16 safetensors).  
> Reconstructed FP16 models are execution-time artefacts and are not used in this compressibility test.

---

## Environment

- OS: Windows (PowerShell)
- Python: invoked via `python` on PATH (virtualenv optional)
- Compression method: Python standard library `zipfile` with ZIP64 + DEFLATE

Why Python instead of PowerShell `Compress-Archive`?
- On this system, PowerShell `Compress-Archive` fails on multi-GB files with:
  - `Stream was too long.`
- Python `zipfile` successfully creates ZIP64 archives for large files.

---

## Reproducible Procedure

### 1) Prepare experiment working directory

Example working directory used:
- `G:\AI_demo\zip_demo_mistral`

Copy DTDR artefact into the folder so both files are local and comparable:

```powershell
Copy-Item "G:\train_jw\compressed_mistral_7b.pkl" "G:\AI_demo\zip_demo_mistral\dtdr_int8_clean.pkl"