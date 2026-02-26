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
Confirm raw sizes:

dir "G:\AI_demo\zip_demo_mistral\int8_baseline\mistral-7b-instruct-v0.2.Q8_0.gguf",
    "G:\AI_demo\zip_demo_mistral\dtdr_int8_clean.pkl" |
Select-Object FullName, Length, @{Name="SizeGB";Expression={[math]::Round($_.Length/1GB,3)}}

2) Create ZIP64 archives using Python

Create a script file zip_large.py in the same directory (G:\AI_demo\zip_demo_mistral):

import zipfile
import os

def zip_large(input_path, output_path):
    print(f"Zipping {input_path} ...")
    with zipfile.ZipFile(output_path, "w",
                         compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        zf.write(input_path, arcname=os.path.basename(input_path))
    print(f"Created: {output_path}")

zip_large(r".\int8_baseline\mistral-7b-instruct-v0.2.Q8_0.gguf",
          r".\rerun_GGUF_Q8_0.zip")

zip_large(r".\dtdr_int8_clean.pkl",
          r".\rerun_DTDR_INT8.zip")

Run:

cd "G:\AI_demo\zip_demo_mistral"
python zip_large.py
3) Measure compressed sizes
dir ".\int8_baseline\mistral-7b-instruct-v0.2.Q8_0.gguf",
    ".\dtdr_int8_clean.pkl",
    ".\rerun_GGUF_Q8_0.zip",
    ".\rerun_DTDR_INT8.zip" |
Select-Object Name, Length, @{Name="SizeGB";Expression={[math]::Round($_.Length/1GB,3)}}
4) Compute compression ratios (optional)

Compression ratio:

ratio = zipped_bytes / raw_bytes

Additional lossless reduction:

reduction_pct = (1 - ratio) * 100

You can compute this in PowerShell:

$gguf = Get-Item ".\int8_baseline\mistral-7b-instruct-v0.2.Q8_0.gguf"
$ggufzip = Get-Item ".\rerun_GGUF_Q8_0.zip"
$dtdr = Get-Item ".\dtdr_int8_clean.pkl"
$dtdrzip = Get-Item ".\rerun_DTDR_INT8.zip"

@(
  [PSCustomObject]@{
    Artifact="GGUF Q8_0"
    RawGB=[math]::Round($gguf.Length/1GB,3)
    ZipGB=[math]::Round($ggufzip.Length/1GB,3)
    ZipRatio=[math]::Round($ggufzip.Length/$gguf.Length,3)
    ZipReductionPct=[math]::Round(100*(1-$ggufzip.Length/$gguf.Length),1)
  },
  [PSCustomObject]@{
    Artifact="DTDR INT8"
    RawGB=[math]::Round($dtdr.Length/1GB,3)
    ZipGB=[math]::Round($dtdrzip.Length/1GB,3)
    ZipRatio=[math]::Round($dtdrzip.Length/$dtdr.Length,3)
    ZipReductionPct=[math]::Round(100*(1-$dtdrzip.Length/$dtdr.Length),1)
  }
) | Format-Table -Auto
Notes and Controls

This experiment compresses one file vs one file to avoid folder metadata effects.

Both archives use the same algorithm (DEFLATE) and ZIP64 support.

If you want a stronger compressor (e.g. LZMA / 7z), install 7-Zip and repeat using 7z a -mx=9 -t7z ....

Results may vary slightly with compressor and version, but the gap between GGUF and DTDR is the key outcome.

Interpretation

The baseline GGUF Q8_0 file shows only marginal residual lossless compressibility (~3–5% in this run), consistent with a representation that is closer to entropy saturation at the quantisation stage.

The DTDR INT8 artefact shows substantial additional lossless compressibility (~25–35% in this run), consistent with DTDR producing a non-terminal transform-domain representation in which quantisation noise and information-bearing structure are distributed across coefficients, leaving residual statistical structure exploitable by general-purpose compressors.

Results (Byte-level details from this run)

GGUF Q8_0 raw: 7,695,857,952 bytes (7.167 GB)

DTDR INT8 raw: 7,248,464,396 bytes (6.751 GB)

GGUF Q8_0 zipped: 7,411,219,447 bytes (6.902 GB)

DTDR INT8 zipped: 5,180,785,451 bytes (4.825 GB)


