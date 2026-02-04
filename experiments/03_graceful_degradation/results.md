# Experiment 03 — Graceful Degradation Results

## Corruption Test

A controlled perturbation was applied to a fraction of DTDR coefficients
prior to reconstruction.

### Parameters
- Corruption fraction: 0.001 (0.1%)

## Output

Device: cpu
Shapes: torch.Size([2, 64, 256]) torch.Size([2, 64, 256]) torch.Size([2, 64, 256])
Cosine similarity (DTDR vs FP): 0.9999980926513672
Relative L2 error (DTDR vs FP): 0.002108924090862274
After toy corruption in DTDR weights (frac=0.0010):
Cosine similarity (corrupt vs FP): 0.999997615814209
Relative L2 error (corrupt vs FP): 0.0023368862457573414



## Notes
- DTDR reconstruction yields a numerically different but functionally equivalent representation.
- Partial corruption of DTDR coefficients produces only marginal degradation, consistent with a distributed representation.
