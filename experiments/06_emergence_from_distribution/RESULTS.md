
---

# 📄 `RESULTS.md`

```markdown
# Experimental Results — Partial Completeness Behaviour

This document summarises empirical results for DTDR under coefficient incompleteness.

Completeness ratio:

\[
\rho = \frac{\text{available coefficients}}{\text{total coefficients}}
\]

---

# 1. Geometric Regime — Embedding Similarity

## Datasets

- SIFT1M
- GloVe-100d

## Dropout Modes

- Random coefficient dropout
- Block (contiguous shard) dropout

---

## Cosine Similarity vs ρ

Across both datasets:

- Cosine similarity scales approximately linearly.
- Random and block dropout behave almost identically.
- No sharp collapse threshold observed.

![Cosine Similarity](urho_combined_cosine.png)

Key observation:

\[
\text{Cosine similarity} \approx \rho
\]

---

## Recall@10 vs ρ (GloVe-100d)

- Recall closely tracks a linear reference line.
- Retrieval quality degrades proportionally.
- Even at moderate ρ (e.g., 0.5), retrieval remains partially meaningful.

![Recall@10](urho_combined_recall.png)

This confirms:

\[
U(\rho) \approx \rho
\]

for similarity tasks.

---

# 2. Functional Regime — Model Weight Reconstruction

Using progressive reconstruction of DTDR-compressed model weights:

Observed behaviour:

- Below threshold → incoherent generative output
- Near threshold → unstable behaviour
- Above threshold → coherent inference emerges

This behaviour is nonlinear and not proportional.

---

# 3. Unified Interpretation

DTDR induces two distinct regimes:

### Geometric Regime
- Linear projection
- Utility scales proportionally
- No collapse threshold
- Governs vector databases

### Functional Regime
- Nonlinear multi-layer constraint system
- Emergence-like threshold
- Governs generative inference

These regimes arise from different mathematical structures and should not be conflated.

---

# 4. Practical Implications

For vector databases:

- Partial exfiltration yields proportional semantic loss.
- Completeness gating provides a quantitative confidentiality gradient.
- No need for homomorphic similarity computation.

For model weights:

- Partial exposure may not immediately enable coherent inference.
- Nonlinear constraint dynamics produce instability below threshold.

---

# 5. Conclusion

Empirical results demonstrate:

- DTDR similarity preservation degrades linearly with completeness.
- Nonlinear inference systems exhibit emergence behaviour.

This dual-regime interpretation resolves apparent contradictions between earlier experiments and provides a coherent foundation for transform-domain semantic storage.