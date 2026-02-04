## Primary Demonstration

The notebook `DTDR_RAG_double_transform_demo.ipynb` provides an
end-to-end demonstration of DTDR as a functional representation,
integrating:

- composite (double) DTDR constructions,
- direct similarity search in the transform domain,
- and retrieval-style workflows.

Readers looking for a single, self-contained demonstration of DTDR
behaviour may wish to begin here. The numbered experiment folders
provide focused, claim-aligned supporting evidence.



# Experiment 01 — Functional Reconstruction of a DTDR-Stored Model

## Purpose

This experiment demonstrates that a numerical model stored in a **Distributed Transform-Domain Representation (DTDR)** can be reconstructed to a working numerical precision that is *functionally equivalent* to the original floating-point representation.

The objective is to show that DTDR constitutes a **persistent representation**, not merely a compression scheme.

---

## Method Summary

1. A reference floating-point model is taken as a baseline.
2. The same model parameters are stored in a DTDR form using a structured orthogonal transform and quantisation.
3. The DTDR representation is reconstructed to a working numerical precision.
4. Functional equivalence between the baseline and reconstructed models is evaluated using similarity metrics.
5. Controlled perturbation of DTDR coefficients is applied to test robustness.

---

## Observations

- The reconstructed DTDR model exhibits cosine similarity extremely close to unity when compared to the floating-point baseline.
- Relative L2 error remains small and stable.
- Partial corruption of DTDR coefficients results in only marginal degradation of similarity metrics.

These results are consistent with a representation in which information is distributed across coefficients rather than localised.

---

## Scope and Limitations

- This experiment evaluates *functional equivalence*, not inference throughput.
- Execution was performed on CPU for clarity and reproducibility.
- Performance benchmarking and GPU execution are addressed in separate experiments.

---

## Conclusion

The results demonstrate that DTDR enables persistent storage of numerical model parameters while permitting reconstruction to a numerically different but functionally equivalent representation suitable for inference and further computation.
