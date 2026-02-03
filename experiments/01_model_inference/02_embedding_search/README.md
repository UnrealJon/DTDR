# Experiment 02 — Similarity Search in the DTDR Domain

## Purpose

This experiment demonstrates that vector embeddings stored in a
**Distributed Transform-Domain Representation (DTDR)** preserve
computationally relevant geometry, enabling similarity search to be
performed **directly in the transformed and quantised domain**.

The objective is to establish that DTDR constitutes a functional
representation suitable for retrieval and RAG-style workflows, rather
than a storage-only compression format.

---

## Experimental Artefact

- Notebook: `Book_demo_RAG.ipynb`

The notebook contains a worked demonstration of similarity-based
retrieval using embeddings that have been transformed and quantised
into a DTDR representation.

---

## Method Overview

1. Textual data are embedded into floating-point vector representations.
2. The embeddings are transformed into a DTDR form using a structured
   orthogonal transform and quantisation.
3. Query embeddings are transformed using the same DTDR procedure.
4. Similarity comparisons are performed directly in the DTDR domain
   using cosine similarity.
5. Retrieval quality is compared against floating-point baselines.

At no point is the DTDR representation reconstructed to full
floating-point precision prior to similarity evaluation.

---

## Observations

The DTDR-domain similarity search exhibits:

- Preservation of relative similarity ordering between vectors.
- Retrieval results comparable to those obtained using floating-point
  embeddings.
- Stable behaviour under transform-domain representation and
  quantisation.

These observations indicate that DTDR preserves the geometric structure
required for similarity-based retrieval.

---

## Scope and Limitations

- This experiment focuses on correctness of similarity relationships,
  not large-scale ANN indexing performance.
- Query latency and index construction time are not optimised.
- Results are intended to demonstrate representational validity rather
  than production deployment characteristics.

---

## Conclusion

This experiment demonstrates that DTDR preserves similarity structure
sufficiently to support direct similarity search within the transformed
and quantised domain. This property distinguishes DTDR from conventional
compression schemes, which typically require reconstruction prior to
meaningful similarity evaluation.
