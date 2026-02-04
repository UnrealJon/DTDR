### \# DTDR — Distributed Transform-Domain Representation



DTDR is a persistent transform-domain representation for semantically encoded high-dimensional numerical data that enables \*\*substantial size reduction\*\* while preserving computational geometry.



Unlike conventional compression schemes, DTDR representations remain \*\*computationally meaningful at rest\*\*, supporting direct similarity search, inference, and reconstruction to a working numerical precision, with graceful degradation under quantisation or partial data loss.



---



\## Key Results (Summary)



| Scenario              | Original Size | DTDR Size | Reduction | Fidelity / Functionality |

|-----------------------|---------------|-----------|-----------|---------------------------|

| Mistral-7B (FP16)     | 14.5 GB       | 6.7 GB    | 2.2×      | ~0.9998 cosine similarity |

| SIFT1M vectors        | 488 MB        | 122 MB    | 4.0×      | ~94% recall@1             |

| 384-D embeddings      | 24.4 MB       | 8.1 MB    | 3.0×      | ~0.9997 cosine similarity |



Additional lossless compression applied to DTDR representations has been shown to yield a further ~30–35% reduction in storage footprint.



Full experimental details, methodology, and validation are provided in the accompanying documentation.

**Experimental provenance:**

- **Mistral-7B (FP16 → DTDR)** results are demonstrated in  
  `experiments/01_model_inference` and `experiments/04_storage_accounting`.

- **384-D embeddings** similarity preservation is demonstrated in  
  `experiments/02_embedding_search`.

- **Graceful degradation behaviour** is demonstrated in  
  `experiments/03_graceful_degradation`.

- **Storage footprint reductions** are summarised in  
  `experiments/04_storage_accounting`.


---



\## Documentation



\- [\[DTDR Overview (PDF)](docs/DTDR\_Overview.pdf)](https://github.com/UnrealJon/DTDR/blob/main/docs/DTDR_Overview.pdf)  

\- Experimental Evidence (https://github.com/UnrealJon/DTDR/tree/main/experiments)  

\- Why DTDR Is Not Compression (https://github.com/UnrealJon/DTDR/blob/main/docs/Why_DTDR_is_not_compression.pdf)



---



\## Patent Status



DTDR is the subject of a filed UK patent application:



\*\*UK Patent Application No. GB 2602157.6\*\*



Patent rights are independent of the accompanying code and documentation.

Commercial deployment requires a separate commercial and patent licence.



---



\## Commercial Licensing



The author is open to discussions with commercial or industrial partners regarding licensing, assignment, or acquisition of the associated intellectual property, particularly in contexts involving:



\- large-scale model storage  

\- vector databases and similarity search  

\- edge deployment  

\- memory- or bandwidth-constrained systems  



\*\*Contact:\*\* dtdr@multiverse1.com  

\*(please reference “DTDR” in correspondence)\*



---



\## License



This repository is licensed under the   \*\*Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)\*\* license.



Research, technical evaluation, internal testing, and non-commercial experimentation — including modification for evaluation purposes — are permitted.



Commercial use requires separate licensing.



See \[LICENSE](LICENSE) and \[LICENSE\_NOTICE.md](LICENSE\_NOTICE.md) for details.







