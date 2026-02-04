\# Experiment 04 — Storage Accounting Results



\## Model



\- Model: Mistral-7B

\- Baseline representation: FP16

\- DTDR representation: structured orthogonal transform + INT8 quantisation

\- Comparator INT8 format: GGUF-style conventional INT8 representation



---



\## Persistent Storage Footprint (Uncompressed)



| Representation | Approximate Size |

|---------------|------------------|

| FP16 | ~14.5 GB |

| INT8 GGUF | ~X.X GB |

| \*\*INT8 DTDR\*\* | \*\*~6.7–6.8 GB\*\* |



This corresponds to a \*\*~50–55% reduction\*\* in persistent storage relative to FP16.



---



\## Secondary Lossless Compression (ZIP)



To assess residual compressibility, the above representations were subjected to

standard ZIP compression.



| Representation | ZIP Reduction | Notes |

|---------------|---------------|-------|

| FP16 | ~0–1% | Near entropy-saturated |

| INT8 GGUF | ~3–4% | Often within noise |

| \*\*INT8 DTDR\*\* | \*\*~30–35%\*\* | Consistent across runs |



In round numbers:



\- \*\*INT8 DTDR → ZIP → ~65–70% of original DTDR size\*\*

\- INT8 GGUF → ZIP → ~96–97% of original size



---



\## Interpretation



The substantial residual compressibility of DTDR representations indicates retained

\*\*structural regularity in the transform domain\*\*.



Conventional INT8 formats are designed to be close to entropy-saturated and therefore

exhibit little further compressibility under lossless coding. DTDR, by contrast,

preserves global structure that can be exploited by generic lossless compressors.



This behaviour demonstrates that:



\- DTDR is \*\*not a terminal compression format\*\*

\- DTDR is a \*\*structured computational representation\*\*

\- Secondary lossless compression is optional and independent of DTDR itself



---



\## Reconstruction Notes



\- DTDR representations are reconstructed to working numerical precision prior to

&nbsp; inference or computation.

\- Reconstruction is performed once at model initialisation.

\- No specialised execution kernels or custom inference frameworks are required

&nbsp; following reconstruction.



---



\## Summary



INT8-DTDR representations achieve substantial persistent storage reduction relative

to FP16, while retaining significant residual structure that allows \*\*an additional

~30–35% lossless compression under ZIP\*\*.



This distinguishes DTDR from conventional quantised formats and supports its

characterisation as a \*\*computational transform-domain representation\*\*, rather than

a conventional compression scheme.



