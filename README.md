Distributed Transform-Domain Representation (DTDR)

A persistent computational memory representation for machine learning systems

DTDR is a transform-domain representation of numerical models and embeddings in which information is stored globally rather than locally.
Instead of encoding parameters as individually meaningful weights, DTDR stores models as a distributed constraint system whose behaviour emerges collectively.

This repository contains a set of experiments demonstrating that DTDR is not a compression trick or quantisation format — but a functional representation with distinct computational properties.

The Core Idea

Most numerical formats store models as collections of independent parameters:

accuracy depends on preserving individual values

DTDR stores models as a system of global constraints:

accuracy depends on preserving consistency of the whole

Because of this, errors behave differently.

Representation	What quantisation preserves	What error destroys
Conventional	parameter precision	function
DTDR	functional consistency	fine detail

In practical terms:

DTDR does not minimise numerical distortion — it redirects distortion into directions that least affect behaviour.

This produces a distinctive combination of effects observed throughout the experiments:

improved functional stability at low precision

graceful degradation under corruption

residual lossless compressibility

localisation without reconstruction

emergence only after global consistency is reached

The representation therefore organises information around the model’s behaviour rather than its coefficients.

What DTDR Is

DTDR represents vectors, embeddings and neural network parameters in an orthogonal transform domain (e.g. Hadamard) followed by structured quantisation.

Unlike conventional formats:

Conventional representation	DTDR
Parameters store features	Coefficients store constraints
Damage breaks parts	Damage becomes noise
Smaller model = weaker model	Partial model = no model
Search explores space	Search localises then ranks
Compression removes detail	Representation redistributes information

DTDR behaves more like an interference pattern than a parameter list.

Core Claim

DTDR is a transform-domain memory representation that preserves functional behaviour, supports computation directly in the stored domain, and exhibits graceful degradation under damage while showing emergence thresholds under incompleteness.

Repository Experiments

The experiments are organised as a progressive argument.

1. Semantic Geometry Exists in the Transform Domain

DTDR embeddings can be compared directly in quantised transform space while preserving similarity ordering.

Result:

Retrieval quality comparable to floating point

No decompression required

Meaning:

DTDR preserves semantic geometry, not just numbers.

2. Neural Networks Still Function

A Mistral-7B model stored in DTDR form reconstructs into a numerically different but functionally equivalent working model.

Meaning:

DTDR stores model behaviour rather than exact parameters.

3. Algorithms Operate Inside DTDR

Vector search pipelines operate directly on DTDR representations.

Meaning:

DTDR is a computational domain, not just a storage format.

4. Search Scaling Changes

DTDR routing keeps the number of searched partitions constant while improving recall through local refinement.

Conventional ANN:

more data → search more machines


DTDR:

more certainty → more local compute


Observed (end-to-end ANN search):

Baseline IVF-HNSW (nprobe=4):   recall@10 = 0.6285   latency ≈ 2.87 ms
With DTDR dilution evidence:    recall@10 = 0.7760   latency ≈ 3.10 ms


~15% absolute recall gain at nearly identical latency, approaching the cost of doubling probes.

Meaning:

DTDR turns search into localisation.

5. Real Embeddings Confirm the Behaviour

On real semantic embeddings:

constant shard fan-out

increasing hit rate via local refinement

Implication:

Reduced cross-node traffic and tail latency risk.

6. Robustness to Damage

Random corruption of stored DTDR parameters produces proportional output degradation instead of catastrophic failure.

Interpretation:

Information is distributed across coefficients.

7. Robustness on Disk

Compared with FP16 storage:

FP16 fails at extremely low corruption

DTDR remains numerically stable across orders of magnitude more damage

Meaning:

DTDR converts storage failure into noise.

8. Persistent Storage Behaviour
Representation	Raw Size	After ZIP	Notes
FP16 HuggingFace	~14.5 GB	~13.8–14.2 GB	Already high entropy
GGUF Q8	~7.2 GB	~7.1–7.2 GB	Essentially incompressible
DTDR-INT8	~6.6–6.9 GB	~4.0–5.0 GB	Significant residual structure
DTDR reconstructed FP16	~14.5 GB runtime	N/A	Functional reconstruction

Meaning:

DTDR is structured memory, not entropy-saturated encoding.

9. Emergence vs Degradation (Key Result)
Condition	Behaviour
Corruption	Smooth degradation
Truncation	Sudden appearance of language

Interpretation:

The model exists only when global constraints close.

Conceptual Interpretation

DTDR behaves like triangulation:

Move satellites → position drifts

Remove satellites → no position exists

A neural network stored in DTDR is a solved system, not a parameter list.

Summary

DTDR simultaneously exhibits:

semantic similarity preservation

functional reconstruction

computation in compressed space

different ANN scaling laws

graceful degradation

emergence thresholds

residual compressibility

No conventional compression or quantisation scheme shows this combination.

Status

Research prototype and experimental demonstration.

---

## Repository Structure



```text
experiments/
├── 01_model_inference/
├── 02_dtdr_end_to_end_search/
├── 03_embedding_search/
├── 04_graceful_degradation/
├── 05_storage_accounting/
├── 06_emergence_from_distribution/

DTDR_RAG_double_transform_demo.ipynb

Patent & Commercial Licensing

DTDR is the subject of a filed UK patent application:

UK Patent Application No. GB2602157.6

This repository is provided for research and evaluation purposes.

For commercial licensing, strategic partnerships, or IP inquiries:

Contact: dtdr@multiverse1.com


See LICENSE_NOTICE.md for evaluation terms.
