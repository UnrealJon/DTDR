# DTDR Start Demo

Minimal, Stand-Alone Demonstration of Transform-Domain Behaviour

This folder contains a fully self-contained, CPU-only demonstration of
the behavioural properties of Distributed Transform-Domain
Representation (DTDR).

It is designed as the first point of contact for understanding DTDR.

No large models, GPUs, or external datasets are required.

Requirements:
numpy
scipy
matplotlib
scikit-learn
requests
------------------------------------------------------------------------

## What This Demo Does

This notebook demonstrates, in a controlled retrieval setting, that:

1.  Orthogonal transforms preserve similarity geometry.
2.  Similarity search can be performed directly in the transform domain.
3.  Quantised transform-domain representations remain compute-capable.
4.  Composite transforms distribute information more evenly across
    coefficients.
5.  Under simulated coefficient loss, DTDR representations degrade
    gradually.
6.  A no-transform INT8 baseline degrades more abruptly.

This illustrates the central DTDR claim:

> A distributed transform-domain representation preserves computational
> structure and exhibits graceful degradation under partial corruption.

------------------------------------------------------------------------

## Scientific Question Being Tested

If numerical information is distributed across transform coefficients,
then:

-   Partial coefficient loss should reduce performance smoothly rather
    than catastrophically.
-   Similarity rankings should remain largely preserved.
-   Composite transforms should improve robustness.

This notebook tests those hypotheses directly.

------------------------------------------------------------------------

## How It Works

1.  Public-domain texts (*Alice in Wonderland*, *Sherlock Holmes*) are
    downloaded automatically.

2.  Text is chunked into passages.

3.  TF-IDF vectors are computed.

4.  Four representations are constructed:

    -   Original floating-point space\
    -   INT8 (no transform baseline)\
    -   Hadamard transform + INT8\
    -   Hadamard + DCT composite transform + INT8

5.  Similarity search is performed directly in each representation.

6.  Controlled coefficient dropout simulates corruption.

7.  Retrieval accuracy and degradation curves are plotted.

------------------------------------------------------------------------

## Expected Runtime

1--3 minutes on a typical laptop CPU.

No GPU required.

------------------------------------------------------------------------

## Running the Demo

### Option A --- Clean Python Environment

``` bash
pip install -r requirements.txt
jupyter notebook
```

Open:

    DTDR_RAG_double_transform_demo.ipynb

Run all cells.

------------------------------------------------------------------------

### Option B --- Google Colab

Upload the notebook directly to Colab and run. All required libraries
will install automatically.

------------------------------------------------------------------------

## What This Demo Is Not

This is not:

-   A large language model compression demo.
-   An ANN benchmark.
-   A production retrieval system.

It is a minimal, controlled mathematical demonstration of DTDR
behaviour.

------------------------------------------------------------------------

## Why This Matters

Conventional quantisation and compression aim to minimise error in local
coefficients.

DTDR instead distributes signal energy across coefficients such that:

-   Computation remains possible in the stored domain.
-   Loss of subsets of coefficients does not immediately destroy
    functionality.
-   Representations remain structurally meaningful under corruption.

This property underlies:

-   Robust storage
-   Compute-capable compressed models
-   DTDR-domain similarity search
-   Multi-resolution retrieval signals
