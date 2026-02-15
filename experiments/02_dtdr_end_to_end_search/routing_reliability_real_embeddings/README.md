# DTDR Routing Reliability on Real Embeddings

This experiment evaluates DTDR as a *routing layer* rather than a full ANN replacement.

Instead of asking whether DTDR alone can solve nearest neighbour search, we ask a different systems question:

> Can DTDR reliably localise the correct answer while keeping the number of searched partitions constant?

The test uses real semantic embeddings (GloVe) rather than synthetic clustered data.

---

## Hypothesis

Conventional ANN systems improve recall by expanding the search breadth:

```
higher recall → probe more partitions/shards
```

DTDR routing proposes the opposite trade-off:

```
higher recall → more local refinement
fan-out remains constant
```

If true, nearest-neighbour search changes from a **distributed search problem** into a **local ranking problem**.

---

## What the script does

`nprobe4_glove_routing.py`:

1. Loads 200k real word embeddings (GloVe)
2. Normalises and converts them into DTDR domain
3. Builds an IVF partitioning
4. Routes queries using dilution evidence
5. Always probes only **2 partitions**
6. Increases only the local candidate budget (`k_local`)
7. Measures:

* recall@10
* hit@1 (true nearest neighbour survives routing)

---

## Why hit@1 matters

Production retrieval pipelines rarely require exact ordering from ANN.
They require that the correct candidate survives to later stages:

```
ANN → reranker → LLM
```

Therefore the key routing metric is:

> Probability the correct item is not discarded early.

---

## Running

Place GloVe vectors at:

```
data/glove/glove.6B.50d.txt
```

Then run:

```
python nprobe4_glove_routing.py
```

---

## Expected Behaviour

As `k_local` increases:

• hit@1 approaches 1.0
• recall improves
• number of partitions searched remains constant (2)

This indicates localisation rather than global search.
