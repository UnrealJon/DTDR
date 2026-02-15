# Results and Implications

## Observed behaviour

On real semantic embeddings:

| k_local | recall@10 | hit@1    |
| ------- | --------- | -------- |
| small   | low       | moderate |
| medium  | moderate  | high     |
| large   | higher    | ~1.0     |

Crucially:

> The number of partitions searched never increases.

---

## Interpretation

DTDR does not perfectly partition semantic space.

Instead it concentrates probability mass:

The correct nearest neighbour becomes increasingly likely to lie inside a constant number of routed regions.

Thus the problem separates into two independent stages:

| Stage          | Role         |
| -------------- | ------------ |
| DTDR routing   | localisation |
| ANN / reranker | ranking      |

---

## Architectural consequence

Conventional ANN scaling:

```
more data → search more machines
```

DTDR routing scaling:

```
more certainty → more local compute
machines contacted unchanged
```

This transforms nearest neighbour retrieval from a distributed search problem into a controllable local refinement problem.

---

## Commercial implication

Large-scale vector systems are dominated by:

• shard fan-out
• tail latency
• cross-node traffic

DTDR routing offers a different control knob:

Operators can trade compute for reliability **without increasing fan-out**.

This makes it suitable as a front-end routing layer for existing ANN infrastructure rather than a replacement search engine.
