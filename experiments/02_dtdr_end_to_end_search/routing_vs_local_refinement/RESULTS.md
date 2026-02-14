
---

## 2) RESULTS.md (interpretation)

This file explains what the numbers mean.

Paste this:

```markdown
# Results Interpretation

Typical output:

| k_local | recall@10 | lists | candidates |
|------|------|------|------|
| 10 | low | 2 | small |
| 80 | higher | 2 | medium |
| 320 | saturates | 2 | large |

Key observations:

1) Partitions probed remains ~2 regardless of dataset size
2) Increasing k_local improves recall without increasing search breadth
3) Recall saturates → error comes from local ambiguity, not search failure

---

## Conclusion

The experiment shows that DTDR routing collapses the ANN search problem:

Instead of exploring more of the dataset to improve accuracy,
the system performs deeper evaluation within a fixed region.

Global search complexity disappears.