# DTDR Routing vs Local Refinement Experiment

This experiment demonstrates a structural property of DTDR-based similarity search:

> Retrieval cost becomes dominated by *local refinement* rather than *global search*.

In conventional ANN systems, improving recall requires searching more partitions as the dataset grows.

In the DTDR routing pipeline:

• The number of partitions probed remains approximately constant  
• Accuracy improves only by increasing local candidate evaluation inside those partitions  

This converts nearest-neighbour search from a **global exploration problem** into a **local ranking problem**.

---

## What the script does

`nprobe4_klocal_sweep.py`:

1. Generates a clustered embedding dataset
2. Applies DTDR transform
3. Routes queries using dilution evidence
4. Sweeps local candidate budget (`k_local`)
5. Measures:

- recall@10
- partitions probed
- candidates evaluated

---

## Expected behaviour

As dataset size increases:

| Property | Classical ANN | DTDR Routing |
|--------|------|------|
| Partitions probed | grows | constant |
| Compute cost | search-dominated | local-ranking dominated |
| Scaling | global | local |

---

## Run

```bash
python nprobe4_klocal_sweep.py --n 200000