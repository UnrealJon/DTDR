# DLMH: Distributed-Like Memory Hypothesis

**A unified framework for understanding hallucination in AI and confabulation in human memory**

---

## The Core Idea

When a person misremembers something, the error is rarely random. They recall something *plausible* — something semantically related, something that fits the context. The memory system doesn't produce noise; it produces a confident, coherent, *wrong* reconstruction. This is confabulation.

DLMH proposes that AI language models make errors for the same reason, through the same underlying mechanism.

> Memory — whether biological or artificial — is not retrieval of stored facts. It is **reconstruction along a constrained trajectory**. When the constraints are insufficient, the system converges on the nearest plausible answer — which may be stably, confidently, and repeatedly wrong.

This repository provides two experiments that demonstrate this mechanism directly: one mathematical (showing *why* it happens) and one behavioural (showing *that* it happens in a real AI model).

---

## Why This Matters for Psychology

The standard view of memory errors treats them as retrieval failures — the right memory was stored but couldn't be accessed. DLMH offers a different account: the error is not a failure of access but a *success* of reconstruction under incomplete information. The system did exactly what it was designed to do; it simply didn't have enough constraints to arrive at the correct answer.

This reframing has immediate consequences:

- **Confabulation** is not pathological. It is the normal operation of a reconstruction system under insufficient constraint.
- **Tip-of-the-tongue states** reflect partially completed trajectories, not blocked retrieval.
- **False memories** are geometrically predictable — they cluster in semantic space near the target, not randomly distributed.
- **Graceful degradation** under neural damage is expected, because information is distributed across the system rather than stored in discrete locations.

These phenomena sit awkwardly in state-based retrieval models. In a trajectory-based model, they are natural consequences of the architecture.

---

## A Useful Analogy

Think of the difference between a **chord** and a **melody**.

A chord exists at a single moment — it can be stored, retrieved, and recognised instantaneously. A melody only exists in time. Its identity is the trajectory, the unfolding sequence. You cannot freeze a melody mid-note without destroying what makes it that melody.

Classical memory models treat recall like recognising a chord. DLMH proposes it is more like playing a melody — an active, temporally extended process of reconstruction guided by constraints, not a lookup.

---

## Key Concepts

### Attractor
A stable resting point in a dynamical system. When a system is perturbed, it tends to settle into the nearest attractor. In memory terms: given a partial or noisy cue, the system converges to the nearest stored pattern. The problem arises when the nearest attractor is *wrong*.

### Constraint Density
The richness of information in a cue or context. High constraint density guides reconstruction toward the correct answer. Low constraint density allows the system to drift toward a strong but incorrect attractor.

### Constraint Closure
The condition in which all alternative interpretations have been explicitly ruled out, leaving only one viable trajectory. This is what is required — not repetition or emphasis — to guarantee correct reconstruction.

### Behavioural Regimes

A system can be characterised by two independent properties: how *consistently* it produces the same answer (stability), and how *correctly* it does so (accuracy). These yield three regimes:

| Regime | Stability | Accuracy |
|---|---|---|
| Stable Correct | High | High |
| Unstable | Low | Variable |
| **Stable Incorrect** | **High** | **Low** |

The third regime — **stable incorrect convergence** — is the core phenomenon DLMH seeks to explain. A system in this regime is not producing random noise. It is confidently, consistently, and systematically wrong. This is the computational signature of confabulation.

---

## The Experiments

This repository contains two complementary investigations of the same phenomenon.

### Experiment 1: The Mathematical Foundation
**File:** `walsh_dlmh_demo_upgraded.ipynb`

This notebook demonstrates, using a minimal mathematical model, that stable incorrect outputs arise *naturally* in any distributed memory system — they are a consequence of geometry, not a bug.

The model is a simplified version of a Hopfield network (a classical mathematical model of associative memory familiar to cognitive neuroscientists). When stored memories are retrieved from a partial or noisy cue, the system settles into the nearest stable configuration. If the constraints provided by the cue are insufficient, it may settle into a neighbouring configuration — consistently, stably, and incorrectly.

The notebook then introduces a key diagnostic distinction: **stability and accuracy are independent**. A system can be highly self-consistent and completely wrong. This is demonstrated by showing that the same starting conditions, under different constraint structures, produce either stable-correct or stable-incorrect convergence — with the stable-incorrect system showing *higher* apparent confidence.

Readers without a mathematical background can focus on the figures and the interpretive commentary in each section. The technical details of the Walsh-Hadamard transform (a mathematical tool used to analyse the structure of the memory landscape) are secondary to the conceptual result.

### Experiment 2: The Behavioural Evidence
**File:** `y_attractor_prompt__1_.ipynb`

This notebook provides direct empirical evidence of stable incorrect convergence in a real AI language model (Mistral-7B, a publicly available model run locally). The experimental design is systematic and the results are striking.

The setup is deceptively simple. The model is told that a variable Y has a specific numerical value, then asked to compute with it. Mistral has a strong internal prior that Y equals 2 — presumably from training data — and the experiment tests how resistant this prior is to various forms of correction.

**The progression of results tells a clear story:**

**Phase 1 — Prompting fails.** Telling the model Y equals a different value (directly, repeatedly, or embedded in a rich context) achieves approximately 10% accuracy. The model consistently reverts to Y=2. This is stable incorrect convergence: high consistency, wrong answer.

**Phase 2 — More sophisticated prompting makes things worse.** Variable substitution (assigning the value to a new variable Z), constraint equations, and two-stage transfer procedures all collapse to 0% accuracy. The attractor is not merely a surface habit — it runs deep.

**Phase 3 — Semantic framing partially works.** When the prompt explicitly declares the *ontological status* of the variable ("Y is an algebraic variable number. The numerical value of Y is 5."), accuracy rises to 75–92%. Framing the variable correctly in semantic space shifts the inference trajectory.

**Phase 4 — Constraint closure works.** When the prompt explicitly states that Y has no other value than the one given — ruling out all competing interpretations — accuracy reaches 100%. This is the operational definition of constraint closure: not emphasis, not repetition, but the explicit exclusion of alternatives.

**Phase 5 — Closure does not persist across steps.** When correct values must be carried through a chain of reasoning steps, accuracy drops to 44–56%. Constraint closure is local. The system must be re-constrained at each step; the correct trajectory does not propagate automatically.

---

## Summary of Findings

1. **Errors are structured, not random.** Incorrect outputs are stable, repeatable, and resistant to prompt variation. This is the signature of attractor dynamics, not noise.

2. **Repetition and emphasis do not help.** Telling a system the correct answer more forcefully, more often, or in richer context does not escape a strong attractor basin.

3. **Constraint closure is what works.** Correct outputs emerge reliably only when competing interpretations are explicitly excluded — when the space of possible trajectories is narrowed to one.

4. **Closure is local.** Even when a single reasoning step is constrained correctly, the constraint does not carry forward. Multi-step reasoning degrades because each step is a fresh trajectory, not a continuation of the previous one.

---

## Connection to Human Memory

The parallel between LLM behaviour and human confabulation is not merely analogical. DLMH proposes they reflect the same computational principle: reconstruction under constraint, where insufficient constraint density allows convergence to a plausible-but-wrong attractor.

The specific phenomena this explains include:

- **Confabulation in amnesia** — patients fill gaps with semantically plausible content, not random noise, because reconstruction is constrained by semantic structure even when episodic detail is unavailable.
- **The misinformation effect** — post-event information alters the constraint landscape, biasing reconstruction toward the updated attractor.
- **Reconsolidation** — retrieving a memory does not replay a stored state; it re-runs a trajectory through a constraint landscape that may itself have changed.
- **Source monitoring errors** — two memories with overlapping constraint structures may reconstruct toward each other, producing confident misattribution.

The full theoretical treatment is in the accompanying paper: `DLMH.pdf`.

---

## Repository Contents

| File | Description |
|---|---|
| `DLMH.pdf` | Full theoretical paper with formal framework and predictions |
| `walsh_dlmh_demo_upgraded.ipynb` | Mathematical model: attractor dynamics and stable incorrect convergence |
| `y_attractor_prompt__1_.ipynb` | Empirical experiment: Y-attractor in Mistral-7B |

---

## Running the Experiments

### Requirements
- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- NumPy, Matplotlib

### LLM Experiment

Requires a local model snapshot (no downloads are performed at runtime). Set the path in Cell 1:

```python
MODEL_PATH = r"your_local_mistral_snapshot"
```

The experiment is designed for Mistral-7B-Instruct (v0.2 or v0.3). Results may differ with other models, which is itself an empirically interesting question.

---

## Citation

West, J. (2025). *Memory as Reconstruction: A Distributed Transform-Domain Framework for Understanding Recall Dynamics* [Preprint]. DLMH repository, GitHub. https://github.com/UnrealJon/DTDR
