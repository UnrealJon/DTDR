\# DLMH: Distributed-Like Memory Hypothesis



\*\*A unified framework for understanding hallucination and confabulation in large language models\*\*



---



\## Overview



The Distributed-Like Memory Hypothesis (DLMH) proposes that:



> LLM outputs are not retrieved facts, but trajectory completions through a high-dimensional probabilistic space.



Under this view, hallucinations and human confabulations arise from the same mechanism:



\- \*\*Completion under insufficient constraint density\*\*



This repository contains two complementary experiments:



1\. \*\*Walsh-space experiment\*\* (neural / geometric model)

2\. \*\*Y-attractor experiment\*\* (LLM behavioural model)



Together, they demonstrate:



\- Stable incorrect outputs are \*structured attractors\*, not random errors

\- Correctness depends on \*constraint closure\*, not repetition or emphasis

\- Multi-step reasoning degrades due to loss of constraint persistence



---



\## Key Concepts



\### Stability (Invariance)

How consistently the model produces the same answer across equivalent prompts.



\### Accuracy

Whether the answer matches ground truth.



\### Behavioural Regimes



| Regime            | Stability | Accuracy |

|------------------|----------|----------|

| Stable Correct    | High     | High     |

| Unstable          | Low      | Variable |

| Stable Incorrect  | High     | Low      |



The third regime — \*\*stable incorrect convergence\*\* — is the core phenomenon.



---



## Experiment 1: Walsh-Space / Neural Energy Model

**File:** `walsh_dlmh_demo_upgraded.ipynb`

### Goal

To demonstrate that attractor behaviour arises naturally in distributed representations
when viewed as an energy minimisation process.

---

### Conceptual framing

This experiment connects DLMH to classical neural models, particularly:

- **Hopfield networks**
- **Energy-based models**

In these systems:

> The network evolves toward low-energy states (attractors)

---

### Method

We construct a simplified analogue of a neural system:

1. Represent signals in a **Walsh (Hadamard) basis**
   - Orthogonal, distributed representation
   - Analogous to population coding in neural systems

2. Apply:
   - truncation (information loss)
   - perturbation (noise / corruption)

3. Reconstruct the signal from degraded coefficients

---

### Interpretation as an energy system

The reconstruction process can be understood as:

- Searching for a configuration consistent with the remaining signal
- Settling into a **locally stable state**

This is directly analogous to:

- Hopfield networks converging to stored patterns
- Energy landscapes with multiple minima

---

### Key result

Even when:

- information is incomplete
- coefficients are corrupted

the system:

- converges to **stable outputs**
- often produces **consistent but incorrect reconstructions**

---

### DLMH interpretation

This demonstrates that:

> Stable incorrect outputs are a natural consequence of energy minimisation
> in distributed systems with incomplete constraints.

In other words:

- The system is not "failing"
- It is **successfully converging to the nearest attractor**

---

### Why Walsh space?

The Walsh transform provides:

- a structured orthogonal basis
- fast computation (no dense matrices)
- a direct analogue to distributed neural encoding

It allows us to:

- study attractor dynamics **without training a network**
- isolate the geometry of the representation itself

---

### Connection to LLMs

This experiment provides a **mechanistic analogy**:

| Neural model            | LLM behaviour              |
|------------------------|----------------------------|
| Energy landscape       | Probability landscape      |
| Attractor state        | Generated output           |
| Partial input          | Prompt constraints         |
| Energy minimisation    | Next-token optimisation    |

---

### Key insight

> LLM hallucinations are not arbitrary errors,
> but convergence to low-energy (high-probability) attractor states
> under insufficient constraint.

This aligns directly with the behavioural results in the Y-attractor experiment.

---



\## Experiment 2: Y-Attractor (LLM Behaviour)



\*\*File:\*\* `llm\_dlmh\_upgraded.ipynb`



\### Goal

Empirically demonstrate attractor dynamics in a real LLM (Mistral-7B)



---



\### Phase 1: Prompting fails



| Condition                | Accuracy |

|------------------------|----------|

| Direct override         | 10%      |

| Repeated override       | 7%       |

| Constraint ladder       | 10%      |

| Prompt micro-world      | 10%      |



👉 All exhibit \*\*stable incorrect convergence\*\*



---



\### Phase 2: Stronger failures



| Condition              | Accuracy |

|----------------------|----------|

| Variable substitution | 0%       |

| Constraint equations  | 0%       |

| Two-stage transfer    | 0%       |



👉 Indicates a \*\*deep attractor basin\*\*



---



\### Phase 3: Partial recovery



| Condition                    | Accuracy |

|-----------------------------|----------|

| Ontological declaration (Z) | 75%      |

| Ontological declaration (Y) | 92%      |



👉 Explicit semantic framing alters the trajectory



---



\### Phase 4: Constraint closure



| Condition                  | Accuracy |

|---------------------------|----------|

| Exhaustive uniqueness      | 100%     |



👉 \*\*Full suppression of competing attractors\*\*



---



\### Phase 5: Multi-step degradation



| Condition                      | Accuracy |

|--------------------------------|----------|

| Substitution chains (with)     | 56%      |

| Substitution chains (without)  | 44%      |



👉 Constraint closure is \*\*local, not persistent\*\*



---



\## Core Findings



\### 1. Errors are structured, not random



Incorrect outputs are:

\- stable

\- repeatable

\- resistant to prompt variation



---



\### 2. Prompt engineering alone does not work



\- Repetition

\- Emphasis

\- Micro-world construction



→ All fail to escape attractors



---



\### 3. Constraint closure is required



Correct behaviour emerges only when:



\- alternative interpretations are explicitly excluded

\- the inference trajectory is tightly constrained



---



\### 4. Reasoning is not stable across steps



Even when single steps are correct:



\- multi-step chains degrade

\- constraints are not preserved



---



\## Interpretation



DLMH suggests that LLMs operate as:



> High-dimensional dynamical systems with attractor basins



Where:



\- Prompts define \*\*initial conditions\*\*

\- Internal priors define \*\*energy landscape\*\*

\- Outputs are \*\*trajectory endpoints\*\*



---



\## Implications



\### For AI safety

\- Stable hallucinations are \*\*predictable failure modes\*\*

\- Not random noise



\### For prompt design

\- Reinforcement is insufficient

\- Must enforce \*\*constraint closure\*\*



\### For cognition

\- Suggests a shared mechanism with human confabulation

\- Supports trajectory-based models of thought



---



\## Running the Experiments



\### Requirements

\- Python 3.10+

\- PyTorch

\- Transformers

\- NumPy, Matplotlib



---



\### LLM Experiment



Requires a \*\*local model\*\* (no downloads performed):



```python

MODEL\_PATH = r"your\_local\_mistral\_snapshot"

