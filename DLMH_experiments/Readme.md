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



\## Experiment 1: Walsh-Space Model



\*\*File:\*\* `walsh\_dlmh\_demo\_upgraded.ipynb`



\### Goal

Demonstrate that attractor behaviour emerges naturally in a distributed representation.



\### Method

\- Represent signals in Walsh (Hadamard) space

\- Apply perturbations / truncations

\- Measure reconstruction stability



\### Key Result

\- Even with degraded or partial information, the system converges to \*\*stable outputs\*\*

\- These outputs may be incorrect but remain \*\*highly invariant\*\*



👉 This provides a \*\*geometric / signal-level model\*\* of DLMH



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

