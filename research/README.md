# Research: Privacy-Preserving LLM Inference

## Problem Statement

Many companies cannot send data to cloud LLMs due to data residency laws (GDPR, HIPAA) or
internal policy. State-of-the-art LLMs are closed-source and only available in vendor data
centers, often outside Europe. The core research question:

> Can a user prompt containing sensitive data be translated to a public-safe form (S0),
> answered by a SOTA cloud LLM, and the response recovered — without losing too much relevant context?

## Sensitivity Taxonomy

| Level | Definition | May leave company? |
|---|---|---|
| S0 | Publicly available information | Yes |
| S1 | Confidential (internal processes, procedures) | No |
| S2 | Highly confidential (merger plans, strategy) | No |
| S3 | Strictly confidential (health data, personal records) | No |

## Research Structure

| File | Contents |
|---|---|
| `01_problem_framing.md` | Full problem definition, taxonomy, technical challenge |
| `02_literature_review.md` | All academic papers by technical family (~40 papers) |
| `03_tools_and_industry.md` | Commercial tools and cloud provider approaches |
| `04_synthesis.md` | Cross-cutting findings, deployability hierarchy, open gaps |
| `05_research_gap_and_hypothesis.md` | Where this project fits, what is novel |

## Quick Summary of Findings

Five technical families exist (2023–2026):
1. **Entity substitution** — replace sensitive tokens, re-map response locally
2. **Differential privacy** — add mathematical noise to tokens (broken by LLMs at 72–94% recovery)
3. **Split inference** — run embedding layer locally, send noisy activations to cloud
4. **Trusted Execution Environments (TEEs)** — run full LLM in hardware enclave (4–8% overhead)
5. **FHE / Secure MPC** — fully encrypted inference (theoretically perfect, practically 3s+/token)

**Key finding:** Token-level DP, the most popular approach, can be reversed by modern LLMs
(Claude-3.5: 94%, GPT-4o: 91%). TEEs are currently the only approach that is both practical
and cryptographically strong.

**Research gap:** The literature handles generic PII well. Business-contextual sensitivity
(S2/S3: mergers, strategy, health) across a formal taxonomy is barely addressed.
