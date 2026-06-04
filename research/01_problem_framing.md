# Problem Framing

## Context

State-of-the-art LLMs (GPT-4, Claude, Gemini) are closed-source, closed-weights, and only
available in vendor-controlled data centers. While EU-region options exist (Azure, Databricks),
they involve delays in capability rollout that harm innovation. Many companies — particularly
in healthcare, finance, and legal — are legally prohibited from sending internal data outside
their infrastructure.

## Sensitivity Taxonomy

```
S0  — Public information (press releases, public filings, general knowledge)
        → May freely leave company data centers

S1  — Confidential (internal processes, org charts, product roadmaps)
        → May not leave company

S2  — Highly confidential (merger/acquisition plans, financial projections)
        → May not leave company; breach = material legal risk

S3  — Strictly confidential (health records, personal data under GDPR/HIPAA)
        → May not leave company; breach = regulatory liability
```

## The Core Technical Challenge

```
[User prompt: S2/S3]
        |
        v
[Local Sanitizer] — translate S2/S3 → S0 without losing context
        |
        v
[Cloud SOTA LLM]  — answer the S0 version of the prompt
        |
        v
[Local Re-mapper] — translate S0 response back to original context
        |
        v
[Useful answer]
```

The fundamental tension:
- **LLM value is proportional to specificity** — the more context, the better the answer
- **Privacy requires removing specificity** — the less context sent, the safer

The research question: where is the Pareto frontier between privacy depth and answer quality,
and is it domain-specific?

## What Survives Sanitization (and What Doesn't)

| Query type | Survives sanitization? | Why |
|---|---|---|
| "Draft a merger NDA template" | Yes | Task is generic; sensitivity = zero |
| "Summarize this process doc" | Partially | Structure survives; named entities are noise |
| "Should we acquire TechCorp given their Q3?" | No | The answer IS the sensitive context |
| "Legal implications of this health data breach?" | Partially | Pattern survives; specifics lost |

## Regulatory Drivers

- **GDPR (EU):** Data minimization (Art. 5(1)(c)), data residency, DPA requirements
- **HIPAA (US):** PHI must not leave HIPAA-covered environments; ONC HTI-1 (Jan 2025) adds algorithm transparency requirements for AI in certified EHR systems
- **EU AI Act (2025–2026):** High-risk AI systems (healthcare, legal, HR) face additional compliance obligations
- **CLOUD Act (US):** US government can compel US-incorporated cloud providers to disclose data regardless of where it is stored — including EU data centers

## Key Open Question

> Can a local "sanitizer-LLM" reliably detect what is sensitive, strip it, preserve semantic
> structure, and then re-inject context into the cloud LLM's response — without the cloud
> model needing the sensitive details at all?

This is testable. Empirical setup:
1. Benchmark of S2-style prompts with known correct answers
2. Apply sanitization pipelines at varying depths
3. Measure answer quality degradation vs. sanitization strength (privacy budget ε)
4. Find the Pareto frontier: privacy ↔ answer quality, per domain
