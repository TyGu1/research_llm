# Research Gap and Hypothesis

---

## Where the Literature Falls Short

The existing literature handles **generic PII** well (names, SSNs, emails, phone numbers).
The S0/S1/S2/S3 framing introduces a more nuanced and practically important problem:
**business-contextual sensitivity** and **multi-level classification**.

| What the literature addresses | What is missing |
|---|---|
| Named entity redaction (person, org, location) | S2-level contextual sensitivity (merger strategy, competitive intel) |
| Formal DP bounds on token-level noise | Sensitivity classification at the semantic level (not just entity type) |
| Domain-specific NER (medical, legal) | Cross-domain sensitivity taxonomies with empirical quality benchmarks |
| Recovery rate for individual entity types | End-to-end answer quality as a function of sanitization depth, by domain |
| "Is this PII?" classification | "What sensitivity level is this concept?" classification |
| Single-round prompt sanitization | Multi-turn conversation privacy across sessions |

The CON-QA paper (contracts domain) is the closest existing work. There is no equivalent for
M&A, internal strategy, or cross-domain healthcare-legal scenarios.

---

## Research Hypothesis

> A local, small-language-model-based sensitivity classifier can reliably identify S0/S1/S2/S3
> spans in a prompt, apply depth-appropriate sanitization (substitution, abstraction, or
> suppression), and pass the result to a SOTA cloud LLM — with measurable answer quality
> degradation that follows a predictable function of sensitivity level and task type.

Corollaries:
1. GPT-4-class models tolerate aggressive sanitization better than smaller models (supported by arXiv:2510.03662)
2. Task structure (template generation, reasoning, summarization, QA) modulates the quality
   degradation at each sanitization depth
3. A domain-specific sensitivity ontology outperforms generic NER for S2/S3 classification

---

## Proposed Research Directions

### Direction A: Empirical Benchmark (First Priority)

Build a benchmark dataset of S0–S3 labeled prompts with known correct answers, across at least
three domains (legal/M&A, healthcare, internal process). Apply sanitization pipelines at varying
depths. Measure:

- Answer quality: ROUGE, BERTScore, human evaluation, factual accuracy
- Privacy leakage: re-identification attack success rate (using the arXiv:2410.12443 methodology)
- Mapping: privacy depth ε → answer quality degradation, per domain and task type

Deliverable: **Privacy-Utility Pareto curves, per sensitivity level and domain.**

### Direction B: Sensitivity Classifier

Train or fine-tune a local SLM to classify prompt spans into S0/S1/S2/S3.
Input: a text span + surrounding context
Output: sensitivity label + confidence

Key questions:
- Can a 7B model do this reliably? (vs. human annotation baseline)
- Does domain-specific fine-tuning outperform general instruction-tuning?
- What annotation schema minimizes inter-annotator disagreement?

### Direction C: Sanitization Pipeline Architecture

Design a complete pipeline:

```
[Raw prompt: S0-S3 mixed]
        |
        v
[Local SLM: sensitivity classifier] → labels each span
        |
        v
[Sanitizer by level]:
  S0 spans → pass through unchanged
  S1 spans → entity substitution (Presidio-style)
  S2 spans → semantic abstraction (replace with role/type, not value)
  S3 spans → suppression + summary placeholder
        |
        v
[Cloud SOTA LLM] — answers the sanitized prompt
        |
        v
[Local re-mapper] — injects original S1/S2/S3 values into response
        |
        v
[Final answer: original context restored]
```

Key design questions:
- How to handle prompts where S2/S3 spans are load-bearing for the answer?
- Should the re-mapper be a rule-based system or a second LLM call?
- How to maintain consistency across multi-turn conversations?

### Direction D: Formal Guarantee

Formalize a privacy notion that:
1. Is stronger than entity redaction alone (which fails against re-identification)
2. Is weaker than FHE (which is impractical)
3. Maps to GDPR Art. 5(1)(c) data minimization and HIPAA de-identification standards

Candidate: extend Pr𝜀𝜀mpt's framework (arXiv:2504.05147) to hierarchical sensitivity levels
with domain-specific sensitivity vocabularies.

---

## Positioning vs. Existing Work

| This research | Closest prior work | Key difference |
|---|---|---|
| S0–S3 sensitivity taxonomy with empirical QA benchmarks | LLM-Redactor (arXiv:2604.12064) | Sensitivity-level framing vs. technique-type framing |
| Business-contextual sensitivity (M&A, strategy) | CON-QA (contracts) | Non-legal business sensitivity; multi-domain |
| End-to-end quality degradation curves | arXiv:2510.03662 (data minimization) | Taxonomy-driven sanitization vs. information quantity |
| Domain-specific sensitivity classifier | Generic NER (Presidio, spaCy) | Sensitivity level, not entity type |

---

## Open Questions to Resolve

1. **What counts as "not losing too much relevant context"?**
   This is domain-specific and task-specific. A merger lawyer's acceptable degradation differs
   from a radiologist's. The first step is to define and operationalize quality metrics per domain.

2. **Can semantic abstraction be automated reliably?**
   Replacing "we plan to acquire TechCorp for €2.3B in Q3" with "a major strategic transaction
   is planned" loses specificity — but is the cloud LLM's answer to the abstracted version still
   useful? This requires empirical measurement.

3. **Does the re-mapper need to be a second LLM call?**
   If the cloud LLM response contains placeholders like [TARGET_COMPANY], substituting back is
   trivial. If the response contains implicit reasoning about the abstracted entity, re-mapping
   is harder. The design of the sanitization must constrain the response format.

4. **What is the minimal viable sensitivity classifier?**
   Full annotation of S0–S3 at span level is expensive. Can a weak supervision approach
   (Snorkel-style) or a small instruction-tuned model bootstrap a useful classifier?
