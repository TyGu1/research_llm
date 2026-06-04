# Research Assessment: Is This Worth Pursuing?

**Short answer: yes — as an empirical paper with clear scope boundaries.**

---

## What Makes It Worth Pursuing

### The Bottleneck Mechanism Is Architecturally Sound

Topic modeling is a *many-to-one* compression. An enormous number of different S2 prompts
map to the same topic distribution. That is fundamentally different from entity substitution,
which is *one-to-one* and therefore reversible.

The reconstruction attacks that break DP sanitization (arXiv:2410.12443) achieve 72–94%
recovery precisely because the substitution mapping is invertible. A topic distribution is not.
You cannot reconstruct "TechCorp, €2.3B, 60% EU market share" from
"{M&A: 0.6, antitrust: 0.25, SaaS: 0.15}" — the compression is irreversible by design.

### The Vocabulary Exclusion Variant Is the Strongest Element

It is not in the literature. Stripping domain vocabulary from the generated text before
question generation adds a second layer of many-to-one compression: the cloud LLM sees
neither the original content nor the vocabulary that would anchor a topic-to-instance mapping.

That combination — topic bottleneck + vocabulary exclusion — has not been studied empirically.
It is a novel contribution.

### The Synthesis Step Is Underrated

The insight that a local LLM can apply an externally-obtained reasoning framework to
locally-held specifics mirrors how a consultant works:

- Look up best practices externally (cloud LLM answers S0 proxy questions)
- Apply them to a confidential internal situation (local LLM synthesizes with S2 context)

The cloud LLM is the "best practices" engine. The local LLM is the "apply to our situation"
engine. No sensitive data crosses the boundary at any point.

### The Gap in the Literature Is Real

The existing literature handles generic PII (names, SSNs, emails) well. Nothing specifically
addresses S2-level business-contextual sensitivity (M&A strategy, competitive intelligence,
internal process) with a formal sensitivity taxonomy. The CON-QA paper (contracts domain) is
the closest work and it is very domain-narrow. A publishable contribution exists here.

---

## The Honest Caveat: Operating Envelope

The pipeline solves the right problem for reasoning queries and the wrong problem for
factual/quantitative queries. The bottleneck strips exactly the information that
quantitative answers depend on.

| Task type | Pipeline fit |
|---|---|
| "How should I think about this strategic decision?" | Strong |
| "What compliance steps apply to this situation?" | Strong |
| "What process should we follow for X?" | Strong |
| "What are the risks of this category of deal?" | Strong |
| "Is €2.3B the right bid for this specific target?" | Weak — specifics stripped |
| "What is TechCorp's competitive position?" | Weak — entity stripped |

This is not fatal. It defines the operating envelope. A system that reliably handles
60–70% of enterprise LLM use cases (reasoning, compliance, process) with provable privacy
properties is highly valuable. The remaining 30–40% (quantitative, entity-specific) can be
routed to a local model or flagged for human review.

A system that knows its limits is more deployable than one that claims to work everywhere.

---

## What Would Make This a Strong Paper

### Result 1: The Reconstruction Attack Test (Headline Result)

Run the reconstruction attack methodology (arXiv:2410.12443) on:
- (a) topic-modeled text without vocabulary exclusion
- (b) topic-modeled text with vocabulary exclusion

If vocabulary-excluded text resists recovery at < 10% — against GPT-4o and Claude 3.5
Sonnet, which achieve 91–94% on DP-sanitized text — that is a direct challenge to the
dominant approach in the field and a strong empirical contribution.

### Result 2: Task-Type Quality Curves (Core Practitioner Contribution)

Empirical Pareto curves: privacy depth vs. answer quality, across task types.

This gives practitioners a principled basis for routing decisions:
- Framework queries → topic-bottleneck pipeline
- Quantitative queries → local model or TEE

### Result 3: Cross-Domain Generalization

If the pipeline matches or beats CON-QA (contracts) and PRISM (edge-cloud routing) on their
own domains while also working on M&A, healthcare, and internal process domains — that is
the positioning against prior work.

---

## The Paper in One Sentence

> "We propose a topic-bottleneck architecture for privacy-preserving LLM query reformulation,
> characterize its operating envelope across task types, and demonstrate that its vocabulary
> exclusion variant resists LLM reconstruction attacks that break existing DP-based methods."

---

## Decision

Pursue. The novel elements are:
1. Topic modeling as an irreversible privacy bottleneck (vs. reversible substitution/DP)
2. Vocabulary exclusion as a second compression layer (not in existing literature)
3. Local synthesis using externally-obtained S0 reasoning frameworks
4. Empirical characterization of the operating envelope by task type

The scope is honest and the contribution is real.
