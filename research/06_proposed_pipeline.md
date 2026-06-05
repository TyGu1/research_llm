# Proposed Pipeline: Topic-Bottleneck Query Reformulation

## Core Idea

Instead of sanitizing a sensitive prompt by replacing or noising its tokens, compress it
through a semantic bottleneck (topic modeling) that strips specific sensitive content while
preserving the reasoning structure. Generate proxy S0 questions from that bottleneck. Answer
those with a SOTA cloud LLM. Synthesize the final answer locally using the original S2 question
as context.

No sensitive data ever leaves the local environment.

---

## Pipeline

```
[S2 Question]
  e.g. "Should we acquire TechCorp at €2.3B given their 60% EU market share?"
        |
        v
[Step 1 — Local LLM: Topic Modeling]
  Extract topic distribution from the question
  e.g. {M&A: 0.6, antitrust: 0.25, enterprise SaaS: 0.15}
        |
        v
[Step 2 — Local LLM: Generate 200–300 words about those topics]
  Generic, factual, S0 text about the topic space
  No reference to the original question's specifics
        |
    ── OPTIONAL HARDENING ──────────────────────────────────
    | Strip all topic vocabulary words from the generated text  |
    | What remains: semantic structure without the lexicon      |
    | e.g. "An entity evaluating consolidation with a dominant  |
    |       counterpart in a regulated sector..."               |
    ────────────────────────────────────────────────────────────
        |
        v
[Step 3 — Local LLM: Generate 8–12 questions from that text]
  e.g.
  - "What EBITDA multiples are typical for SaaS acquisitions?"
  - "How is antitrust risk assessed for dominant European tech players?"
  - "What due diligence steps matter most in cross-border software M&A?"
  - "How does market concentration affect regulatory approval timelines?"
        |
        v
[Step 4 — Local LLM: Compare original S2 question to the 8–12 candidates]
  Select the 3 closest proxy questions
  (original S2 question seen only by local LLM — never leaves)
        |
        v
[Step 5 — Cloud SOTA LLM: Answer the 3 S0 proxy questions]
  Receives no sensitive data
  Answers with full capability on generic S0 questions
        |
        v
[Step 6 — Local LLM: Synthesize final answer]
  Inputs: original S2 question + 3 S0 answers from cloud LLM
  Output: answer grounded in cloud reasoning, applied to local S2 specifics
```

---

## Key Innovation: The Information Bottleneck

The central mechanism is **lossy semantic compression via topic modeling**.

Prior approaches either:
- Replace sensitive entities with pseudonyms (substitution) — reversible by modern LLMs at 72–94%
- Add DP noise to tokens — reversible at similar rates (arXiv:2410.12443)

This approach does neither. Topic modeling compresses the question into a *semantic category
distribution*, not a modified version of the original text. What survives compression:

- The type of problem (M&A reasoning)
- The relevant knowledge domains (valuation, antitrust, SaaS)
- The reasoning structure (comparative, risk-weighted, regulatory)

What does not survive:
- Specific entity names (TechCorp)
- Specific numerical values (€2.3B, 60%, Q3 EBITDA)
- Temporal specifics (Q3, current deal timeline)

These non-surviving elements are exactly the S2 content.

---

## The Vocabulary Exclusion Hardening Variant

An additional step between Step 2 and Step 3: strip all topic-vocabulary words from the
200–300 word generated text before passing it to question generation.

**Example transformation:**

Without exclusion:
> "In software M&A, antitrust regulators examine market concentration metrics. EBITDA multiples
> for SaaS targets typically range 10–20x depending on growth rate and churn."

With exclusion (topic words removed: M&A, antitrust, market, concentration, EBITDA, SaaS):
> "In strategic consolidation, regulators examine dominance metrics. Cash-flow multiples for
> targets typically range 10–20x depending on growth rate and attrition."

The second version conveys the same logical structure but uses no domain-specific vocabulary
that could anchor re-identification. No paper found in the literature (as of June 2026) applies
this specific technique.

---

## Where This Sits vs. Prior Work

| Component | Closest prior work | Key difference |
|---|---|---|
| Topic modeling as privacy bottleneck | PRISM "semantic sketch" (arXiv:2511.22788) | PRISM uses LDP + edge SLM refinement; here topic modeling is the bottleneck itself |
| Synthetic proxy question generation | SAGE for RAG (arXiv:2406.14773) | SAGE generates synthetic documents; this generates synthetic questions |
| Local LLM synthesizes using S0 answers | HaS "Seek" step (arXiv:2309.03057) | HaS re-maps entity substitutions; this applies S0 reasoning frameworks to S2 specifics |
| Vocabulary exclusion from topic text | — | **Not found in existing literature. Appears novel.** |
| Full pipeline | — | **No prior paper combines topic bottleneck + proxy question selection + local synthesis** |

---

## Strengths

1. **No entity-level substitution required** — avoids the substitution→reconstruction attack
   that breaks DP and Presidio-style approaches at 72–94% recovery rates.

2. **Architecturally clean** — each step has a well-defined role; the bottleneck location is
   explicit and auditable.

3. **Local LLM sees the S2 question twice** (steps 4 and 6) but only as the input to
   local-only reasoning — never as content to transmit.

4. **Cloud LLM receives genuinely S0 content** — the proxy questions are derived from topic
   modeling of the question, not from the question itself.

5. **Composable** — the vocabulary exclusion variant can be toggled independently based on
   the sensitivity level (S1 → standard, S2 → exclusion on, S3 → exclusion + suppression of
   high-uniqueness topics).

---

## Challenges and Limitations

### Challenge 1: What survives the bottleneck?

The bottleneck preserves *what kind of problem this is* but strips *what the specific
problem is*. This matters differently by task type:

| Task type | Bottleneck preserves enough? |
|---|---|
| Framework / reasoning ("how should I think about X?") | Yes |
| Compliance / regulatory ("what rules apply to X situation?") | Yes |
| Process / procedure ("what steps does X require?") | Yes |
| Quantitative / specific ("is €2.3B the right bid for this target?") | No — specifics stripped |
| Factual lookup ("what is TechCorp's market position?") | No — entity stripped |

The pipeline has a **task-type dependency**: it works well for reasoning and framework
queries; it degrades for quantitative or entity-specific queries.

### Challenge 2: The synthesis step carries heavy load

In step 6, the local LLM must bridge:
- S2 original question (with specifics: €2.3B, TechCorp, 60%)
- S0 generic answers (without those specifics: SaaS M&A frameworks, antitrust heuristics)

The local LLM applies the S0 reasoning framework to its locally-held S2 specifics. This is
legitimate — but the answer quality is bounded by:
- The local LLM's ability to apply external frameworks to specific data
- How well the topic modeling captured the right semantic domain for the original question

### Challenge 3: Is topic-modeled text actually S0?

Empirically unverified. The re-identification risk does not vanish — it depends on how unique
the topic combination is. "High-multiple acquisition in concentrated European software vertical"
narrows the set of plausible targets for a domain expert even without naming them.

The vocabulary exclusion step mitigates this but needs to be tested against the reconstruction
attack methodology (arXiv:2410.12443): can a SOTA LLM recover the original S2 question from
the generated S0 text?

### Challenge 4: Question selection criterion in step 4

"Closest" requires a precise definition. Three candidates:

- **Semantic similarity (embedding distance):** Most natural, but could select questions that
  inadvertently reveal the sensitive domain if the embedding space is not privacy-neutral.
- **Structural similarity (same reasoning pattern):** Safer, but harder to operationalize.
- **Utility-maximizing selection (which S0 answer best supports final synthesis):** Ideal
  but requires anticipating step 6 — a second-order reasoning problem.

---

## Proposed Empirical Tests

### Test 1: Bottleneck Fidelity
**Question:** How well do the 3 selected proxy questions span the semantic space of the original?

- Metric: BERTScore / semantic similarity between original S2 question and proxy questions
- Baseline: entity-substituted version of original question (Presidio-style)
- Hypothesis: proxy questions cover the topic space but not the specific instance

### Test 2: Sensitive Data Recovery from Transmitted Content
**Question:** Can an attacker recover the sensitive data in the original S2 prompt from
what is actually sent to the cloud?

The tested object is not the methodology — it is the transmitted content (S0 proxy questions
and/or generated text). The methodology is irrelevant to the attacker.

```
Input to attacker:  S0 proxy questions (what the cloud actually receives)
Attack goal:        recover sensitive content from the original S2 prompt
                    (entity names, deal values, patient data, etc.)
Comparison:         DP reconstruction attack recovers 91–94% from DP-sanitized text
```

This is the same security question as arXiv:2410.12443, applied to our pipeline output
rather than DP-sanitized text. The claim is not that the methodology is hard to reverse-
engineer — it is that the transmitted content contains no sensitive data to recover, because
the information bottleneck is irreversible, not merely noisy.

- Method: run three attack variants on the S0 proxy questions:
  (a) Direct recovery prompt — attacker asks LLM to reconstruct the source situation
  (b) In-context learning attack — attacker provides (S2, S0) example pairs, then targets
  (c) Domain expert prompt — attacker poses as domain expert, guesses source situation
- Test conditions: with and without vocabulary exclusion
- Models: GPT-4o, Claude 3.5 Sonnet (highest known recovery rates in arXiv:2410.12443)
- Recovery scoring:
  - Domain recovery (M&A, healthcare, legal) — easiest; some leakage acceptable
  - Situation recovery ("large acquisition in European software") — medium
  - Specific value recovery (company name, deal size, patient identity) — must be near zero
- Success criterion: specific value recovery < 10% (vs. 91–94% for DP sanitization)

### Test 3: Answer Quality
**Question:** How close is the synthesized answer to the oracle answer?

- Oracle: cloud SOTA LLM given the full S2 question directly
- Baseline A: local LLM answering the S2 question alone (no cloud access)
- Baseline B: entity-substitution pipeline answer
- This pipeline: topic-bottleneck + synthesis
- Metrics: human evaluation, factual accuracy, BERTScore vs. oracle

### Test 4: Task-Type Sensitivity
**Question:** Does quality degradation follow the task-type hypothesis?

- Query taxonomy: framework/reasoning, compliance, quantitative, entity-specific
- Expected result: significant degradation for quantitative/entity-specific tasks;
  minimal degradation for framework/reasoning tasks
- This result would define the *operating envelope* of the pipeline

---

## Sensitivity Level Adaptation

The pipeline can be adapted per sensitivity level:

| Level | Modification |
|---|---|
| S1 | Standard topic bottleneck; no vocabulary exclusion |
| S2 | Topic bottleneck + vocabulary exclusion |
| S3 | Topic bottleneck + vocabulary exclusion + suppress topics with uniqueness score > threshold |

The uniqueness score for a topic distribution measures how many real-world entities could
plausibly generate that exact distribution — a proxy for re-identification risk.

---

## Open Questions

1. Which topic modeling method is best suited? (LDA, NMF, BERTopic, neural topic models like
   CTM) — trade-off between interpretability, topic coherence, and bottleneck strength.

2. What is the right granularity for topic modeling? Fine-grained topics (e.g.,
   "cross-border SaaS M&A") are more useful but harder to anonymize. Coarse topics
   (e.g., "corporate finance") are safer but generate less useful proxy questions.

3. Can step 4 (question selection) be replaced by step 3 directly generating S0 questions
   that are maximally useful for step 6 synthesis? This removes the selection step but
   requires a more capable local LLM.

4. How does the pipeline degrade under multi-turn conversations where session-level context
   accumulates across questions?
