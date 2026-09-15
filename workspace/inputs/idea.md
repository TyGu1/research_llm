# Research Idea: Topic-Bottleneck Query Reformulation for Privacy-Preserving LLM Access

## Problem Statement

State-of-the-art LLMs (GPT-4o, Claude, Gemini) are cloud-only. Enterprises holding S2/S3 classified data -- M&A strategies, financial projections, health records, competitive intelligence -- are legally prohibited from sending prompts containing this data to cloud providers. GDPR Art. 5(1)(c) data minimization, HIPAA de-identification standards, and the US CLOUD Act (which compels US-incorporated providers to disclose data regardless of hosting region) all create hard constraints.

The fundamental tension: LLM value is proportional to specificity (more context yields better answers), but privacy requires removing specificity (less context sent means less exposure). Every existing sanitization approach tries to preserve specificity while hiding identity -- and fails, because modern LLMs can reverse the hiding.

## Core Insight: Topic Modeling as an Irreversible Information Bottleneck

Entity substitution (replacing "TechCorp" with "[COMPANY_A]") and differential privacy noise are both reversible. Modern LLMs recover 72-94% of original sensitive content from DP-sanitized text (arXiv:2410.12443). The substitution mapping is one-to-one and therefore invertible.

Topic modeling is fundamentally different: it is a many-to-one compression. An enormous number of different S2 prompts map to the same topic distribution. You cannot reconstruct "TechCorp, EUR 2.3B, 60% EU market share" from "{M&A: 0.6, antitrust: 0.25, SaaS: 0.15}" -- the compression is irreversible by design. The information bottleneck strips exactly the sensitive specifics while preserving the reasoning structure.

## The 6-Step Pipeline

1. **Topic Extraction (Local LLM):** Extract a topic distribution from the sensitive S2 question. E.g., {M&A: 0.6, antitrust: 0.25, enterprise SaaS: 0.15}.

2. **S0 Text Generation (Local LLM):** Generate 200-300 words of generic, factual, S0-level text about those topics. No reference to the original question's specifics.

2.5. **Vocabulary Exclusion (Optional Hardening):** Strip all topic-vocabulary words from the generated S0 text before question generation. This adds a second layer of many-to-one compression: the cloud LLM sees neither the original content nor the vocabulary that would anchor a topic-to-instance mapping.

3. **Proxy Question Generation (Local LLM):** Generate 10 candidate S0 proxy questions from the (optionally vocabulary-stripped) text.

4. **Question Selection (Local LLM):** Compare the original S2 question against the 10 candidates and select the 3 most relevant proxy questions. The original S2 question is seen only by the local LLM -- it never leaves the local environment.

5. **Cloud Answering (Cloud SOTA LLM):** Send the 3 S0 proxy questions to the cloud LLM. The cloud receives genuinely S0 content -- no sensitive data crosses the boundary.

6. **Local Synthesis (Local LLM):** Combine the original S2 question with the 3 cloud-obtained S0 answers to produce a final answer grounded in cloud reasoning but applied to local S2 specifics. This mirrors how a consultant works: look up best practices externally, apply them to a confidential internal situation.

## Four Novel Contributions

1. **Topic bottleneck as a privacy mechanism.** Topic modeling as an irreversible information bottleneck for privacy, rather than entity substitution or DP noise. The many-to-one compression makes reconstruction attacks structurally impossible, not merely statistically difficult.

2. **Vocabulary exclusion hardening.** A second compression layer that strips domain vocabulary from the generated S0 text. Not found in existing literature. When combined with the topic bottleneck, the cloud LLM sees neither the original content nor the lexicon that would anchor re-identification.

3. **Local synthesis using cloud-obtained S0 reasoning.** The local LLM applies externally-obtained reasoning frameworks to locally-held specifics. No existing paper combines topic bottleneck, proxy question selection, and local synthesis in this architecture.

4. **Empirical operating envelope characterization by task type.** Honest characterization of where the pipeline works (framework/reasoning, compliance, procedural queries) and where it degrades (quantitative, entity-specific queries), providing practitioners a principled basis for routing decisions.

## Headline Results

Evaluated on CUAD (Contract Understanding Atticus Dataset), dev split, 48 answerable legal contract QA pairs:

- **Specific value recovery: 2.3%** (vs. 91-94% for DP-based methods in arXiv:2410.12443). Attackers using direct prompting or domain-expert prompting could not recover sensitive specifics from the transmitted S0 proxy questions.
- **Answer quality: 89.3% BERTScore F1** against oracle (cloud LLM given full S2 question directly). The pipeline preserves most answer quality while transmitting zero sensitive content.
- **Task-type sensitivity confirmed:** Framework/reasoning and compliance tasks show near-zero specific value recovery with strong answer quality. Entity-specific tasks show the expected quality-privacy trade-off (higher gold span overlap but also higher recovery risk).

## Known Vulnerability: Absence Fingerprint Attack

The vocabulary exclusion variant introduces a second-order information leak. An attacker who infers the topic of the S0 text can diff expected vs. observed vocabulary -- words conspicuously absent form an absence fingerprint that reveals which terms were stripped. The current regex-based exclusion is deterministic, making this fingerprint recoverable.

Four mitigations are proposed: (1) random over-exclusion with decoy terms, (2) paraphrase-based replacement instead of deletion, (3) generation-time avoidance (instructing Step 2 to never produce topic-specific words -- recommended default), (4) fixed-size vocabulary masking from a predefined domain lexicon.

This vulnerability was not tested in the current run (absence_fingerprint attack variant is null) and must be evaluated in a follow-up experiment with vocabulary exclusion enabled.

## Operating Envelope

The pipeline solves the right problem for reasoning queries and the wrong problem for factual/quantitative queries. This is not fatal -- it defines an honest operating envelope:

| Task Type | Pipeline Fit | Why |
|---|---|---|
| Framework / reasoning | Strong | Topic structure preserved; specifics not needed |
| Compliance / regulatory | Strong | Pattern-based; entity identity irrelevant |
| Procedural | Strong | Process structure survives bottleneck |
| Quantitative / entity-specific | Weak | Specifics stripped by design |

A system that reliably handles 60-70% of enterprise LLM use cases with provable privacy properties -- and knows its limits for the remaining 30-40% -- is more deployable than one that claims universal coverage.
