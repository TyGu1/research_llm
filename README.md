# Topic-Bottleneck Query Reformulation for Privacy-Preserving Cloud LLM Inference

Andreas Tonio Liebrand · Paper draft, November 2026

**[Read the paper (PDF)](research/privacy-preserving-llm-inference.pdf)**

## Abstract

We introduce *topic-bottleneck query reformulation*, a pipeline for privacy-preserving inference with cloud-hosted large language models (LLMs) that routes sensitive enterprise queries through a many-to-one information bottleneck.
A local model compresses a sensitive query into a topic distribution, generates generic proxy questions from that distribution, and sends only these questions to the cloud; the local model then applies the cloud's answers to the locally-held sensitive context.
Unlike entity substitution and token-level differential privacy (DP), nothing derived token-by-token from the sensitive text crosses the trust boundary.
We evaluate the defense against an *adaptive* attacker that knows the pipeline design, on 425 contract questions from 40 CUAD contracts, with a re-implemented DP baseline (SanText+) and entity substitution attacked by the same attacker on the same samples.
Six attack families—leak scanning, direct and in-context-learning reconstruction, pipeline-aware category inversion, the absence-fingerprint attack on vocabulary exclusion, embedding linkage, and multi-query aggregation—recover at most **3.7%** of sensitive values from the transmitted questions, and no party name, date or amount, while the same attacks recover **30–62%** from SanText+ output; a frontier-model attacker on a calibration subset is no stronger.
The defense does *not* hide the subject of a query: a pipeline-aware attacker identifies the clause type in 73% of queries.
We therefore claim value confidentiality, not topic confidentiality.
We also report a modest utility result: the cloud's answers help a small local model locate the right clause (+9 points gold-span overlap) but do not bring its answers closer to a strong model's.

## Repository

- `src/` — pipeline, LLM clients, data loading, baselines, attacks and evaluation code
- `experiments/` — scripts that run the evaluations and analyze results
- `results/` — experiment outputs and logs
- `tests/` — unit and integration tests
- `research/` — research notes, design decisions (ADRs) and the paper PDF
- `workspace/` — paper drafts, figures and LaTeX sources
