# ADR-002: v0.2 — Fast Local Model and Latency Reduction

**Status:** Accepted
**Date:** 2026-06-22
**Author:** Andreas Tonio Liebrand

## Context

The v0.1 pipeline uses Phi-3 (3.8B, 4K context) via Ollama as the local model for all privacy steps (1–4) and local synthesis (step 6). Measured on 48 CUAD samples with Azure GPT-4o as the cloud model:

| Component | Mean latency | Share |
|---|---|---|
| Step 1: Topic extraction | 2.6s | 6.6% |
| Step 2: S0 text generation | 3.8s | 9.8% |
| Step 3: Proxy question generation | 2.3s | 6.0% |
| Step 4: Question selection | 0.8s | 1.9% |
| **Privacy overhead (steps 1–4)** | **9.5s** | **24.4%** |
| Step 5: Cloud answering | 24.1s | 61.8% |
| Step 6: Local synthesis | 5.4s | 13.8% |
| **Total** | **38.9s** | **100%** |

The pipeline adds a **1.6x slowdown** vs. calling the cloud directly (38.9s vs. 24.1s). The 9.5s privacy tax is dominated by local LLM inference on Phi-3, which ran on CPU in our setup.

For v0.2, we need to:
1. Replace Phi-3 with a faster model suitable for on-device deployment
2. Reduce the privacy overhead to <5s per query
3. Maintain or improve answer quality

## Decision

### Local Model: Replace Phi-3 with GPT-4o-mini

**GPT-4o-mini** is the new default local model for v0.2:

| Property | Phi-3 (v0.1) | GPT-4o-mini (v0.2) |
|---|---|---|
| Context window | 4K | 128K |
| Latency per call | ~2–4s (CPU) | ~0.3–0.5s |
| Cost | Free (local) | $0.15/1M input tokens |
| Quality | Good | Better instruction following |
| Truncation needed | Yes (2000 tok) | No |

**Why GPT-4o-mini:** It's fast (~0.3s per call), cheap ($0.15/1M input), has 128K context (no truncation), and produces higher quality topic extractions and synthesis than Phi-3. The privacy architecture remains intact — GPT-4o-mini runs the local steps (1–4, 6) and never sees the cloud's responses combined with the sensitive context in a single call.

**Note on threat model:** Using a cloud model for "local" steps means the topic extraction call does see the sensitive query. In a production deployment, this would be replaced with an on-premise model. For research benchmarking, GPT-4o-mini gives us the best latency and quality baseline to evaluate the pipeline's privacy properties independent of local model limitations.

### Latency targets for v0.2

| Metric | v0.1 (Phi-3) | v0.2 target | How |
|---|---|---|---|
| Privacy overhead | 9.5s | <3s | Faster local model + batching |
| Total pipeline | 38.9s | <30s | Steps 1-3 parallelizable |
| Slowdown factor | 1.6x | <1.3x | Cloud step dominates |

### Specific optimizations

1. **Remove context truncation.** Qwen2.5-3B has 32K context — set `local_model_max_tokens: 0`. No more information loss from cutting CUAD contracts at 2000 tokens.

2. **Batch steps 1+2.** Topic extraction and S0 text generation can be a single LLM call ("Extract topics AND generate S0 text"). Saves one round-trip (~2.5s).

3. **Stream step 3 from step 2.** Start generating proxy questions as soon as S0 text is available, don't wait for full text generation to complete.

4. **Cache topic distributions.** If the same document is queried multiple times, cache the topic distribution (step 1) and skip directly to step 2.

## Config changes

```python
# v0.1
local_model: str = "phi3"
local_model_max_tokens: int = 2000  # context truncation for phi3 (4k ctx)

# v0.2
local_model: str = "gpt-4o-mini"
local_model_max_tokens: int = 0     # no truncation (128K context)
```

## What changes in the paper

- Replace "Phi-3 (Microsoft), a 3.8B-parameter model" with the new default
- Update latency numbers in Section 7.3 and Appendix C
- Add a local model ablation comparing Phi-3, Qwen2.5-3B, and GPT-4o-mini

## Consequences

- **Better:** 128K context means full CUAD contracts fit without truncation → better topic extraction and synthesis quality
- **Better:** ~0.3s per call vs ~2–4s → privacy overhead drops from 9.5s to ~2s per query
- **Better:** Higher quality instruction following → better proxy questions and synthesis
- **Trade-off:** GPT-4o-mini is a cloud model, so using it as "local" means the topic extraction step sees the sensitive query via a cloud API. For research this is acceptable; for production deployment, swap to an on-premise model.
- **Risk:** Need to re-run evaluation (48 samples × 3 attack variants) with the new model to verify privacy/quality metrics hold
- **Cost:** At $0.15/1M input tokens, 48 samples with ~5 calls each ≈ $0.01 total — negligible
