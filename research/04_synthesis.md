# Synthesis: Cross-Cutting Findings

---

## 1. The Privacy-Utility Trade-off Is Unavoidable — But Quantifiable

Every approach trades off privacy depth against answer quality. The empirical picture from the
literature (as of 2026):

| Approach | Privacy strength | Utility preserved | Latency cost |
|---|---|---|---|
| Entity redaction + substitution | Medium (no formal guarantee) | 85–99% | Very low |
| Token-level DP (ε ≥ 6) | Formally bounded | Degrades below ε=6 | Low |
| Split inference + LDP embeddings | Medium | High | Low–medium |
| Format-Preserving Encryption | High (for structured tokens) | Near-perfect | Very low |
| TEE (CPU: TDX/SGX, GPU: H100 CC) | High (hardware guarantee) | 92–96% | Low (4–20% overhead) |
| FHE | Strongest (information-theoretic) | Near-perfect (once practical) | Extreme |
| Secure MPC | Very high | High | Very high (3s/token best case) |
| Synthetic data for RAG | High | Comparable to original | Medium |
| Federated learning | High (training only) | Comparable to centralized | Medium |
| Local-cloud hybrid routing | Variable | High for correct routes | Low–medium |

## 2. Token-Level DP Is Fundamentally Broken Against Modern LLMs

The landmark paper (arXiv:2410.12443, ICLR 2025) demonstrated that modern LLMs can reconstruct
DP-sanitized text with 72–94% recovery rates:

```
LLaMA-2 70B:      72.18%
LLaMA-3 70B:      82.39%
Gemma-2:          75.35%
GPT-4o:           91.2%
Claude-3.5 Sonnet: 94.01%
```

The mechanism: LLMs memorize broad training data; DP-sanitized text is within the distribution
of that training data; in-context learning can bridge the gap. This is not a model-specific
vulnerability — it worsens as models improve.

**Implication for this research:** Pure DP-based prompt sanitization should not be considered
a sufficient defense for S2/S3 data against a cloud LLM adversary.

## 3. The Deployability Hierarchy (Practical to Theoretical, 2026)

```
DEPLOYABLE TODAY (weeks)
├── Local routing: sensitive queries → local model; S0 queries → cloud
├── Entity redaction + substitution (Presidio, Skyflow, Nightfall, LLM Guard)
└── Semantic rephrasing via local LLM (pre-processes prompt before cloud send)

DEPLOYABLE WITH HARDWARE INVESTMENT
└── TEE inference: Intel TDX/SGX or NVIDIA H100 CC (4–20% overhead)
    Commercial: Opaque Systems, Phala Cloud, Azure CC

RESEARCH → EARLY PRODUCTION (2–3 years)
├── Split inference + LDP embeddings (PRISM architecture)
└── Hybrid cloud-edge routing with entity-level sensitivity

RESEARCH STAGE (5+ years to production)
├── FHE: 200x GPU speedup still means minutes/token for GPT-4-class models
└── Secure MPC: 3s/token best case (PermLLM on ChatGLM-6B at 1Gbps)
```

## 4. Large Models Tolerate Aggressive Redaction

From arXiv:2510.03662 (tested across 9 LLMs, 4 datasets including legal and medical):

- **GPT-4-class models:** Tolerate 85.7% information redaction while maintaining task performance
- **Small models (Qwen2.5-0.5B):** Degrade past 19.3% redaction

**Implication:** The practical sanitization floor is much lower than intuition suggests — for
sophisticated queries, you may not need to send most of the context to get a useful answer.
LLMs also exhibit a "bias toward abstraction that leads to oversharing" — models don't know
what they actually need to complete a task.

## 5. Domain Matters Enormously

| Domain | Best-in-class result | Key paper |
|---|---|---|
| Legal contracts | 98.8% entity restoration, 88.1% answer correctness | CON-QA (arXiv:2509.19925) |
| Radiology reports | HIPAA-compliant on-premise anonymization | Becker et al. (EJRAI 2025) |
| General PII | 0.6% PII leak, zero exact leaks | LLM-Redactor (arXiv:2604.12064) |
| General text | 85.7% redaction tolerated | arXiv:2510.03662 |

S2 business data (mergers, strategy) and multi-stakeholder S3 (health + legal) are barely
addressed — the gap this research targets.

## 6. Semantic Re-identification Is an Unsolved Problem

Even perfectly anonymized text can be re-identified:

- If the described scenario is unique (a niche M&A in a specific industry with a specific structure)
- If the cloud LLM has enough world knowledge to narrow the set of plausible originals
- If the adversary can aggregate multiple queries from the same session

The RUPTA paper (arXiv:2407.11770, ACL 2025) explicitly defends against LLM-based
re-identification attacks — demonstrating that classical methods are insufficient.

## 7. No Privacy Metric Is Legally Sufficient

From arXiv:2512.01109 (ACL IJCNLP 2025):

> Reviews 47 papers across 6 distinct privacy notions. Existing metrics "often fail to fully
> capture legal standards and user expectations" — GDPR and HIPAA compliance cannot be
> certified by any current anonymization tool alone.

This is a significant gap: the research community lacks a metric that maps to regulatory
compliance, not just technical privacy bounds.

## 8. CLOUD Act Is an Architectural Constraint

Even TEEs don't protect against US government legal process served to a US-incorporated
cloud provider. True sovereignty requires:
- EU-incorporated cloud infrastructure (OVHcloud, Hetzner, Deutsche Telekom)
- On-premise deployment with open-source weights
- TEE with open-source models where the provider holds neither model nor key
