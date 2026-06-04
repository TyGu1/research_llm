# Literature Review

Surveyed as of June 2026. ~40 papers across 5 technical families.

---

## Family 1: Prompt Sanitization and Entity Substitution

The most deployable family. Detect sensitive entities, replace with pseudonyms or abstractions,
send to cloud LLM, re-map the response back locally.

### Pr𝜀𝜀mpt: Sanitizing Sensitive Prompts for LLMs
- **Authors:** Amrita Roy Chowdhury, David Glukhov et al.
- **Year:** 2025 (arXiv:2504.05147)
- **Technique:** Cryptographically rigorous sanitizer. Classifies sensitive tokens into:
  (1) format-dependent tokens (SSNs, credit cards) — protected via Format-Preserving Encryption (FPE);
  (2) value-dependent tokens (age, salary) — protected via metric Differential Privacy (mDP).
  Client registers once, receives sanitized prompt, sends to untrusted cloud LLM, desanitizes locally.
- **Quality:** Near-lossless utility; outperforms prior methods.
- **Limitation:** Requires pre-classification of token sensitivity types.
- **Link:** https://arxiv.org/abs/2504.05147

### Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection
- **Authors:** Yu Chen, Tingxin Li, Huiming Liu, Yang Yu
- **Year:** 2023 (arXiv:2309.03057)
- **Technique:** Two-stage. "Hide": anonymizes private entities (LLM or label-based substitution).
  "Seek": small locally-trained model de-anonymizes the returned LLM results — solving the
  critical gap that naive anonymizers fail to recover replaced entities in model outputs.
- **Quality:** Optimal balance on translation and classification tasks.
- **Limitation:** Local de-anonymization model adds inference overhead.
- **Link:** https://arxiv.org/abs/2309.03057

### CON-QA: Privacy-Preserving QA Using Cloud LLMs in Contract Domain
- **Year:** 2025 (arXiv:2509.19925)
- **Technique:** Three-stage hybrid for enterprise legal QA:
  (1) Local LLM decomposes query and retrieves document chunks;
  (2) GLiNER NER model detects sensitive entities; one-to-many anonymization mapping prevents
      cross-session entity inference;
  (3) Cloud LLM generates response over anonymized text; local system reconstructs original answer.
- **Results:** Private Entity Restoration Accuracy = 98.8%; Response Relevancy = 97.8%;
  Answer Correctness = 88.1%; Faithfulness = 98.9%. Evaluated on CUAD-QA (85k QA pairs, 510 contracts).
- **Limitation:** Dependent on NER accuracy; mapping complexity grows with session length.
- **Link:** https://arxiv.org/abs/2509.19925

### Anonymous-by-Construction: An LLM-Driven Framework for Privacy-Preserving Text
- **Authors:** Federico Albanese, Pablo Ronco, Nicolás D'Ippolito
- **Year:** 2026 (arXiv:2603.17217)
- **Technique:** Fully on-premise LLM pipeline replaces PII with realistic, type-consistent
  surrogates. Evaluated against Microsoft Presidio, Google DLP, ZSTS on three dimensions:
  privacy, semantic utility, trainability (for downstream fine-tuning).
- **Results:** State-of-the-art on combined privacy-utility-trainability frontier.
- **Limitation:** Requires on-premise LLM capacity; cost scales with corpus.
- **Link:** https://arxiv.org/abs/2603.17217

### LLM-Redactor: An Empirical Evaluation of Eight Techniques
- **Authors:** Justice Owusu Agyemang et al.
- **Year:** 2026 (arXiv:2604.12064)
- **Technique:** Benchmarks 8 approaches on 1,300 labeled samples with 4,014 annotations:
  (A) local-only inference, (B) redaction + placeholder restoration, (C) semantic rephrasing,
  (D) TEE inference, (E) split inference, (F) FHE, (G) MPC, (H) DP noise.
  Open-source shim compatible with MCP and any OpenAI-compatible API.
- **Results:** A+B+C combined: 0.6% combined PII leak, zero exact leaks across 500 samples,
  at cost of 31.3% proprietary code leak. No single technique dominates.
- **Link:** https://arxiv.org/abs/2604.12064

### Robust Utility-Preserving Text Anonymization (RUPTA)
- **Authors:** Tianyu Yang, Xiaodan Zhu, Iryna Gurevych
- **Year:** 2024/2025 (arXiv:2407.11770; ACL 2025 Main)
- **Technique:** Three LLM modules: privacy evaluator, utility evaluator, optimization component.
  Explicitly defends against LLM-based re-identification attacks — key insight: modern LLMs
  can reason over dispersed patterns to re-identify individuals even from classically-anonymized text.
- **Results:** Outperforms baselines on re-identification robustness + downstream utility.
- **Code:** github.com/UKPLab/acl2025-rupta
- **Link:** https://arxiv.org/abs/2407.11770

### Operationalizing Data Minimization for Privacy-Preserving LLM Prompting
- **Authors:** Jijie Zhou, Niloofar Mireshghallah, Tianshi Li
- **Year:** 2025 (arXiv:2510.03662)
- **Technique:** Priority-queue tree search to find the minimum information users must share
  to maintain response quality. Tested on ShareGPT, WildChat, CaseHold (legal), MedQA (medical).
- **Key finding:** GPT-4-class models tolerate 85.7% information redaction while maintaining
  task performance. Smaller models (Qwen2.5-0.5B) degrade past 19.3% redaction.
  LLMs exhibit "bias toward abstraction that leads to oversharing."
- **Link:** https://arxiv.org/abs/2510.03662

### Preempting Text Sanitization Utility in Resource-Constrained Settings
- **Authors:** Robin Carpentier et al.
- **Year:** 2024/2025 (arXiv:2411.11521)
- **Technique:** Middleware SLM predicts, before submission, whether a DP-sanitized prompt
  will produce poor output. If predicted utility is below threshold, request is rejected or
  re-routed rather than wasted on a cloud call.
- **Results:** Prevents resource waste for up to 20% of prompts.
- **Link:** https://arxiv.org/abs/2411.11521

### Privacy- and Utility-Preserving NLP with Pseudonymization
- **Year:** 2023 (arXiv:2306.05561; ACL TrustNLP Workshop)
- **Technique:** Evaluates multiple pseudonymization techniques (rule-based to LLM-based) on
  text classification and summarization. LLM-based pseudonymization generally outperforms rule-based.
- **Link:** https://arxiv.org/abs/2306.05561

---

## Family 2: Differential Privacy for Text — and Its Fundamental Limitation

Applies mathematical noise to token embeddings before sending to cloud. Provides formal
privacy guarantee parameterized by epsilon (ε). Lower ε = more privacy = worse utility.

### Foundational Methods

**SanText** (Yue et al., 2021): Exponential Mechanism over word embedding distances. Baseline.

**CusText** (ACL Findings 2023; arXiv:2207.01193): Customized output sets with arbitrary
similarity measures. Substantially higher accuracy than SanText at same epsilon.

**TEM: Truncated Exponential Mechanism** (arXiv:2107.07928): Optimizes sampling calibrated
to local token density.

### Recent Advances

### CAPE: Context-Aware Prompt Perturbation with Differential Privacy
- **Year:** 2025 (arXiv:2505.05922; ICML 2025)
- **Technique:** Token-level DP perturbation; hybrid utility function integrating contextual
  information with token embedding distances. Bucketized sampling addresses large NLP vocabulary.
  Runs entirely on client side — no model modification needed.
- **Link:** https://arxiv.org/abs/2505.05922

### InferDPT: Privacy-Preserving Inference for Closed-Box LLMs
- **Year:** 2023 (arXiv:2310.12214)
- **Technique:** DP exponential mechanism over embedding distances + knowledge-distillation-style
  extraction module to produce coherent text from noisy generations.
- **Results:** Quality comparable to non-private GPT-4 at ε=6.0.
- **Link:** https://arxiv.org/abs/2310.12214

### DYNTEXT: Semantic-Aware Dynamic Text Sanitization
- **Year:** 2025 (ACL 2025 Findings; aclanthology.org/2025.findings-acl.1038)
- **Technique:** Three steps: (1) density information modeling extracts per-token embedding density
  under DP; (2) dynamic adjacency list construction per token; (3) private token replacement via
  similarity sampling. Adapts protection level to semantic context.
- **Link:** https://aclanthology.org/2025.findings-acl.1038/

### CluSanT: Differentially Private and Semantically Coherent Text Sanitization
- **Year:** 2025 (NAACL 2025; aclanthology.org/2025.naacl-long.187)
- **Technique:** Unifies SanText, TEM, CusText as special cases with extremal parameters.
  LLM-assisted clustering for semantically coherent replacements.
- **Link:** https://aclanthology.org/2025.naacl-long.187/

### ★ The Kill-Shot Paper: DP Sanitization Can Be Reversed

### Reconstruction of Differentially Private Text Sanitization via LLMs
- **Authors:** Shuchao Pang, Zhigang Lu, Yongbin Zhou et al.
- **Year:** 2024 (arXiv:2410.12443; ICLR OpenReview:wEqCzKFIvU)
- **Technique:** Two attacks — black-box (in-context learning) and white-box (fine-tuning).
  Both exploit that LLMs memorize training data and can bridge sanitized → original.
- **Results:**

  | Model | Recovery rate |
  |---|---|
  | LLaMA-2 70B | 72.18% |
  | LLaMA-3 70B | 82.39% |
  | Gemma-2 | 75.35% |
  | GPT-4o | 91.2% |
  | Claude-3.5 Sonnet | **94.01%** |

- **Implication:** Word-level DP sanitization is insufficient when the cloud LLM has broad
  training data coverage. This fundamentally limits all pure-DP approaches for high-stakes data.
- **Link:** https://arxiv.org/abs/2410.12443

---

## Family 3: Split Inference + Embedding Noise

Run the token embedding layer locally; add noise to intermediate activations before sending
to the cloud. Cloud processes noisy activations and returns output.

### Split-and-Denoise (SnD): Protect LLM Inference with Local DP
- **Year:** 2023 (arXiv:2310.09130)
- **Technique:** Token embedding layer on client; LDP noise added to embeddings before
  transmission. Local denoiser (pre-trained on public data + synthetic noise) recovers clean
  output from noisy server outputs. No model parameter modification needed.
- **Results:** +10% average improvement over baselines under same privacy budget.
- **Link:** https://arxiv.org/abs/2310.09130

### Privacy-Aware Split Inference with Speculative Decoding over WANs
- **Year:** 2026 (arXiv:2602.16760)
- **Technique:** Splits transformer between trusted local GPU and untrusted cloud GPU.
  Communicates only intermediate activations. Lookahead decoding (no separate draft model).
  First detailed empirical WAN deployment measurement.
- **Link:** https://arxiv.org/abs/2602.16760

### DP Split Inference via Stochastic Quantization and Soft Prompt
- **Year:** 2026 (arXiv:2602.11513)
- **Technique:** Embedding layer on user side. Latent representations encoded via differentially
  private stochastic n-bit quantization — simultaneously provides LDP and reduces communication cost.
- **Link:** https://arxiv.org/abs/2602.11513

### PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference
- **Authors:** Junfei Zhan et al.
- **Year:** 2025 (arXiv:2511.22788; AAAI 2026)
- **Technique:** Four-stage cloud-edge framework:
  (1) Edge profiles entity-level sensitivity;
  (2) Soft gating selects execution mode (cloud, edge, or collaborative);
  (3) Two-layer LDP applied per-entity risk;
  (4) Cloud generates "semantic sketch" from perturbed prompt; edge SLM refines with local context.
- **Results:** 40–50% energy and latency reduction vs. uniform and selective LDP baselines.
- **Link:** https://arxiv.org/abs/2511.22788

### What Does the Server See? Privacy Leakage from Split Inference
- **Year:** 2025 (arXiv:2605.23158)
- **Technique:** Analysis of what information is actually learnable by server from intermediate
  activations in split inference.
- **Link:** https://arxiv.org/abs/2605.23158

---

## Family 4: Trusted Execution Environments (TEEs)

Run the complete LLM inference pipeline inside a hardware-secured enclave. Input, output, and
model weights are cryptographically hidden from the cloud provider.

### Confidential LLM Inference: Performance Across CPU and GPU TEEs
- **Authors:** Marcin Chrapek, Marcin Copik, Etienne Mettaz, Torsten Hoefler (ETH Zurich)
- **Year:** 2025 (arXiv:2509.18886)
- **Models tested:** Llama2-7B, 13B, 70B
- **Hardware tested:** Intel TDX, SGX (with AMX acceleration); NVIDIA H100 Confidential Compute

  | Hardware | Throughput overhead | Latency overhead |
  |---|---|---|
  | Intel TDX/SGX (CPU, AMX) | < 10% | < 20% |
  | NVIDIA H100 CC (GPU) | 4–8% | Diminishes at batch scale |

- **Assessment:** CPU TEEs can be more cost-effective than GPU TEEs depending on scenario.
  Best practical option for strong privacy today.
- **Link:** https://arxiv.org/abs/2509.18886

### Good-Enough LLM Obfuscation (GELO)
- **Authors:** Anatoly Belikov, Ilya Fedotov
- **Year:** 2026 (arXiv:2603.05035)
- **Technique:** For shared-accelerator settings. Hides hidden states from adversaries with
  memory read access (KV caches). Non-orthogonal and orthogonal mixing with "shield" vectors
  that pollute higher-order statistics.
- **Results:** ~20–30% latency overhead; defeats ICA/BSS and anchor-based attacks on Llama-2 7B.
- **Link:** https://arxiv.org/abs/2603.05035

### AloePri: Covariant Obfuscation for Industrial LLM Inference
- **Authors:** Yu Lin, Qizhi Zhang et al. (ByteDance + Nanjing University)
- **Year:** 2026 (arXiv:2603.01499)
- **Technique:** Jointly transforms data and model parameters (covariant obfuscation) so each
  Transformer component preserves inference accuracy while hiding input/output. First designed
  for industrial LMaaS applications.
- **Link:** https://arxiv.org/abs/2603.01499

### PrivacyRestore: Privacy Removal and Restoration
- **Year:** 2024 (arXiv:2406.01394)
- **Technique:** Server pre-trains "restoration vectors" per privacy span type. Client removes
  privacy spans, aggregates restoration vectors via Attention-aware Weighted Aggregation (AWA)
  into a single meta vector. Server uses activation steering during inference.
  AWA prevents attackers from inferring privacy spans from the meta vector.
- **Link:** https://arxiv.org/abs/2406.01394

---

## Family 5: Homomorphic Encryption (FHE) and Secure Multi-Party Computation (MPC)

Theoretically ideal: input is encrypted, computation happens on ciphertext, only client can
decrypt. No information leaks even to the cloud provider. Currently too slow for interactive use.

### EncryptedLLM: Privacy-Preserving LLM Inference via GPU-Accelerated FHE
- **Authors:** Leo De Castro, Daniel Escudero et al. (J.P. Morgan AI Research)
- **Year:** 2025 (ICML 2025; OpenReview:PGNff6H1TV)
- **Technique:** LLM evaluated homomorphically using CKKS/TFHE. First open-sourced GPU
  extension of OpenFHE library. GPU acceleration achieves 200× speedup vs. CPU baseline.
- **Status:** Demonstrated on GPT-2. Still not real-time for large models.
- **Link:** https://proceedings.mlr.press/v267/de-castro25a.html

### Practical Secure Inference for Fine-tuned LLMs with FHE
- **Year:** 2025 (arXiv:2501.01672)
- **Technique:** FHE + LoRA. Gaussian kernel replaces softmax (FHE-friendly approximation).
- **Results:** 1.61 seconds/token for small fine-tuned models.
- **Link:** https://arxiv.org/abs/2501.01672

### PermLLM: Private Inference within 3 Seconds under WAN
- **Authors:** Fei Zheng, Chaochao Chen et al. (Zhejiang University)
- **Year:** 2024 (arXiv:2405.18744)
- **Technique:** Two-party private inference using Secure Random Permutation to accelerate
  non-linear function evaluation. Combined with optimized secret sharing + FHE.
- **Results:** ~3 seconds/token for ChatGLM-6B at 10ms RTT / 1Gbps. Best known MPC result.
- **Limitation:** 3s/token is still slow for interactive use; 10ms RTT / 1Gbps is infeasible for many WANs.
- **Link:** https://arxiv.org/abs/2405.18744

### Privacy-Preserving LLM Inference in Practice (IACR Survey)
- **Year:** 2026 (IACR ePrint 2026/105)
- **Coverage:** Practical comparison of TEE, MPC, FHE, hybrid approaches for Transformer inference.
  Proposes deployment trajectory: TEE now → crypto-augmented → FHE as long-term goal.
- **Link:** https://eprint.iacr.org/2026/105

---

## Privacy-Preserving RAG

### The Good and the Bad: Exploring Privacy Issues in RAG
- **Year:** 2024 (ACL 2024 Findings; arXiv:2402.16893)
- **Findings:** RAG databases can be extracted via novel attacks. Three mitigations: post-retrieval
  filtering, DP preprocessing, synthetic data substitution.
- **Link:** https://aclanthology.org/2024.findings-acl.267/

### SAGE: Mitigating Privacy Issues in RAG via Pure Synthetic Data
- **Year:** 2024 (arXiv:2406.14773)
- **Technique:** Two-stage synthetic data generation replaces real private documents in the
  retrieval corpus. Attribute-based extraction preserves key context; agent-based refinement
  improves privacy properties.
- **Results:** Comparable retrieval performance vs. original data; substantially reduced exposure.
- **Link:** https://arxiv.org/abs/2406.14773

### Privacy-Preserving RAG with Differential Privacy
- **Year:** 2024 (arXiv:2412.04697)
- **Link:** https://arxiv.org/abs/2412.04697

---

## Federated Learning + LLMs (Training-Time Privacy)

Note: FL addresses training-time privacy (keeping private data on local devices during
fine-tuning). This is complementary to, but distinct from, inference-time prompt privacy.

### A Survey on Federated Fine-tuning of Large Language Models
- **Year:** 2025 (arXiv:2503.12016; v3 June 2025)
- **Link:** https://arxiv.org/abs/2503.12016

### PriFFT: Privacy-preserving Federated Fine-tuning via Hybrid Secret Sharing
- **Year:** 2025 (arXiv:2503.03146)
- **Link:** https://arxiv.org/abs/2503.03146

### DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device LLMs
- **Year:** 2025 (arXiv:2509.09097)
- **Link:** https://arxiv.org/abs/2509.09097

---

## Domain-Specific: Healthcare

### Keeping Private Patient Data Off the Cloud: Local LLMs for Radiology Report Anonymization
- **Authors:** A.S. Becker et al.
- **Year:** 2025 (European Journal of Radiology Artificial Intelligence; DOI: 10.1016/j.ejrai.2025.100020)
- **Technique:** Seven LLMs (Qwen-2.5-coder 7B/32B, Llama v3.1 8B/70B, Llama v3.3 70B, Phi3/4 14B)
  tested on 1,000 radiology reports. Local LLMs generate RegEx rules based on identified PHI.
- **Results:** Qwen-2.5-coder outperforms Llama and Phi. Enables HIPAA-compliant on-premise anonymization.
- **Link:** https://www.sciencedirect.com/science/article/pii/S3050577125000180

### LLM-Anonymizer: Deidentifying Medical Documents with Local LLMs
- **Year:** 2024/2025 (NEJM AI; DOI: 10.1056/AIdbp2400537)
- **Link:** https://ai.nejm.org/doi/full/10.1056/AIdbp2400537

---

## Surveys and Meta-Analysis

### A Survey on Current Trends and Recent Advances in Text Anonymization
- **Year:** 2025 (arXiv:2508.21587; IEEE DSAA 2025)
- **Coverage:** NER-based, LLM-based, formal privacy models (k-anonymity, DP), authorship
  anonymization. Healthcare, law, finance, education. Three open problems identified:
  privacy-utility trade-off, quasi-identifiers, LLM implications.
- **Link:** https://arxiv.org/abs/2508.21587

### How Do We Measure Privacy in Text? A Survey of Text Anonymization Metrics
- **Year:** 2025 (arXiv:2512.01109; ACL IJCNLP 2025 Findings)
- **Key finding:** Reviews 47 papers; identifies 6 distinct privacy notions. Existing metrics
  often fail to capture HIPAA and GDPR standards.
- **Link:** https://arxiv.org/abs/2512.01109

### Privacy-Preserving Large Language Models: Mechanisms, Applications, and Future Directions
- **Authors:** Guoshenghui Zhao, Eric Song
- **Year:** 2024 (arXiv:2412.06113)
- **Coverage:** DP, FL, cryptographic protocols, TEEs. Applications in healthcare, finance, education.
- **Link:** https://arxiv.org/abs/2412.06113
