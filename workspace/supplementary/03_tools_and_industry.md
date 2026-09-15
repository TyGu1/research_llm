# Tools, Frameworks, and Industry Approaches

---

## Open-Source Tools

### Microsoft Presidio
- **Type:** Open-source (Apache 2.0)
- **URL:** https://github.com/microsoft/presidio
- **Technique:** Modular PII detection and anonymization across text, images, and structured data.
  NLP backends: spaCy, ONNX, Transformers. Supports redaction, replacement, hashing, encryption,
  custom anonymizers.
- **Supported entities:** Credit cards, names, locations, SSNs, phone numbers, financial data,
  Bitcoin wallets, plus custom recognizers (India Aadhaar, PAN, etc.).
- **LLM integrations (2024–2025):**
  - LiteLLM Gateway proxy-based PII masking
  - LangGraph integration for AI agent pipelines
  - LangChain integration
  - RAG pipeline sanitization
  - "PII Shield" — Microsoft privacy proxy for every LLM call
- **Strengths:** Most widely integrated; mature ecosystem; customizable recognizers.
- **Limitations:**
  - No formal privacy guarantee (no DP)
  - Does not handle semantic/contextual re-identification
  - Non-English language performance varies
  - Requires tuning for domain-specific entities

### LLM Guard (Protect AI)
- **Type:** Open-source (MIT); 2.5M+ downloads
- **URL:** https://protectai.com/llm-guard
- **Technique:** Security toolkit with 15 input scanners and 20 output scanners. PII anonymization
  on input; sensitive data leak detection on output. Also covers prompt injection, secrets, toxicity,
  bias, malicious URLs. ML-based (trained on real patterns). All processing runs locally.
- **LLM compatibility:** GPT, Llama, Mistral, Falcon, any provider.
- **Limitations:** Broader security focus; anonymization quality not benchmarked vs. academic methods.

---

## Commercial Tools

### Skyflow LLM Privacy Vault
- **URL:** https://www.skyflow.com/product/skyflow-for-genai
- **Technique:** Purpose-built data vault for AI pipelines. Tokenizes/de-identifies data defined
  in a "sensitive data dictionary" before it flows through LLMs. Two-way rehydration: redacted
  data sent to LLM; original sensitive values restored locally in the response.
  Field-level, contextual, identity-aware protection.
- **Compliance:** GDPR, HIPAA, PCI, EU AI Act.
- **2024–2025 developments:**
  - Agentic AI Security Layer (Dec 2024) with Databricks + Workato partnerships
  - Runtime AI Data Security Platform for AWS AgentCore (Dec 2025)
- **Limitations:** Commercial cost at scale; relies on accurate pre-defined sensitivity schema.

### Nightfall AI
- **URL:** https://www.nightfall.ai
- **Technique:** AI-native DLP platform. 100+ AI-based detection models + LLM-based file
  classifiers + computer vision (95% accuracy). API-based scan: outgoing prompt → Nightfall →
  detected findings + redacted payload → LLM. Browser plugin for real-time monitoring of
  ChatGPT, Copilot, Gemini, Claude, DeepSeek.
- **2025–2026 developments:** AI Browser Security solution (Jan 2026); AI-Native DLP Copilot.
- **Limitations:** Cloud-based detection (data traverses Nightfall servers); subscription cost;
  primarily DLP/detection rather than semantic-preserving anonymization.

### Protecto AI
- **URL:** https://www.protecto.ai
- **Technique:** "AI Context Control Layer" — every LLM request flows through a privacy layer.
  Detects PII, PHI, and confidential enterprise data. Uses tokenization that preserves semantic
  meaning. RBAC-aware: contextual control based on identity, role, context.
  Pre-built HIPAA, GDPR, PDPL, DPDP policies.
- **Integrations:** LangChain, vector databases, model gateways.
- **Limitations:** Commercial; tokenization effectiveness depends on model understanding.

### Private AI (privateai.com)
- **Type:** Commercial (API + on-prem)
- **Technique:** Detects and replaces 50+ entity types across text, structured data, audio, video.
  Anonymization, pseudonymization, synthesis. Designed for healthcare, legal, financial services.

### Opaque Systems
- **URL:** https://opaque.co
- **Technique:** Confidential LLM Inference within secure enclaves (TEEs). Runs LLM models
  inside Opaque-managed secure enclaves optimized for Microsoft Azure Confidential Computing.
  Multi-party data analytics: multiple organizations analyze combined confidential data without
  exposing underlying raw data. Zero-trust data clean rooms.
- **Use case:** Enterprise queries remain private even from the model/service provider.

### Phala Cloud
- **URL:** https://phala.com/learn/Confidential-LLMs
- **Technique:** Confidential AI on TEE-protected GPUs (Intel TDX + NVIDIA H100/H200).
  Private LLM inference, sealed agents, verifiable compute.

---

## Cloud Provider Data Residency

### OpenAI Enterprise
- **Regions:** EU, UK, US, Canada, Japan, South Korea, Singapore, India, Australia, UAE.
- **Training:** By default, API/Enterprise data not used for model training.
- **Security:** AES-256 at rest; TLS 1.2+ in transit; Enterprise Key Management; SOC 2 Type 2; GDPR DPA.
- **Limitation:** US legal entity creates CLOUD Act exposure regardless of region.
  Fine-tuning operations may involve "temporary data relocation" (updated terms March 2025).
- **Link:** https://openai.com/enterprise-privacy/

### Azure OpenAI Service
- **Regions:** 28 regions; EU and US Data Zones (announced 2024).
- **Compliance:** GDPR, HIPAA, SOC 2, FedRAMP High.
- **Security:** Customer-Managed Keys (CMK), pseudonymization, VNet integration.
- **Feb 2025 update:** Enhanced audit logging, regional data residency, content moderation for
  regulated industries.
- **Limitation:** Fine-tuning data processing may occur in centralized location even with
  regional deployment. CLOUD Act exposure.
- **Link:** https://azure.microsoft.com/en-us/blog/enterprise-trust-in-azure-openai-service-strengthened-with-data-zones/

### AWS Bedrock
- **EU regions:** eu-west-1 (Ireland), eu-central-1 (Frankfurt), eu-north-1 (Stockholm).
- **Compliance:** GDPR (Art.28 DPA), HIPAA, SOC, FedRAMP High (GovCloud).
- **Privacy:** Customer data not used to train foundation models. Model invocations within
  customer VPC via PrivateLink — data doesn't traverse public internet.
- **Limitation:** US legal entity → CLOUD Act exposure; AWS cannot inspect PrivateLink
  invocations but legal process could compel disclosure.
- **Link:** https://aws.amazon.com/bedrock/security-privacy-responsible-ai/

### Google Vertex AI
- **Regions:** Available but historically less clear than Azure/AWS (improving 2025–2026).
- **Limitation:** Does not hold FedRAMP High (unlike Azure and Bedrock as of 2026).

### Databricks
- **Approach:** Databricks AI Security Framework (DASF) — 62 identified security risks,
  53 recommended controls. DBRX open-source LLM for organizations reducing proprietary API
  reliance. All customer data encrypted at rest with CMK support. Partnership with Protect AI
  for red-teaming Mosaic AI Model Serving endpoints.
- **Link:** https://www.databricks.com/blog/introducing-databricks-ai-security-framework-dasf

---

## The CLOUD Act Problem

The US Clarifying Lawful Overseas Use of Data (CLOUD) Act (2018) allows US law enforcement
to compel US-incorporated cloud providers to disclose data regardless of where it is physically
stored — including EU data centers.

**Implication:** Even a fully GDPR-compliant EU-hosted deployment on Azure/AWS/GCP can be
legally compelled to disclose data by US government order. TEEs provide technical protection
(the provider cannot read the data) but cannot prevent court orders.

**Mitigations:**
- EU-incorporated cloud providers (OVHcloud, Hetzner, Deutsche Telekom)
- On-premise deployment
- TEE with open-source models (provider doesn't hold the model or the key)
