## S1 - Core Claim and Narrative
_Written by: outline-agent, Step 1_

**Core claim:** Topic-bottleneck query reformulation achieves 2.3% specific value recovery (vs. 91-94% for DP methods) while preserving 89.3% BERTScore F1, by routing sensitive queries through an irreversible many-to-one information bottleneck.

**Narrative tension:** Existing privacy-preserving LLM inference methods (entity substitution, DP noise) are structurally reversible by modern LLMs. The gap is a mechanism that provides irreversible privacy without requiring specialized hardware.

**Key novelty framing:** Topic modeling as a privacy mechanism (many-to-one compression), vocabulary exclusion as a novel hardening layer, and honest operating envelope characterization by task type.

**Outline decisions:**
- Plotting plan: 4 figures (pipeline diagram, recovery comparison, task heatmap, quality-privacy scatter)
- Related Work clusters: Entity Substitution, DP for Text, Hybrid/Split Inference, Data Minimization
- Section structure: Introduction, Related Work, Threat Model, Methodology, Experimental Setup, Results, Discussion, Conclusion

**Potential weaknesses flagged at outline stage:**
- Single dataset (CUAD) limits generalizability claims
- Small sample size (N=48) yields wide CIs
- Absence fingerprint vulnerability not empirically tested
- Cross-dataset comparison with Pang et al. recovery rates needs careful framing

## S2 - Literature Landscape
_Written by: literature-review-agent, Step 3_

**What the literature says about the core claim:** No existing work combines topic-bottleneck compression with proxy question generation for privacy-preserving LLM inference. The closest work (CON-QA, PRISM, HaS) all use entity-level processing rather than semantic compression.

**Strongest prior work (must address in the paper):**
- pang2024: The DP reconstruction attack paper is the primary motivating negative result
- conqa2025: Closest domain (contract QA) with different architecture
- yang2025: RUPTA explicitly addresses LLM-based re-identification

**Gaps confirmed by the literature:**
- No method provides irreversible privacy without hardware requirements
- Business-contextual sensitivity (M&A, strategy) barely addressed
- No privacy metric maps to legal compliance (GDPR/HIPAA)

**Baseline comparisons - verification status:**
| Baseline | In citation_pool? | Confidence tier |
|---|---|---|
| Pang et al. DP recovery | Yes (pang2024) | high |
| Entity substitution (Presidio) | Referenced | high |
| CON-QA | Yes (conqa2025) | high |

**Related Work cluster coverage:**
| Cluster | Papers found | Notes |
|---|---|---|
| Entity Substitution | 6 papers | Well covered |
| DP for Text | 8 papers | Comprehensive |
| Hybrid/Split Inference | 5 papers | Good coverage |
| Data Minimization | 4 papers | Adequate |
