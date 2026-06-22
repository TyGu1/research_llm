# PoPETs Review: Topic-Bottleneck Query Reformulation for Privacy-Preserving Cloud LLM Inference

---

## 1. Summary

This paper proposes "topic-bottleneck query reformulation," a pipeline for privacy-preserving inference with cloud-hosted LLMs. The core idea is to compress a sensitive enterprise query into a topic distribution using a local model, generate generic proxy questions from that distribution, send only these proxy questions to the cloud LLM, and then locally synthesize a final answer by combining the cloud's generic responses with the original sensitive context. Evaluated on 48 contract QA pairs from CUAD, the pipeline achieves 2.3% specific value recovery (vs. 72--94% for DP methods per Pang et al.) while preserving 89.3% BERTScore F1. An optional vocabulary exclusion layer reduces recovery to 0% at a significant quality cost.

---

## 2. Strengths

- **Well-motivated problem.** The tension between LLM utility (requiring specific prompts) and privacy (requiring removing specifics) is real and important for enterprise adoption. The paper articulates this clearly.

- **Conceptually clean architecture.** The six-step pipeline is easy to understand and the "consultant analogy" (look up generic best practices externally, apply to confidential data locally) is intuitive and well-chosen.

- **Honest limitations section.** The paper explicitly acknowledges the single-dataset limitation, small sample size, unevaluated attack variants (ICL, absence fingerprint), and entity-specific query weakness. This level of honesty is commendable and unusual.

- **Absence fingerprint vulnerability disclosure.** The authors proactively identify a second-order information leak introduced by their own vocabulary exclusion mechanism and propose four mitigations. This demonstrates intellectual honesty and security-mindedness.

- **Clear threat model.** The adversary model is explicit about what is and is not protected (domain recovery is accepted; specific value recovery is the target). The sensitivity classification taxonomy is useful.

- **Practical deployability argument.** The pipeline requires only a local SLM and a cloud API -- no specialized hardware, no cryptographic protocols, no model modifications. This is a genuine advantage over TEE/FHE/split-inference approaches.

- **Task-type breakdown.** The per-task-type analysis provides actionable deployment guidance rather than a single aggregate number, and the identification of an "operating envelope" is practically valuable.

- **Good related work coverage.** The survey of entity substitution, DP text methods, split inference, TEE/FHE, and data minimization is thorough and fairly recent.

---

## 3. Weaknesses

- **No formal privacy guarantee.** The paper claims "irreversible by design" and "information-theoretic compression" but provides no formal proof or information-theoretic bound. The many-to-one argument is intuitive but informal. At a privacy venue like PoPETs, the absence of any formal guarantee (even a relaxed one) is a significant gap. The paper acknowledges this as future work (Section 8) but deploys strong language ("irreversible," "impossible by design") without the formal backing.

- **Very small evaluation scale.** N=48 samples (12 per task type) is insufficient for reliable conclusions. The 95% CI for entity-specific recovery is [0.000, 0.258] -- spanning from zero to 26%. Many of the reported differences are not statistically significant at this sample size. For a systems paper this might be acceptable as a proof of concept, but the paper makes quantitative claims ("2.3%") that the sample size cannot reliably support.

- **Single dataset, single domain.** All evaluation is on CUAD legal contracts. The claim that "privacy properties should transfer" to healthcare, finance, etc. is unsubstantiated. Topic distributions for medical queries may behave very differently from legal ones (e.g., rare disease queries might map to highly specific topic distributions that are nearly invertible).

- **Unfair comparison with DP methods.** The 2.3% vs. 91--94% comparison is the paper's headline result, but it compares across different datasets, different tasks, and different threat models. The DP numbers are from Pang et al. on their datasets; the 2.3% is on CUAD. The paper acknowledges this caveat in one sentence (Section 6.1: "a direct comparison across datasets requires caution") but then proceeds to make this comparison the centerpiece of every section, the abstract, and Figure 1. This is an overclaim. A fair comparison would require running DP baselines on the same CUAD data.

- **Missing critical baselines.** The paper compares against oracle, local-only, and entity substitution, but does not run any DP method on CUAD. Given that the entire narrative rests on outperforming DP, the absence of a same-dataset DP baseline is a serious experimental gap. Running SanText or CusText on the same 48 CUAD samples would significantly strengthen the paper.

- **The "local-only" baseline raises questions.** Local-only achieves 93.3% BERTScore F1 vs. the pipeline's 89.3%. The paper dismisses this by saying local-only "provides no cloud LLM reasoning capability" and that its high similarity reflects "lexical overlap rather than superior answer utility." But no evidence is provided for this claim. If a 3.8B local model already achieves 93.3% quality, why is the cloud LLM needed at all? The paper needs a more rigorous quality analysis (e.g., human evaluation, factual correctness assessment) to justify the pipeline's added complexity.

- **Two of four proposed attack variants are unevaluated.** The ICL attack and absence fingerprint attack are described but not tested. The ICL attack is particularly concerning because an adversary with access to example (S2, S0) pairs could potentially learn to invert the topic bottleneck. Without evaluating this, the privacy claims are incomplete.

- **Privacy metric definition is unclear.** "Specific value recovery rate" is the primary privacy metric, but its precise definition and measurement methodology are not described. How is recovery determined -- by exact string match, fuzzy match, semantic equivalence? Who judges recovery -- an automated system, human annotators, the LLM itself? This is critical for reproducibility.

---

## 4. Detailed Comments

1. **Abstract, line 36:** "2.3% specific value recovery rate (compared to 91--94% for DP methods)" -- This cross-dataset comparison should be flagged as indirect in the abstract itself, not buried in Section 6.1.

2. **Section 1, line 65:** "reconstruction from the distribution is impossible by design" -- This is too strong without a formal proof. The reconstruction of the *exact* query is impossible, but an adversary might narrow down the space significantly if the topic distribution is fine-grained or if there are few plausible instances. Consider: if only one company in a specific sector is undergoing an acquisition at a given time, topic + timing could be enough.

3. **Section 3.2, line 161:** The adversary model assumes the attacker sees only the proxy questions from a single query. Multi-query/session-level attacks are mentioned in limitations but not modeled. In practice, an enterprise user sends many queries, and correlating them could be devastating.

4. **Section 4.2, line 188:** The example topic extraction ("M&A 0.6, antitrust 0.25, SaaS 0.15") uses a very coarse granularity with only 3 topics. Is this always the case? What is the range of k in practice? Finer-grained topic distributions would narrow the anonymity set.

5. **Section 4.3, line 210:** There is an inconsistency: the text says vocabulary exclusion strips topic vocabulary, but the proxy question example "What EBITDA multiples are typical for SaaS acquisitions?" contains the topic vocabulary ("SaaS," "acquisitions"). This example appears to be from the non-hardened configuration, but this is not made clear and is confusing in context.

6. **Section 5.1, line 230:** The 48 samples are described as "answerable legal contract question-answer pairs" from CUAD's development split. How were these 48 selected from the full CUAD dataset? Was the selection random, stratified, or manual? If manual, selection bias could affect results.

7. **Section 5.1, lines 232--239:** The four task-type categories (12 each) appear to be manually assigned. Who performed the categorization? Was inter-annotator agreement measured? Could a query fall into multiple categories?

8. **Section 5.3, line 275--278:** The paper states that the ICL attack "was not evaluated in this run" and is "planned for future work." For a privacy paper, this is a significant gap. The ICL attack is the most realistic advanced attack scenario.

9. **Table 2 (line 337):** The entity substitution baseline achieves 82.0% BERTScore but 58.1% semantic similarity -- a large gap. The pipeline achieves 89.3% and 85.3% respectively. However, the entity substitution baseline is described as "Standard Presidio-style" with no further detail on configuration or whether it was optimized. An under-tuned baseline inflates the comparative advantage.

10. **Table 3 (line 363):** Entity-specific tasks show BERTScore F1 of 0.907, which is *higher* than framework (0.890), compliance (0.890), and procedural (0.887). This is counterintuitive -- the task type where the pipeline supposedly struggles has the highest BERTScore. The paper does not explain this anomaly.

11. **Section 6.2, line 352:** "The pipeline produces answers that are on average 1.61x the length of oracle answers." BERTScore is known to be length-sensitive. Longer answers that repeat generic reasoning content could achieve high BERTScore without actually answering the question correctly. This is a validity concern for the primary quality metric.

12. **Section 7.1, line 400:** "The qualitative difference between our 2.3% recovery rate and the 91--94% rate for DP methods is not an incremental improvement" -- This comparison is repeated too frequently throughout the paper. It appears in the abstract, introduction, Section 6.1, Section 7.1, Table 5, and the conclusion. Repetition does not strengthen an unfair comparison.

13. **Section 7.2, line 432:** "These task types together represent an estimated 60--70% of enterprise LLM use cases." No source is cited for this estimate. It appears to be an unsupported claim.

14. **Section 7.3, line 453:** Latency of 39 seconds per query is acknowledged but not analyzed in depth. For many enterprise use cases, this would be prohibitive.

15. **Table 5, line 408:** The comparison table lists entity substitution as "Invertible" for recovery, but the paper's own entity substitution baseline achieved 82% BERTScore, suggesting it is a functional (if imperfect) approach. "Invertible" without qualification is misleading.

16. **References:** Several bibliography entries use placeholder author names like "{CON-QA Authors}" (line 33 of refs.bib) or omit authors entirely. This is sloppy and should be corrected.

---

## 5. Questions for the Authors

1. How exactly is "specific value recovery rate" measured? What constitutes a successful recovery -- exact match, partial match, semantic equivalence? Is it automated or human-judged?

2. How were the 48 CUAD samples selected? Was it random sampling from the development split, or were they hand-picked? Were any samples excluded, and if so, why?

3. Why was no DP baseline run on the same CUAD dataset? This would be the most impactful addition to the evaluation.

4. Why does entity-specific BERTScore (0.907) exceed framework BERTScore (0.890)? If the pipeline degrades on entity-specific queries, shouldn't quality be lower?

5. What is the distribution of k (number of topics) across the 48 samples? Is k always 3 as in the example, or does it vary? How sensitive are the results to k?

6. The local-only baseline achieves 93.3% BERTScore. Can you provide evidence (e.g., human evaluation, factual accuracy scoring) that the pipeline's answers are qualitatively better despite the lower BERTScore?

7. For the vocabulary exclusion variant, semantic similarity drops from 0.853 to 0.413 while BERTScore drops only from 0.893 to 0.823. Why is the semantic similarity drop so much steeper? Does this indicate that BERTScore is not capturing a real quality degradation?

8. Has any analysis been done on the size of the anonymity set -- i.e., how many distinct sensitive queries map to the same topic distribution in practice?

---

## 6. Missing References

- **Utpala et al. (2023), "Locally Differentially Private Document Generation Using Zero-Shot Prompting"** -- Directly relevant as it combines LDP with LLM-based text generation.

- **Li et al. (2024), "Privacy-Preserving Prompt Tuning for Large Language Model Services"** -- Addresses prompt-level privacy with a different mechanism.

- **Duan et al. (2024), "Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models"** (ICML 2024) -- DP prompt learning, relevant to the DP baseline discussion.

- **Xie et al. (2024), "DP-OPT: Make Large Language Model Your Privacy-Preserving Prompt Engineer"** (ICLR 2024) -- Another approach to privacy-preserving prompting via local optimization.

- **Hong et al. (2024), "DP-Forward: Fine-tuning and Inference on Language Models with Differential Privacy in Forward Pass"** -- Relevant to the DP discussion.

- The paper would benefit from citing work on **topic modeling privacy** specifically (e.g., the literature on whether topic distributions can be inverted), as this directly relates to the core privacy claim.

---

## 7. Overall Recommendation

**Weak Reject**

**Confidence: High**

The paper presents an interesting and practical idea -- using topic compression as a privacy bottleneck for cloud LLM queries. The architecture is clean, the problem is well-motivated, and the limitations discussion is admirably honest. However, the paper has several significant weaknesses that prevent acceptance at a top privacy venue:

(1) The headline comparison (2.3% vs. 91--94%) is across different datasets and threat models, making it an unfair centerpiece for the paper's claims.
(2) The evaluation is too small (N=48) and too narrow (single dataset, single domain) to support the quantitative claims made.
(3) No formal privacy guarantee is provided despite strong language about "irreversibility" and "information-theoretic" properties.
(4) Critical attack variants (ICL, absence fingerprint) are acknowledged but not evaluated.
(5) The most important baseline (a DP method on the same CUAD data) is missing.

The paper reads more like a well-executed workshop paper or a strong technical report than a PoPETs-ready contribution. With a larger evaluation, same-dataset baselines, at least one formal privacy result, and evaluation of the ICL attack, this could become a solid contribution.

---

## 8. Suggestions for Improvement

1. **Run DP baselines on CUAD.** Implement SanText, CusText, or CAPE on the same 48 CUAD samples and apply the Pang et al. reconstruction attack. This single addition would transform the paper's comparative claims from indirect to direct.

2. **Scale the evaluation.** Increase to at least 200--500 samples across multiple domains (legal, healthcare, finance). CUAD alone has hundreds of samples; there is no reason to stop at 48.

3. **Provide a formal privacy bound.** Even a simple bound on mutual information I(q_s; proxy_questions | topic_distribution) would substantially strengthen the theoretical contribution. Alternatively, formalize the anonymity set size as a function of topic granularity.

4. **Evaluate the ICL attack.** Construct 5--10 example (sensitive query, proxy questions) pairs and give them to the adversary as in-context examples. If the bottleneck is truly irreversible, this attack should also fail.

5. **Clarify the recovery metric.** Provide a precise definition, examples of what counts as recovered vs. not, and ideally compute inter-annotator agreement if human judges are involved.

6. **Add human evaluation.** BERTScore alone is insufficient to assess answer quality, especially given the length ratio issue (1.61x). Have domain experts rate answer correctness and completeness on a subset.

7. **Address the local-only anomaly.** Either demonstrate that the pipeline's answers are qualitatively superior to local-only (via human evaluation or factual accuracy metrics), or reframe the contribution to acknowledge that the pipeline's value is specifically the *combination* of privacy + cloud reasoning, not quality improvement over local-only.

8. **Tone down cross-dataset comparisons.** Present the DP comparison as suggestive rather than definitive. Use language like "indicative of a qualitative difference" rather than stating the numbers side-by-side as if they were measured under the same conditions.

9. **Analyze multi-query privacy.** Even a simple simulation of what an adversary learns from observing 10 queries from the same user on the same topic would address a critical practical concern.

10. **Fix bibliography.** Complete all author fields, remove placeholder entries, and ensure consistent formatting.
