# Paper Writing Frameworks and Best Practices

Surveyed as of June 2026. Covers AI-powered paper generation systems, writing methodologies,
and venue-specific guidance relevant to this project.

---

## AI-Powered Paper Generation Frameworks

### PaperOrchestra (Google, April 2026)

- **Paper:** arXiv:2604.05018
- **Project page:** yiwen-song.github.io/paper_orchestra/
- **What it does:** Transforms unconstrained pre-writing materials (notes, code, results) into
  submission-ready LaTeX manuscripts with literature review and generated figures.
- **Architecture:** Specialized agents (Outline Agent, Plotting Agent, Literature Agent)
  operating in parallel with iterative self-reflection. ~60-70 LLM API calls, ~40 minutes
  per paper.
- **Results:** 50-68% absolute win rate over baselines in literature review quality; 14-38%
  improvement in overall manuscript quality. Generates ~46 citations per paper (vs 10-14 for
  competing systems).
- **Benchmark:** Introduced PaperWritingBench — 200 reverse-engineered papers from CVPR 2025
  and ICLR 2025.
- **Relevance:** Closest match to our "code complete, need paper" situation.

### Agent Laboratory

- **Repo:** github.com/SamuelSchmidgall/AgentLaboratory
- **What it does:** Multi-agent system with PhD, Postdoc, ML Engineer, SW Engineer, and
  Professor roles. Three phases: Literature Review (arXiv API), Experimentation (code
  generation via mle-solver), Report Writing (LaTeX via paper-solver).
- **Modes:** Autonomous (fully hands-off) or Co-pilot (human feedback at checkpoints).
- **Relevance:** Co-pilot mode with human checkpoints fits a "code complete, need paper"
  workflow. More general than PaperOrchestra but less polished for paper-specific output.

### OpenDraft

- **Repo:** github.com/federicodeponte/opendraft
- **Hosted:** Free tier at openpaper.dev (3 papers/day)
- **What it does:** 19 specialized agents across research, structure, writing, citation
  validation, and export phases. Generates 10,000-20,000+ word documents in 10-20 minutes.
- **Key differentiator:** Citation verification against CrossRef, OpenAlex, Semantic Scholar,
  arXiv — eliminates hallucinated references.
- **Output:** PDF, Word, LaTeX. Supports 57+ languages.
- **Cost:** ~$0.35-$3.00 per draft depending on model (Gemini Flash to Claude Opus).

### Stanford STORM

- **Repo:** github.com/stanford-oval/storm
- **What it does:** Two-stage pipeline — (1) Pre-writing: internet research to gather
  references + outline creation via perspective-guided question asking and simulated expert
  conversations; (2) Writing: generates full-length articles with inline citations.
- **Co-STORM** (2024): Adds human-AI collaborative curation with dynamic mind map, multiple
  LLM agents (experts + moderator), discourse protocol for turn management. EMNLP 2024.
- **Technical stack:** DSPy-based modular design, LiteLLM integration (GPT-4, Claude, etc.),
  supports You.com/Bing/Brave/Tavily/Google search backends.
- **Limitation:** Generates Wikipedia-style articles, not venue-formatted research papers.
  Best used as a research synthesis/outline generation tool.

### AI Scientist v2 (Sakana Labs, 2025)

- **Repo:** github.com/SakanaAI/AI-Scientist-v2
- **Paper:** arXiv:2504.08066
- **What it does:** End-to-end autonomous research — hypothesis generation, experiment
  execution, data analysis, manuscript writing, and self-review. Uses best-first tree search
  (BFTS) with experiment manager agent.
- **Key achievement:** First fully AI-generated manuscript accepted through peer review
  (workshop paper, avg reviewer score 6.33/10, top ~45%).
- **Relevance:** More relevant for automating full research loops. Less relevant when
  experiments are already designed and code is complete.

### WriteHERE

- **Repo:** github.com/principia-ai/WriteHERE
- **What it does:** Heterogeneous Recursive Planning (HRP) — dynamically decomposes writing
  into retrieval, reasoning, and composition subtasks. Can generate 40,000+ word reports.
- **Status:** Accepted as oral presentation at EMNLP 2025.
- **Best for:** Long-form writing (reports, surveys), less targeted at conference papers.

### Orchestra AI-Research-SKILLs

- **Repo:** github.com/Orchestra-Research/AI-Research-SKILLs
- **What it does:** 98 skills for the full AI research lifecycle, including ML Paper Writing
  with LaTeX templates and citation verification for NeurIPS/ICML/ICLR/ACL/AAAI.
- **Integration:** Plugs directly into Claude Code, Hermes Agent, Cursor, Gemini CLI.
- **Relevance:** Lowest friction option if using Claude Code. Skills-based, not end-to-end.

### Academic Writing Agents (Claude Code Plugin)

- **Repo:** github.com/andrehuang/academic-writing-agents
- **What it does:** Multi-agent orchestrator with 10 specialist agents for academic writing
  review, research, drafting, and polishing. Designed specifically for Claude Code.

---

## Writing Methodology: Best Practices for ML/Security Papers

### Paper Structure (IMRAD for Empirical ML)

1. **Abstract** (150-250 words) — 5-sentence formula (Farquhar via Nanda):
   - Achievement: "We introduce/prove/demonstrate..."
   - Significance: Why this is hard and important
   - Method: How you solve it (include specialist keywords)
   - Evidence: What validation you provide
   - Remarkable result: Your most striking number

2. **Introduction** (1-1.5 pages) — Methods should start by page 2-3
   - Opening hook (2-3 sentences)
   - Background/challenge
   - Your approach
   - 2-4 contribution bullets (specific, falsifiable)
   - Results preview

3. **Background/Related Work** — Simon Peyton Jones recommends pushing to end;
   security venues (USENIX) conventionally place it in Section 2.

4. **Methods/Approach**

5. **Experimental Setup + Results** — Include:
   - Baseline comparisons (tune baselines properly)
   - Ablation studies
   - Statistical significance
   - Error bars / confidence intervals

6. **Discussion/Limitations** — All top venues now explicitly reward transparency.

7. **Conclusion**

### Writing Order (Recommended)

Write what you know first, build narrative around it:

1. Results/methods (you know this)
2. Contribution bullets (specific, falsifiable)
3. Figure plan (what figures tell your story?)
4. Introduction (reverse-engineered from results)
5. Abstract (5-sentence formula, written last)
6. Related work (after understanding your own positioning)

### Time Allocation (Neel Nanda's Rule)

Spend roughly equal time on:
- Abstract
- Introduction
- Figures
- Everything else combined

Reviewers read: title → abstract → intro → figures → methods (if still interested).

### Common Rejection Reasons

1. Vague contributions: "We study X" vs "X achieves Y% improvement over Z"
2. Weak baselines: Not tuning competitor methods
3. Cherry-picked results without noting selection criteria
4. Overclaiming beyond evidence
5. Burying the contribution in methods instead of foregrounding in intro
6. Missing reproducibility details (hyperparameters, compute, data splits)
7. Hallucinated or misleading citations

### Presenting Limited Operating Envelopes

Directly relevant to this project — the pipeline works for reasoning/framework queries
but degrades for quantitative/entity-specific queries.

- Frame as informative, not failed: "Either outcome is possible and informative"
- Set expectations in the introduction
- Be explicit about scope: "Works under conditions A, B, C but not D, E" is a strength
- Calibrate claim strength to evidence level
- NeurIPS guidelines: authors "rewarded rather than punished" for transparency

---

## Venue-Specific Requirements

### NeurIPS

- Single-column, 9-page limit (excl. references + checklist)
- 16-item mandatory checklist (limitations, statistical significance, compute, LLM usage)
- Scored on: Quality, Clarity, Significance, Originality (1-4 each)
- Novel contributions need NOT be entirely new methods
- Checklist: neurips.cc/public/guides/PaperChecklist

### ACL / EMNLP (Rolling Review)

- Scored on: Soundness, Excitement, Overall Recommendation
- Negative results and reproductions explicitly in scope
- Responsible NLP Checklist mandatory as appendix (desk rejection if incomplete)
- Desk rejection triggers: hallucinated references, anonymity violations, missing Limitations
- Contemporaneous work rule: papers <3 months before deadline excused

### USENIX Security

- Two-column, 20-page limit (incl. bibliography + appendices)
- Mandatory Open Science appendix listing all artifacts + access instructions
- Artifacts (code, data, scripts) must be available at submission time
- Ethics evaluation part of review — stakeholder-based ethics analysis required
- Must use unaltered official template

### Reference Counts (Typical)

- NeurIPS/ICML/ICLR: 30-60 references
- ACL/EMNLP: 30-50 references
- USENIX Security: 40-60 references (positions against both security and ML)

---

## Key Resources

### Writing Guides

- **Simon Peyton Jones** — "How to Write a Great Research Paper"
  simon.peytonjones.org/great-research-paper/ (slides + 34-min video)

- **Henning Schulzrinne** — Writing Technical Articles
  cs.columbia.edu/~hgs/etc/writing-style.html

- **Neel Nanda** — "Highly Opinionated Advice on How to Write ML Papers" (May 2025)
  forum.effectivealtruism.org/posts/E5qW6SKgpFpSfD5Pp/

- **Auto-Empirical-Research-Skills Writing Guide** — Synthesis of Nanda, Karpathy,
  Gopen & Swan, Lipton, Steinhardt, Farquhar
  github.com/brycewang-stanford/Auto-Empirical-Research-Skills/.../writing-guide.md

### Literature Discovery Tools

- **Semantic Scholar** — AI-powered academic search
- **Connected Papers** — Citation graph visualization from seed papers
- **Litmaps** — Visual literature mapping
- **PaperQA** (github.com/future-house/paper-qa) — QA over scientific literature

### LLM Security/Privacy Paper Collection

- **chawins/llm-sp** (github.com/chawins/llm-sp) — Curated papers on prompt injection,
  jailbreak, privacy extraction/memorization/PII leakage/inference attacks.
  Useful for positioning related work.

### Document Tooling Stack (Recommended 2026)

| Layer | Recommended | Alternative |
|---|---|---|
| Document format | LaTeX (conferences) | Typst (PDF-only venues) |
| Editor | Overleaf (collab) / VS Code + LaTeX Workshop | Octree (AI-native) |
| Reference manager | Zotero + Better BibTeX | JabRef |
| AI writing assistant | Writefull ($65/yr) | Paperpal |
| Experiment tracking | Weights & Biases (free academic) | MLflow |
| Code-paper linking | Papers with Code | GitHub repo |
