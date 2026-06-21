# ADR-001: Adopt PaperOrchestra for Paper Generation

**Status:** Proposed
**Date:** 2026-06-21
**Decision makers:** Andreas Tonio Liebrand
**Context:** Project is code-complete with no experimental results yet. Need to produce a
submission-ready academic paper from existing research notes, code, and (future) results.

---

## Context

The research_llm project has:
- 7 research documents (problem framing through assessment)
- Complete pipeline implementation (6 steps, all functional)
- 4 evaluation tests implemented
- 3 baselines implemented
- ~40 validated references in the literature review
- No experimental results yet
- No paper draft

We need a framework to help generate a submission-ready paper from these materials. Several
AI-powered paper generation frameworks exist as of June 2026.

---

## Decision

**Adopt PaperOrchestra (Google, arXiv:2604.05018) via the open-source skill pack
(github.com/Ar9av/PaperOrchestra).**

Two repositories exist:
- **Official:** github.com/google-research/paper-orchestra (Apache-2.0, "not an officially
  supported Google product", PaperWritingBench dataset not yet released)
- **Community fork:** github.com/Ar9av/PaperOrchestra (MIT, 588 stars, agent-native —
  no embedded LLM calls, delegates reasoning to host agent like Claude Code)

---

## Options Considered

### Option A: PaperOrchestra (Google) -- SELECTED

**What it is:** Multi-agent framework with 5 specialized agents (Outline, Plotting, Literature
Review, Section Writing, Content Refinement) that transforms pre-writing materials into
submission-ready LaTeX manuscripts.

**Pros:**
- Designed specifically for the "code complete, need paper" workflow
- Open-source skill pack available (github.com/Ar9av/PaperOrchestra, 588 stars, MIT license)
- Plugs directly into Claude Code (symlink skills, no API keys or SDK dependencies needed)
- Citation verification via Semantic Scholar API (prevents hallucinated references)
- Parallel execution: plotting and literature review run concurrently
- Benchmarked: 50-68% win rate over baselines in lit review quality, 14-38% improvement
  in overall manuscript quality
- Generates ~46 citations per paper (close to human average of ~59)
- Content refinement agent simulates peer review with iterative revision
- LaTeX output with conference template support
- PaperWritingBench provides evaluation methodology (200 papers from CVPR/ICLR 2025)

**Cons:**
- Evaluated only on two AI venues (CVPR, ICLR) -- not tested on security venues (USENIX)
- Benchmark uses LLM-reverse-engineered papers, not genuine lab notes
- Citation grounding verifies paper existence but not whether cited works support specific claims
- Visual fidelity: generated plots are raster images (not vector), limiting editability
- Heavy reliance on LLM-as-judge scoring; rating compression may flatten scientific novelty
- ~60-70 LLM calls per paper (~40 minutes runtime)
- Official PaperWritingBench dataset not yet released, limiting independent benchmarking
- Community adoption still early (moderate HackerNews engagement)
- Risk of "closed-loop echo chamber" if human oversight is insufficient

**Cost:** Community fork delegates LLM calls to host agent (Claude Code) -- no additional
API costs beyond existing usage. Official repo: estimated $2-10/paper depending on model
(Gemini Flash = cents, GPT-4o/Gemini Pro = higher). Optional Semantic Scholar API key for
faster citation verification. Optional Exa search at ~$0.007/query.

**Head-to-head vs AI Scientist v2:** PaperOrchestra outperforms by 39-86% win margin on
overall paper quality and 88-99% on literature review quality (automated evaluations).
AI Scientist v2 does everything but writes worse papers.

### Option B: Agent Laboratory

**What it is:** Multi-agent system with role-based agents (PhD, Postdoc, ML Engineer,
SW Engineer, Professor). Co-pilot mode with human checkpoints.

**Pros:**
- Co-pilot mode fits human-in-the-loop workflow
- Covers full research lifecycle (literature review, experimentation, report writing)
- Human feedback at checkpoints reduces risk of drift

**Cons:**
- More general-purpose -- less optimized for paper writing specifically
- Experimentation phase unnecessary (our code is already complete)
- Less benchmarked on paper quality than PaperOrchestra
- No built-in citation verification against external APIs

**Why not chosen:** Too broad. We don't need experiment automation -- we need paper generation
from existing materials. PaperOrchestra is more targeted.

### Option C: OpenDraft

**What it is:** 19 specialized agents for research, structure, writing, citation validation,
and export. Hosted version at openpaper.dev.

**Pros:**
- Strong citation verification (CrossRef, OpenAlex, Semantic Scholar, arXiv)
- Fast: 10,000-20,000+ words in 10-20 minutes
- Cheap: $0.35-$3.00 per draft
- Supports LaTeX, PDF, Word output

**Cons:**
- Optimized for theses and long-form documents, not conference papers
- Less targeted at ML venue formatting
- No peer-review simulation step
- No benchmarking against accepted papers

**Why not chosen:** Better suited for theses/dissertations. Lacks the iterative refinement
and venue-specific benchmarking that PaperOrchestra offers.

### Option D: Stanford STORM

**What it is:** Two-stage pipeline for research synthesis and outline creation via simulated
expert conversations. DSPy-based.

**Pros:**
- Strong at research synthesis and multi-perspective question asking
- Co-STORM adds human-AI collaboration (EMNLP 2024)
- Well-established (Stanford backing)

**Cons:**
- Generates Wikipedia-style articles, not venue-formatted papers
- No LaTeX output
- No conference template support
- Best as an upstream research tool, not a paper generator

**Why not chosen:** STORM is a research synthesis tool, not a paper generation framework.
Could complement PaperOrchestra for literature review but cannot replace it.

### Option E: AI Scientist v2 (Sakana Labs)

**What it is:** End-to-end autonomous research system. Generates hypotheses, runs experiments,
writes papers, and self-reviews.

**Pros:**
- Most comprehensive: covers the full research loop
- First AI-generated paper accepted through peer review
- Best-first tree search for experiment exploration

**Cons:**
- Massive overkill -- designed for autonomous research, we have completed research
- Replaces human judgment on experimental design and interpretation
- Risk of "reward hacking" LLM evaluators
- Less transparent than targeted tools

**Why not chosen:** We don't want autonomous research -- we want paper generation from
existing materials. Using AI Scientist v2 would bypass the human research decisions already
made.

### Option F: Manual Writing (No Framework)

**What it is:** Write the paper manually using LaTeX + Overleaf/VS Code, with LLM assistance
for individual sections.

**Pros:**
- Full control over every sentence
- No framework dependencies
- Proven approach for all venues

**Cons:**
- Significantly slower
- No structured citation verification
- No peer-review simulation
- No automated outline generation from research notes
- Higher risk of structural issues (burying contribution, weak intro)

**Why not chosen:** We want to leverage the structured multi-agent approach for first-draft
generation, then manually refine. PaperOrchestra with human oversight is strictly better than
manual-only for the first draft.

---

## Implementation Plan

### Phase 1: Setup

**Option A — Community fork (recommended for Claude Code users):**
```bash
git clone https://github.com/Ar9av/PaperOrchestra ~/paper-orchestra
cd ~/paper-orchestra && pip install -r requirements.txt

# Symlink skills into Claude Code
mkdir -p ~/.claude/skills
for s in paper-orchestra outline-agent plotting-agent literature-review-agent \
  section-writing-agent content-refinement-agent paper-writing-bench \
  paper-autoraters agent-research-aggregator; do
  ln -sf ~/paper-orchestra/skills/$s ~/.claude/skills/$s
done
```

**Option B — Official Google repo:**
```bash
git clone https://github.com/google-research/paper-orchestra ~/paper-orchestra
cd ~/paper-orchestra && pip install -r requirements.txt
# Requires OPENAI_API_KEY or GEMINI_API_KEY
# Optional: SEMANTIC_SCHOLAR_API_KEY for faster citation verification
# Includes Streamlit demo: streamlit run frontend/app.py
```

### Phase 2: Prepare Inputs

Structure existing research materials into PaperOrchestra workspace format:

```
workspace/inputs/
  idea.md                    -- derived from 05_research_gap_and_hypothesis.md
  experimental_log.md        -- generated from pipeline run results (after evaluation)
  template.tex               -- venue-specific LaTeX template (TBD: venue selection)
  conference_guidelines.md   -- venue formatting requirements
```

Additionally, feed the 7 existing research documents as supplementary context.

The community fork also supports an aggregation mode that discovers scattered agent caches
(.claude/, .cursor/) and synthesizes them into structured inputs:
```bash
python skills/agent-research-aggregator/scripts/discover_logs.py \
  --search-roots ~/research_llm --out workspace/ara/discovered_logs.json
```

### Phase 3: Generate First Draft

Run PaperOrchestra pipeline via Claude Code:
1. Outline Agent structures the paper
2. Plotting Agent generates figures from experimental results
3. Literature Review Agent validates and expands citations (using our 40 verified references)
4. Section Writing Agent drafts content
5. Content Refinement Agent simulates peer review

### Phase 4: Human Refinement

- Review and heavily edit all generated content
- Verify all claims match experimental evidence
- Ensure absence fingerprint vulnerability is properly discussed
- Verify all citations are accurate (cross-check against our validated reference list)
- Prepare venue-specific submission materials

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Generated text contains hallucinated claims | Human review of every claim against code/results |
| Citations don't actually support claims made | Cross-check against our validated reference list |
| Venue template not supported | Fall back to manual LaTeX with official template |
| Over-reliance on generated draft | Treat as scaffolding only; rewrite aggressively |
| Framework quality insufficient | Fall back to Option F (manual writing) |

---

## Dependencies

- **Blocking:** Experimental results must exist before Phase 2. Cannot generate a paper
  without evaluation data (recovery rates, quality scores, task-type curves).
- **Venue selection:** Template and formatting depend on target venue. This ADR is
  venue-agnostic; a separate decision is needed for venue selection.

---

## Success Criteria

1. First draft generated within 1 day of providing inputs
2. Draft covers all required sections for target venue
3. All citations verified against external APIs
4. Draft provides a usable scaffolding that reduces total paper-writing time by >50%
   compared to writing from scratch

---

## References

- PaperOrchestra paper: arxiv.org/abs/2604.05018
- Official Google repo: github.com/google-research/paper-orchestra (Apache-2.0)
- Community fork (agent-native): github.com/Ar9av/PaperOrchestra (MIT, 588 stars)
- Project page: yiwen-song.github.io/paper_orchestra/
- PaperWritingBench: 200 papers from CVPR 2025 + ICLR 2025 (not yet publicly released)
- AgentReview (peer review simulation): github.com/Ahren09/AgentReview (EMNLP 2024)
- PaperBanana (figure generation): github.com/dwzhu-pku/PaperBanana
- MarkTechPost coverage: marktechpost.com/2026/04/08/google-ai-research-introduces-paperorchestra
- EmergentMind analysis: emergentmind.com/papers/2604.05018
