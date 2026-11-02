# Adaptive-Attack Evaluation Plan (branch `adaptive-attack`, written 2026-11-02)

Goal: turn the topic-bottleneck paper from "defense evaluated against a passive
attacker on N=48" into "defense + strongest adaptive attack + honest map of where it
breaks". Every attack below is labelled with the exact observation set the attacker
gets, because the current code's absence-fingerprint attack hands the attacker the
S0 text, which the cloud never sees. Mixing observation sets is how defenses get
over- or under-claimed.

## 0. State found in step 1

| Item | Finding |
|---|---|
| Paper's raw data | `results/phi3_20260621_160315.json` is NOT in git (gitignored); recovered copy at `results/reference_phi3_2026-06/` from the old laptop snapshot. 48 samples, Phi-3 local, GPT-4o cloud + attacker, LLM selector, 10 proxy → 3 selected. ICL attack on it: mean SVR 2.1%. |
| Threat model gap | Only `selected_questions` reach the cloud (`runner.py` step 5). The coded absence-fingerprint attack uses `s0_text_excluded` + topics: an upper bound, not the realistic attacker. |
| SVR metric | exact, case-insensitive substring of the first 50 chars of each gold span in the reconstruction. Strict; paraphrases don't count. Keep as primary (comparable), add typed metrics. |
| Domain-recovery metric | any >3-char word of the category label in the reconstruction: too lax (e.g. "license"). Replace with 41-way category identification. |
| CUAD dev split | 4,690 answerable questions over 508 contracts, median 9 per contract (min 1, max 21). Task types: entity_specific 2,263 · compliance 1,136 · framework 825 · procedural 466. Median context 46k chars. |
| Local compute | vLLM :8020 is DOWN (preflight: free). Production recipe known (Gemma-4-31B-it NVFP4 + MTP, ~82 GB of the 121 GiB pool, max-num-seqs 16-24). Ollama :11434 up with only qwen2.5:3b. No competing ML jobs. 108 GB available. |
| Cloud LLM | `cloud_client.py` is OpenAI-compatible (OpenAI / Azure / "anthropic" via OpenAI-compat). Old `.env` had Azure OpenAI GPT-4o creds; validity unknown; GPT-4o list price $2.50/M in, $10/M out. Not needed for the attack study (see §2). |
| Env | miniconda base has a broken transformers/huggingface-hub pin; project `.venv` created with uv (torch 2.15 cu132, CUDA ok). `pyproject.toml` build backend name was invalid (`setuptools.backends.legacy:build`) and is fixed to `setuptools.build_meta`. 122 unit tests pass. |

## 1. Sampling: contract-grouped, not question-stratified

Draw ~45 dev-split contracts (seed 42) and take ALL their answerable questions
(expected ~400 samples, hard floor 300). This serves two purposes at once:

- CIs on ≥300 samples per the review, with the natural task-type mix (per-type CIs too).
- The multi-query attack (rung c) needs many queries per "company"; a contract is the
  company.

All confidence intervals are cluster bootstraps by contract (questions from one
contract share parties, dates and vocabulary, so a per-question bootstrap would be
too narrow). 95% CIs, 2,000 resamples, seed 42.

## 2. Models and cost

| Role | Model | Why |
|---|---|---|
| Defense local model (steps 1–4) | Gemma-4-31B-it NVFP4 on vLLM `:8020` | strongest realistic local model on the box; a stronger local model is a stronger defense, so attacks that still succeed here are the conservative finding. 8,000-char context cap (as the baselines already use). |
| Cloud stand-in (step 5, oracle, baseline answering) | same Gemma | zero cost; the utility axis is relative (pipeline vs oracle vs DP), so a shared answerer is consistent. |
| Bulk attacker | same Gemma, client concurrency 12 | free, batchable. Weaker than GPT-4o/Claude, so it under-states attack success: every "defense holds" number is a lower bound. |
| Calibration attacker | Claude Fable via `claude -p --model claude-fable-5-1` on a 30–40 sample subset, rungs a and b only | measures how much a frontier attacker gains over Gemma; subscription only, no API key (R852). |
| Optional | GPT-4o on Azure, 100-sample subset ≈ €4 (needs valid key + `cloud_spend.sh draw`) | only if Tonio wants attacker parity with the paper's original run. Default: skip. |

Load estimate: ~7k Gemma calls (pipeline ~10/sample, attacks ~6/sample, baselines
~6/sample), ~13M prefill + ~3M decode tokens ≈ 4–8 h wall-clock at concurrency 12.
Every stage checkpoints to `results/adaptive/*.jsonl` and resumes.

## 3. Attack ladder (weakest → strongest), with observation sets

Observation sets: **Q** = the 3 transmitted proxy questions (what the cloud sees);
**Q+** = Q plus pipeline knowledge (prompts, that a topic step exists, CUAD's 41 clause
categories); **S0** = the S0 text (never transmitted: upper-bound only, labelled as such);
**Qxk** = Q for k queries from the same contract; **Pool** = a candidate population
(the 45 contracts × 41 categories), i.e. the attacker knows the user's possible
documents (CUAD contracts are public SEC exhibits, so this is realistic).

| Rung | Attacker | Observes | Success metrics |
|---|---|---|---|
| 0. Leak scan | none (regex/exact) | Q | gold span in Q; party name of the contract in Q; any date / amount / capitalised multi-word entity in Q. Deterministic; shows where the paper's 2.3% actually came from. |
| a. Single-query LLM reconstruction | direct, expert (as in paper); ICL-k with k ∈ {3, 20} examples from OTHER contracts; ICL-same with examples from the SAME contract (auxiliary-knowledge adversary) | Q | SVR (exact), typed recovery (party / date / amount), situation score, 41-way category id |
| b1. Category inversion | LLM picks the CUAD category (41-way), then reconstructs conditioned on it | Q+ | category accuracy vs 2.4% chance; SVR after conditioning |
| b2. Absence fingerprint | infer excluded vocabulary from what the questions avoid, then reconstruct | Q+ (realistic) and S0 (upper bound) | recall/precision of the excluded word set; SVR |
| b3. Replay / dictionary linkage | attacker runs the public pipeline over Pool and matches generated proxy sets to the observed Q by embedding | Q+, Pool | top-1 / top-5 identification of (contract, category); chance 1/1845 |
| c. Multi-query aggregation | per contract, observe k ∈ {1, 3, 5, all} proxy sets; embedding linkage and LLM reconstruction over the union | Qxk, Pool | contract identification among 45 (chance 2.2%) vs k; SVR over the union of gold spans; party-name recovery vs k |

Rungs 0, b3 and c (embedding) need no attacker LLM and run in minutes; they are the
ones most likely to break the "anonymity set is combinatorially large" claim.

Vocabulary-exclusion variant: run the pipeline a second time with
`use_vocab_exclusion=True` on the same samples so rung b2 and the quality cost are
measured on identical inputs.

## 4. Baselines on the SAME samples

| Baseline | Transmits | Attacks |
|---|---|---|
| Entity substitution (existing regex, Presidio-style) | substituted question + 8k-char context | rung a (direct, ICL-20), rung c |
| SanText-style metric-LDP (Yue et al. 2021), ε ∈ {1, 3} (+10 if time) | sanitized question + 8k-char context excerpt | rung a (direct, ICL-20: the Pang et al. recipe), rung c |

SanText needs word embeddings: GloVe-300d via HF if downloadable, else MiniLM
word vectors (documented deviation). Utility for every arm = BERTScore F1 and semantic
similarity of the stand-in cloud answer vs the oracle answer, same evaluator as the paper.

## 5. Deliverables

- `src/attacks/` — `leak_scan.py`, `reconstruction.py` (a, b1, b2), `linkage.py` (b3, c embedding), `multi_query.py` (c LLM); `src/baselines/santext.py`; `src/evaluation/ci.py` (cluster bootstrap).
- `experiments/run_adaptive_eval.py` (stages: sample → pipeline ×2 → baselines → attacks → Fable calibration) and `experiments/analyze_adaptive.py` (tables with CIs, figures).
- `tests/unit/test_attacks.py`, `test_santext.py`, `test_ci.py` — mock-client, red-first.
- `workspace/drafts/adaptive_attacks.tex` — "Where the defense holds, where it breaks", with the results table and the corrected threat model.
- Branch `adaptive-attack` pushed; merge to master after `/ship review`.

## 6. Open decisions for Tonio

1. Launch the production Gemma vLLM on `:8020` for the duration (it is down; same recipe as the 2026-10-05 restore; ~82 GB of the shared pool; undo = kill that PID).
2. Optional GPT-4o attacker subset (≈ €4 of the €10/day pool) for attacker parity with the paper. Default: skip; Fable calibration covers it.
3. Stretch: a second defense config with a small local model (Gemma-4-E2B, ~6 GB, or qwen2.5:3b on Ollama) on a 100-sample subset, to show attack success vs local-model strength (the paper's Phi-3 premise). Default: only if the main ladder finishes early.
