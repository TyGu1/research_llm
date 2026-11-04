#!/usr/bin/env python
"""
Adaptive-attack evaluation runner (branch adaptive-attack; plan: research/09_adaptive_attack_plan.md).

Every stage is resumable: outputs are JSONL files under --run-dir keyed by sample_id
(and attack name); rows already present are skipped.

Stages
  sample      contract-grouped CUAD sample + gold-span windows        -> samples.json
  pipeline    steps 1-4 of the defense (what is transmitted), 2 arms  -> pipeline_plain.jsonl, pipeline_vocab.jsonl
  baselines   entity substitution + SanText+ (eps list)               -> arm_entity_sub.jsonl, arm_santext_e{eps}.jsonl
  leak        rung 0, every arm                                       -> leak_<arm>.jsonl
  linkage     rung b3-lite + rung c (embedding), pipeline arms        -> linkage_<arm>.json
  attacks     rungs a, b1, b2 (LLM attacker), every arm               -> attacks_<arm>.jsonl
  multiquery  rung c (LLM attacker), pipeline arms                    -> multiquery_<arm>.jsonl
  utility     stand-in cloud answers for every arm + oracle           -> utility_<arm>.jsonl

Usage
  .venv/bin/python experiments/run_adaptive_eval.py --run-dir results/adaptive/main --stage all
  .venv/bin/python experiments/run_adaptive_eval.py --run-dir results/adaptive/smoke --n-contracts 2 --stage all --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # embeddings on CPU; the GPU belongs to vLLM
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.data.cuad_loader import CUADDataset, CUADSample, Split, QUESTION_TEMPLATES
from src.data.contract_sampling import sample_by_contract, party_tokens_by_contract
from src.attacks.leak_scan import scan as leak_scan
from src.attacks.scoring import Scorer
from src.attacks.reconstruction import LLMAttacker, ICLExample
from src.attacks.linkage import EmbeddingLinker, LinkageRow, rank_of
from src.attacks.multi_query import group_by_contract, make_draws
from src.pipeline.base import TopicDistribution, S0Answer
from src.pipeline.topic_extraction import LLMTopicExtractor
from src.pipeline.text_generation import LLMTextGenerator
from src.pipeline.vocab_exclusion import RegexVocabExcluder
from src.pipeline.question_generation import LLMQuestionGenerator
from src.pipeline.question_selection import LocalEmbeddingQuestionSelector
from src.pipeline.cloud_answering import StandardCloudAnswerer
from src.pipeline.synthesis import LLMSynthesizer
from src.baselines.entity_substitution import EntitySubstitutionBaseline, SYSTEM_PROMPT as ANSWER_SYSTEM
from src.baselines.oracle import PROMPT_TEMPLATE as ANSWER_PROMPT

WINDOW_CHARS = 4000       # the "sensitive document" every arm receives: a window of the contract
PRE_SPAN_CHARS = 1000     # ... starting this many chars before the first gold span
PIPELINE_ARMS = ["plain", "vocab"]
ALL_STAGES = ["sample", "pipeline", "baselines", "leak", "linkage", "attacks", "multiquery", "utility"]
# "retokenize" (not in ALL_STAGES): recompute party tokens in samples.json with the current rule; re-run "leak" after it.


# ── io helpers ───────────────────────────────────────────────────────────────

def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class JsonlWriter:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rows: list[dict]) -> None:
        with self._lock:
            with open(self._path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_parallel(items: list, fn, writer: JsonlWriter, workers: int, label: str) -> None:
    """fn(item) -> list[dict]; rows are appended as they complete."""
    if not items:
        print(f"  [{label}] nothing to do")
        return
    t0 = time.time()
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            try:
                rows = fut.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                rows = [{"sample_id": getattr(futs[fut], "id", str(futs[fut])), "error": f"{type(e).__name__}: {e}"}]
            writer.write(rows)
            done += 1
            if done % 10 == 0 or done == len(items):
                el = time.time() - t0
                print(f"  [{label}] {done}/{len(items)} ({el:.0f}s, {el/done:.1f}s/item, errors={errors})", flush=True)


# ── context ──────────────────────────────────────────────────────────────────

class Ctx:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._embed = None
        self._client = None
        self._scorer = None
        self._samples: list[CUADSample] | None = None
        self._party: dict[str, set[str]] | None = None

    def embed(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            self._embed = SentenceTransformer(self.args.embed_model, device="cpu")
        return self._embed

    def client(self):
        if self._client is None:
            from src.llm.vllm_client import VLLMClient
            self._client = VLLMClient(base_url=self.args.vllm_url, model=self.args.model, max_tokens=700)
        return self._client

    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = Scorer(self.embed())
        return self._scorer

    def load_samples(self) -> tuple[list[CUADSample], dict[str, set[str]]]:
        if self._samples is None:
            data = json.loads((self.run_dir / "samples.json").read_text())
            self._party = {t: set(v) for t, v in data["party_tokens"].items()}
            self._samples = [
                CUADSample(
                    id=r["id"], question=r["question"], raw_question="", context=r["window"],
                    answers=r["answers_in_window"], category=r["category"], task_type=r["task_type"],
                    doc_title=r["doc_title"], is_impossible=False, split=Split.DEV,
                )
                for r in data["samples"]
            ]
            if self.args.limit:   # smoke tests: at most N questions per contract
                seen: dict[str, int] = {}
                kept = []
                for s in self._samples:
                    if seen.get(s.doc_title, 0) < self.args.limit:
                        kept.append(s)
                        seen[s.doc_title] = seen.get(s.doc_title, 0) + 1
                self._samples = kept
        return self._samples, self._party

    def by_id(self) -> dict[str, CUADSample]:
        samples, _ = self.load_samples()
        return {s.id: s for s in samples}


# ── stage: sample ────────────────────────────────────────────────────────────

def make_window(context: str, answers: list[str]) -> tuple[str, bool]:
    low = context.lower()
    positions = [low.find(a.lower()[:50]) for a in answers]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return context[:WINDOW_CHARS], False
    start = max(0, min(positions) - PRE_SPAN_CHARS)
    return context[start:start + WINDOW_CHARS], True


def stage_retokenize(ctx: Ctx) -> None:
    out = ctx.run_dir / "samples.json"
    data = json.loads(out.read_text())
    titles = {r["doc_title"] for r in data["samples"]}
    tokens = party_tokens_by_contract(CUADDataset().load())
    data["party_tokens"] = {t: sorted(v) for t, v in tokens.items() if t in titles}
    out.write_text(json.dumps(data, indent=1))
    n_with = sum(1 for t in titles if data["party_tokens"].get(t))
    print(f"  party tokens recomputed: {n_with}/{len(titles)} contracts have name tokens")
    for lp in ctx.run_dir.glob("leak_*.jsonl"):
        lp.unlink()
    print("  leak files removed; re-run --stage leak")


def stage_sample(ctx: Ctx) -> None:
    out = ctx.run_dir / "samples.json"
    if out.exists():
        print(f"  samples.json exists ({len(json.loads(out.read_text())['samples'])} samples); skipping")
        return
    ds = CUADDataset()
    all_samples = ds.load()
    tokens = party_tokens_by_contract(all_samples)
    chosen = sample_by_contract(all_samples, ctx.args.n_contracts, ctx.args.seed, Split.DEV, True, ctx.args.min_per_contract)
    rows = []
    n_found = 0
    for s in chosen:
        window, found = make_window(s.context, s.answers)
        n_found += found
        in_win = [a for a in s.answers if a.lower()[:50] in window.lower()]
        rows.append({
            "id": s.id, "doc_title": s.doc_title, "category": s.category, "task_type": s.task_type,
            "question": s.question, "answers_all": s.answers, "answers_in_window": in_win,
            "window": window, "window_contains_span": found, "context_chars": len(s.context),
        })
    out.write_text(json.dumps({
        "config": {"n_contracts": ctx.args.n_contracts, "seed": ctx.args.seed, "window_chars": WINDOW_CHARS,
                   "pre_span_chars": PRE_SPAN_CHARS, "min_per_contract": ctx.args.min_per_contract},
        "party_tokens": {t: sorted(v) for t, v in tokens.items() if t in {r["doc_title"] for r in rows}},
        "samples": rows,
    }, indent=1))
    titles = {r["doc_title"] for r in rows}
    print(f"  sampled {len(rows)} questions from {len(titles)} contracts; window contains a gold span for {n_found}")


# ── stage: pipeline (steps 1-4) ──────────────────────────────────────────────

def stage_pipeline(ctx: Ctx, variant: str) -> None:
    samples, _ = ctx.load_samples()
    out = ctx.run_dir / f"pipeline_{variant}.jsonl"
    done = {r["sample_id"] for r in jsonl_read(out) if not r.get("error")}
    todo = [s for s in samples if s.id not in done]
    client = ctx.client()
    extractor = LLMTopicExtractor(client)
    textgen = LLMTextGenerator(client, min_words=200, max_words=300)
    excluder = RegexVocabExcluder() if variant == "vocab" else None
    qgen = LLMQuestionGenerator(client, n=ctx.args.n_proxy)
    selector = LocalEmbeddingQuestionSelector(ctx.embed())

    def one(s: CUADSample) -> list[dict]:
        t: dict[str, float] = {}
        t0 = time.monotonic(); topics = extractor.extract(s.question, s.context); t["topic"] = time.monotonic() - t0
        t0 = time.monotonic(); s0 = textgen.generate(topics); t["text"] = time.monotonic() - t0
        working, s0x = s0, ""
        if excluder is not None:
            s0x = excluder.exclude(s0, topics)
            working = s0x
        t0 = time.monotonic(); gq = qgen.generate(working, n=ctx.args.n_proxy); t["questions"] = time.monotonic() - t0
        sel = selector.select(s.question, gq, k=ctx.args.n_selected)
        return [{"sample_id": s.id, "variant": variant, "topics": topics.topics, "s0_text": s0,
                 "s0_text_excluded": s0x, "generated_questions": gq, "selected_questions": sel,
                 "timings": t, "error": "" if sel else "no questions selected"}]

    run_parallel(todo, one, JsonlWriter(out), ctx.args.workers, f"pipeline:{variant}")


# ── stage: baselines ─────────────────────────────────────────────────────────

def stage_baselines(ctx: Ctx) -> None:
    samples, _ = ctx.load_samples()
    # entity substitution (no LLM call: only what is transmitted)
    out = ctx.run_dir / "arm_entity_sub.jsonl"
    done = {r["sample_id"] for r in jsonl_read(out)}
    es = EntitySubstitutionBaseline(cloud_client=None, max_context_chars=WINDOW_CHARS)
    rows = []
    for s in samples:
        if s.id in done:
            continue
        emap: dict[str, str] = {}
        counters: dict[str, int] = {}
        q = es._substitute(s.question, emap, counters)
        c = es._substitute(s.context, emap, counters)
        rows.append({"sample_id": s.id, "transmitted_question": q, "transmitted_context": c, "substitution_map": emap})
    JsonlWriter(out).write(rows)
    print(f"  entity_sub: {len(rows)} new rows")

    # SanText+
    from src.baselines.santext import SanTextPlus, load_glove_vocab
    vocab = emb = None
    for eps in ctx.args.santext_eps:
        out = ctx.run_dir / f"arm_santext_e{eps:g}.jsonl"
        done = {r["sample_id"] for r in jsonl_read(out)}
        todo = [s for s in samples if s.id not in done]
        if not todo:
            print(f"  santext eps={eps}: nothing to do")
            continue
        if vocab is None:
            vocab, emb = load_glove_vocab(ctx.args.glove_vocab)
            print(f"  GloVe vocab loaded: {len(vocab)} words, dim {emb.shape[1]}")
        st = SanTextPlus(vocab, emb, epsilon=eps, p=ctx.args.santext_p, sensitive_rank=ctx.args.santext_sensitive_rank, seed=ctx.args.seed)
        rows = []
        t0 = time.time()
        for s in todo:
            rq = st.sanitize(s.question)
            rc = st.sanitize(s.context)
            rows.append({"sample_id": s.id, "epsilon": eps, "transmitted_question": rq.text, "transmitted_context": rc.text,
                         "n_tokens": rc.n_tokens, "n_replaced": rc.n_replaced, "n_oov": rc.n_oov})
        JsonlWriter(out).write(rows)
        print(f"  santext eps={eps}: {len(rows)} rows in {time.time()-t0:.0f}s")


# ── arms: what each arm transmits ────────────────────────────────────────────

def arm_names(ctx: Ctx) -> list[str]:
    return PIPELINE_ARMS + ["entity_sub"] + [f"santext_e{e:g}" for e in ctx.args.santext_eps]


def load_arm(ctx: Ctx, arm: str) -> dict[str, dict]:
    """sample_id -> row with 'transmitted' (list[str] for pipeline arms, str otherwise)."""
    if arm in PIPELINE_ARMS:
        rows = jsonl_read(ctx.run_dir / f"pipeline_{arm}.jsonl")
        return {r["sample_id"]: dict(r, transmitted=r["selected_questions"]) for r in rows if not r.get("error") and r.get("selected_questions")}
    rows = jsonl_read(ctx.run_dir / f"arm_{arm}.jsonl")
    return {r["sample_id"]: dict(r, transmitted=f"Question: {r['transmitted_question']}\n\nContract excerpt:\n{r['transmitted_context']}")
            for r in rows if not r.get("error")}


# ── stage: leak (rung 0) ─────────────────────────────────────────────────────

def stage_leak(ctx: Ctx) -> None:
    by_id = ctx.by_id()
    _, party = ctx.load_samples()
    for arm in arm_names(ctx):
        data = load_arm(ctx, arm)
        out = ctx.run_dir / f"leak_{arm}.jsonl"
        done = {r["sample_id"] for r in jsonl_read(out)}
        rows = []
        for sid, r in data.items():
            if sid in done or sid not in by_id:
                continue
            s = by_id[sid]
            ls = leak_scan(r["transmitted"], s.answers, party.get(s.doc_title, set()))
            rows.append({"sample_id": sid, "arm": arm, **ls.to_dict()})
        JsonlWriter(out).write(rows)
        print(f"  leak {arm}: {len(rows)} new rows")


# ── stage: linkage (rungs b3-lite and c, embedding) ──────────────────────────

def stage_linkage(ctx: Ctx) -> None:
    samples, _ = ctx.load_samples()
    by_id = ctx.by_id()
    linker = EmbeddingLinker(ctx.embed(), chunk_chars=1500, max_chunks=40)
    # candidate pool: the full text of every sampled contract (public documents)
    ds = CUADDataset()
    full = {}
    titles = {s.doc_title for s in samples}
    for s in ds.load():
        if s.doc_title in titles and s.doc_title not in full:
            full[s.doc_title] = s.context
    contract_index = linker.build_index(full)
    category_index = linker.build_index({c: [t] for c, t in QUESTION_TEMPLATES.items()})
    print(f"  pool: {contract_index.n} contracts ({len(contract_index.chunk_owner)} chunks), {category_index.n} categories")

    for arm in PIPELINE_ARMS:
        data = load_arm(ctx, arm)
        cat_rows, con_rows = [], []
        qs_by_sample = {sid: r["transmitted"] for sid, r in data.items() if sid in by_id}
        for sid, qs in qs_by_sample.items():
            s = by_id[sid]
            cat_rows.append(LinkageRow(sid, s.category, rank_of(s.category, linker.rank(qs, category_index)), category_index.n).to_dict())
            con_rows.append(LinkageRow(sid, s.doc_title, rank_of(s.doc_title, linker.rank(qs, contract_index)), contract_index.n).to_dict())
        titles_by_sample = {sid: by_id[sid].doc_title for sid in qs_by_sample}
        by_title = group_by_contract(titles_by_sample, qs_by_sample)
        draws = make_draws(by_title, qs_by_sample, ks=[1, 3, 5, "all"], n_draws=ctx.args.n_draws, seed=ctx.args.seed)
        multi_rows = []
        for d in draws:
            rk = rank_of(d.doc_title, linker.rank(d.union_questions, contract_index))
            multi_rows.append({"doc_title": d.doc_title, "k": d.k, "draw": d.draw, "sample_ids": d.sample_ids,
                               "rank": rk, "top1": rk == 1, "top5": rk <= 5, "n_candidates": contract_index.n})
        (ctx.run_dir / f"linkage_{arm}.json").write_text(json.dumps({
            "arm": arm, "n_contracts": contract_index.n, "n_categories": category_index.n,
            "category_rows": cat_rows, "contract_rows": con_rows, "multi_rows": multi_rows,
        }, indent=1))
        print(f"  linkage {arm}: {len(cat_rows)} samples, {len(multi_rows)} multi-query draws")


# ── stage: attacks (rungs a, b1, b2 — LLM attacker) ──────────────────────────

def _icl_examples(ctx: Ctx, data: dict[str, dict], by_id: dict[str, CUADSample], sid: str, k: int, same: bool) -> list[ICLExample]:
    s = by_id[sid]
    pool = [o for o in data if o != sid and o in by_id and ((by_id[o].doc_title == s.doc_title) == same)]
    rng = random.Random(f"{ctx.args.seed}:{sid}:{k}:{same}")
    rng.shuffle(pool)
    out = []
    for o in pool[:k]:
        so = by_id[o]
        out.append(ICLExample(questions=data[o]["transmitted"], original_question=so.question, gold=so.answers, context_excerpt=so.context[:300]))
    return out


def _text_examples(ctx: Ctx, data: dict[str, dict], by_id: dict[str, CUADSample], sid: str, k: int) -> list[tuple[str, str]]:
    s = by_id[sid]
    pool = [o for o in data if o != sid and o in by_id and by_id[o].doc_title != s.doc_title]
    rng = random.Random(f"{ctx.args.seed}:{sid}:{k}:text")
    rng.shuffle(pool)
    return [(data[o]["transmitted_context"][:700], by_id[o].context[:700]) for o in pool[:k]]


def stage_attacks(ctx: Ctx, arm: str) -> None:
    by_id = ctx.by_id()
    _, party = ctx.load_samples()
    data = load_arm(ctx, arm)
    out = ctx.run_dir / f"attacks_{arm}.jsonl"
    done = {(r["sample_id"], r["attack"]) for r in jsonl_read(out) if not r.get("error")}
    attacker = LLMAttacker(ctx.client(), max_tokens=700)
    scorer = ctx.scorer()
    if arm in PIPELINE_ARMS:
        attack_names = ["direct", "expert", "icl3", "icl20", "icl_same", "category_inversion"]
        if arm == "vocab":
            attack_names += ["absence_q", "absence_s0"]
    elif arm == "entity_sub":
        attack_names = ["sub_direct", "dp_icl"]
    else:
        attack_names = ["dp_direct", "dp_icl"]
    if ctx.args.attacks:
        attack_names = [a for a in attack_names if a in ctx.args.attacks]

    def one(sid: str) -> list[dict]:
        s = by_id[sid]
        r = data[sid]
        tokens = party.get(s.doc_title, set())
        rows = []
        for name in attack_names:
            if (sid, name) in done:
                continue
            try:
                if name == "direct":
                    ao = attacker.direct(r["transmitted"])
                elif name == "expert":
                    ao = attacker.expert(r["transmitted"])
                elif name == "icl3":
                    ao = attacker.icl(r["transmitted"], _icl_examples(ctx, data, by_id, sid, 3, False), "icl3")
                elif name == "icl20":
                    ao = attacker.icl(r["transmitted"], _icl_examples(ctx, data, by_id, sid, 20, False), "icl20")
                elif name == "icl_same":
                    ex = _icl_examples(ctx, data, by_id, sid, 5, True)
                    if len(ex) < 2:
                        continue
                    ao = attacker.icl(r["transmitted"], ex, "icl_same")
                elif name == "category_inversion":
                    ao = attacker.category_inversion(r["transmitted"])
                elif name == "absence_q":
                    ao = attacker.absence_q(r["transmitted"], TopicDistribution(r["topics"]) if r.get("topics") else None)
                elif name == "absence_s0":
                    ao = attacker.absence_s0(r["s0_text_excluded"], TopicDistribution(r["topics"]) if r.get("topics") else None)
                elif name == "sub_direct":
                    ao = attacker.sub_direct(r["transmitted"])
                elif name == "dp_direct":
                    ao = attacker.dp_direct(r["transmitted"])
                elif name == "dp_icl":
                    obs = "SUB-text" if arm == "entity_sub" else "DP-text"
                    ao = attacker.dp_icl(r["transmitted"], _text_examples(ctx, data, by_id, sid, ctx.args.dp_icl_k), observation=obs)
                else:
                    continue
                sc = scorer.score(ao.reconstruction, s, tokens)
                extra = dict(ao.extra)
                if name == "category_inversion":
                    cats = extra.get("categories_norm") or []
                    extra["cat_top1"] = bool(cats) and cats[0] == s.category
                    extra["cat_top3"] = s.category in cats[:3]
                rows.append({"sample_id": sid, "arm": arm, "attack": ao.attack, "observation": ao.observation,
                             "reconstruction": ao.reconstruction, "prompt_chars": ao.prompt_chars,
                             "extra": extra, "score": sc.to_dict()})
            except Exception as e:  # noqa: BLE001
                rows.append({"sample_id": sid, "arm": arm, "attack": name, "error": f"{type(e).__name__}: {e}"})
        return rows

    todo = [sid for sid in data if sid in by_id and any((sid, a) not in done for a in attack_names)]
    run_parallel(todo, one, JsonlWriter(out), ctx.args.workers, f"attacks:{arm}")


# ── stage: multiquery (rung c, LLM attacker) ─────────────────────────────────

def stage_multiquery(ctx: Ctx, arm: str) -> None:
    by_id = ctx.by_id()
    _, party = ctx.load_samples()
    data = load_arm(ctx, arm)
    out = ctx.run_dir / f"multiquery_{arm}.jsonl"
    done = {(r["doc_title"], r["k"]) for r in jsonl_read(out) if not r.get("error")}
    qs_by_sample = {sid: r["transmitted"] for sid, r in data.items() if sid in by_id}
    by_title = group_by_contract({sid: by_id[sid].doc_title for sid in qs_by_sample}, qs_by_sample)
    draws, seen = [], set(done)
    for d in make_draws(by_title, qs_by_sample, ks=[3, 5, "all"], n_draws=1, seed=ctx.args.seed):
        if (d.doc_title, d.k) in seen:   # "all" coincides with k=3/5 for small contracts
            continue
        seen.add((d.doc_title, d.k))
        draws.append(d)
    attacker = LLMAttacker(ctx.client())
    scorer = ctx.scorer()

    def one(d) -> list[dict]:
        ao = attacker.multi_query(d.question_sets)
        gold = [a for sid in d.sample_ids for a in by_id[sid].answers]
        tokens = party.get(d.doc_title, set())
        from src.attacks.scoring import svr_exact, party_hit
        svr = svr_exact(ao.reconstruction, gold)
        return [{"doc_title": d.doc_title, "arm": arm, "k": d.k, "sample_ids": d.sample_ids, "reconstruction": ao.reconstruction,
                 "svr_union": svr, "any_gold": svr > 0, "n_gold": len(gold), "party_hit": party_hit(ao.reconstruction, tokens),
                 "situation_score": scorer.situation(ao.reconstruction, by_id[d.sample_ids[0]].context)}]

    run_parallel(draws, one, JsonlWriter(out), ctx.args.workers, f"multiquery:{arm}")


# ── stage: utility (stand-in cloud answers) ──────────────────────────────────

def stage_utility(ctx: Ctx) -> None:
    samples, _ = ctx.load_samples()
    if ctx.args.utility_limit:
        # a fixed, seeded subset spread over contracts: the first n of each contract in id order
        per = max(1, ctx.args.utility_limit // len({s.doc_title for s in samples}))
        seen: dict[str, int] = {}
        subset = []
        for s in samples:
            if seen.get(s.doc_title, 0) < per:
                subset.append(s)
                seen[s.doc_title] = seen.get(s.doc_title, 0) + 1
        samples = subset[: ctx.args.utility_limit]
    by_id = {s.id: s for s in samples}
    client = ctx.client()

    def answer(question: str, context: str) -> str:
        return client.generate(ANSWER_PROMPT.format(question=question, context=context), system=ANSWER_SYSTEM, max_tokens=500)

    # oracle
    out = ctx.run_dir / "utility_oracle.jsonl"
    done = {r["sample_id"] for r in jsonl_read(out) if not r.get("error")}
    run_parallel([s for s in samples if s.id not in done],
                 lambda s: [{"sample_id": s.id, "answer": answer(s.question, s.context)}],
                 JsonlWriter(out), ctx.args.workers, "utility:oracle")

    # pipeline arms: cloud answers to the proxy questions + local synthesis
    answerer = StandardCloudAnswerer(client)
    synth = LLMSynthesizer(client, max_context_chars=WINDOW_CHARS)
    for arm in PIPELINE_ARMS:
        data = load_arm(ctx, arm)
        out = ctx.run_dir / f"utility_{arm}.jsonl"
        done = {r["sample_id"] for r in jsonl_read(out) if not r.get("error")}

        def one(sid: str, arm=arm, data=data) -> list[dict]:
            s = by_id[sid]
            s0 = answerer.answer(data[sid]["transmitted"])
            final = synth.synthesize(s.question, s.context, s0)
            return [{"sample_id": sid, "arm": arm, "s0_answers": [{"q": a.question, "a": a.answer} for a in s0], "answer": final}]

        run_parallel([sid for sid in data if sid in by_id and sid not in done], one, JsonlWriter(out), ctx.args.workers, f"utility:{arm}")

    # entity substitution: answer on substituted text, restore placeholders
    data = load_arm(ctx, "entity_sub")
    out = ctx.run_dir / "utility_entity_sub.jsonl"
    done = {r["sample_id"] for r in jsonl_read(out) if not r.get("error")}
    es = EntitySubstitutionBaseline(cloud_client=None)

    def one_sub(sid: str) -> list[dict]:
        r = data[sid]
        raw = answer(r["transmitted_question"], r["transmitted_context"])
        return [{"sample_id": sid, "arm": "entity_sub", "answer": es._restore(raw, r["substitution_map"])}]

    run_parallel([sid for sid in data if sid in by_id and sid not in done], one_sub, JsonlWriter(out), ctx.args.workers, "utility:entity_sub")

    # santext arms: answer on sanitized text (no restoration possible)
    for eps in ctx.args.santext_eps:
        arm = f"santext_e{eps:g}"
        data = load_arm(ctx, arm)
        out = ctx.run_dir / f"utility_{arm}.jsonl"
        done = {r["sample_id"] for r in jsonl_read(out) if not r.get("error")}

        def one_dp(sid: str, arm=arm, data=data) -> list[dict]:
            r = data[sid]
            return [{"sample_id": sid, "arm": arm, "answer": answer(r["transmitted_question"], r["transmitted_context"])}]

        run_parallel([sid for sid in data if sid in by_id and sid not in done], one_dp, JsonlWriter(out), ctx.args.workers, f"utility:{arm}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stage", default="all", help="comma list of stages or 'all'")
    p.add_argument("--arms", default="", help="comma list of arms for attacks/multiquery (default: all)")
    p.add_argument("--attacks", default="", help="comma list of attack names to restrict to")
    p.add_argument("--n-contracts", type=int, default=45)
    p.add_argument("--min-per-contract", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="use at most N questions per contract (smoke tests)")
    p.add_argument("--utility-limit", type=int, default=0, help="utility stage on a seeded subset of this size")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--n-proxy", type=int, default=10)
    p.add_argument("--n-selected", type=int, default=3)
    p.add_argument("--n-draws", type=int, default=3)
    p.add_argument("--dp-icl-k", type=int, default=8, help="ICL examples for the DP/substitution text reconstruction")
    p.add_argument("--santext-eps", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--santext-p", type=float, default=0.3)
    p.add_argument("--santext-sensitive-rank", type=int, default=20000)
    p.add_argument("--glove-vocab", type=int, default=50000)
    p.add_argument("--vllm-url", default="http://127.0.0.1:8020/v1")
    p.add_argument("--model", default="local/active-model")
    p.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    a = p.parse_args()
    a.attacks = [x for x in a.attacks.split(",") if x]
    a.arms = [x for x in a.arms.split(",") if x]
    return a


def main():
    args = parse_args()
    ctx = Ctx(args)
    stages = ALL_STAGES if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    t_start = time.time()
    for st in stages:
        print(f"\n=== stage {st} ({time.strftime('%H:%M:%S')}) ===", flush=True)
        if st == "sample":
            stage_sample(ctx)
        elif st == "retokenize":
            stage_retokenize(ctx)
        elif st == "pipeline":
            for v in (args.arms or PIPELINE_ARMS):
                if v in PIPELINE_ARMS:
                    stage_pipeline(ctx, v)
        elif st == "baselines":
            stage_baselines(ctx)
        elif st == "leak":
            stage_leak(ctx)
        elif st == "linkage":
            stage_linkage(ctx)
        elif st == "attacks":
            for arm in (args.arms or arm_names(ctx)):
                stage_attacks(ctx, arm)
        elif st == "multiquery":
            for arm in (args.arms or PIPELINE_ARMS):
                if arm in PIPELINE_ARMS:
                    stage_multiquery(ctx, arm)
        elif st == "utility":
            stage_utility(ctx)
        else:
            raise SystemExit(f"unknown stage {st}")
    print(f"\nALL STAGES DONE in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
