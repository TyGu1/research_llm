#!/usr/bin/env python
"""
Fable calibration subset: re-run a few attacks with Claude Fable through the Claude
Code subscription (`claude -p`, OAuth — NO API key, rule R852) on a seeded subset of
an existing run, and compare with the Gemma attacker on the same (sample, attack) pairs.

  .venv/bin/python experiments/run_fable_calibration.py --run-dir results/adaptive/main --n 36
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.attacks.reconstruction import LLMAttacker
from src.attacks.scoring import Scorer
from src.evaluation.ci import cluster_bootstrap_mean

REFUSAL_RE = re.compile(r"\b(I can(?:'|’)?t|I cannot|I won(?:'|’)?t|I(?:'| a)m not able|not able to help|unable to (?:help|assist)|I(?:'| wi)ll not)\b", re.IGNORECASE)


class ClaudeCLIClient:
    """generate() through `claude -p`. Always passes --model explicitly (rule R014)."""

    def __init__(self, model: str, timeout: int = 420, cwd: str | None = None):
        self.model = model
        self.timeout = timeout
        self.cwd = cwd
        self.calls = 0
        self.failures = 0

    def generate(self, prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int | None = None) -> str:
        cmd = ["claude", "-p", "--model", self.model, "--output-format", "text", "--no-session-persistence",
               "--tools", ""]
        if system:
            cmd += ["--append-system-prompt", system]
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)       # allow nesting inside a Claude Code session
        last = ""
        for attempt in range(3):
            self.calls += 1
            try:
                r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout, cwd=self.cwd, env=env)
                out = r.stdout.strip()
                if r.returncode == 0 and out:
                    return out
                last = f"rc={r.returncode} stderr={r.stderr.strip()[:300]}"
            except subprocess.TimeoutExpired:
                last = "timeout"
            self.failures += 1
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"claude -p failed: {last}")


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", default="plain")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--attacks", default="direct,icl20,category_inversion")
    ap.add_argument("--max-calls", type=int, default=200, help="hard cap on claude -p invocations")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    attacks = a.attacks.split(",")

    data = json.loads((run_dir / "samples.json").read_text())
    meta = {r["id"]: r for r in data["samples"]}
    party = {t: set(v) for t, v in data["party_tokens"].items()}
    arm_rows = {r["sample_id"]: r for r in jsonl(run_dir / f"pipeline_{a.arm}.jsonl") if not r.get("error") and r.get("selected_questions")}
    gemma = {(r["sample_id"], r["attack"]): r for r in jsonl(run_dir / f"attacks_{a.arm}.jsonl") if not r.get("error")}

    # subset: round-robin over contracts, seeded, only samples that have Gemma results for every requested attack
    by_title = defaultdict(list)
    for sid in arm_rows:
        if all((sid, at) in gemma for at in attacks):
            by_title[meta[sid]["doc_title"]].append(sid)
    rng = random.Random(a.seed)
    titles = sorted(by_title)
    rng.shuffle(titles)
    for t in titles:
        rng.shuffle(by_title[t])
    subset: list[str] = []
    rnd = 0
    while len(subset) < a.n and any(len(by_title[t]) > rnd for t in titles):
        for t in titles:
            if len(by_title[t]) > rnd and len(subset) < a.n:
                subset.append(by_title[t][rnd])
        rnd += 1

    out_p = run_dir / f"fable_{a.arm}.jsonl"
    done = {(r["sample_id"], r["attack"]) for r in jsonl(out_p) if not r.get("error")}
    client = ClaudeCLIClient(a.model)
    attacker = LLMAttacker(client, max_tokens=700)
    from sentence_transformers import SentenceTransformer
    scorer = Scorer(SentenceTransformer("all-MiniLM-L6-v2", device="cpu"))

    # ICL examples: reuse exactly the Gemma run's construction (same seed, other contracts)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_adaptive_eval import _icl_examples
    from src.data.cuad_loader import CUADSample, Split

    class _Ctx:
        args = a
    by_id = {sid: CUADSample(id=sid, question=m["question"], raw_question="", context=m["window"], answers=m["answers_in_window"],
                             category=m["category"], task_type=m["task_type"], doc_title=m["doc_title"], is_impossible=False, split=Split.DEV)
             for sid, m in meta.items()}
    arm_data = {sid: dict(r, transmitted=r["selected_questions"]) for sid, r in arm_rows.items()}

    jobs = [(sid, at) for sid in subset for at in attacks if (sid, at) not in done]
    if len(jobs) > a.max_calls:
        print(f"capping {len(jobs)} jobs to --max-calls {a.max_calls}")
        jobs = jobs[: a.max_calls]
    print(f"subset {len(subset)} samples over {len({meta[s]['doc_title'] for s in subset})} contracts; {len(jobs)} calls to {a.model}")

    def one(job):
        sid, at = job
        s = by_id[sid]
        qs = arm_data[sid]["transmitted"]
        if at == "direct":
            ao = attacker.direct(qs)
        elif at == "expert":
            ao = attacker.expert(qs)
        elif at == "icl20":
            ao = attacker.icl(qs, _icl_examples(_Ctx, arm_data, by_id, sid, 20, False), "icl20")
        elif at == "icl3":
            ao = attacker.icl(qs, _icl_examples(_Ctx, arm_data, by_id, sid, 3, False), "icl3")
        elif at == "category_inversion":
            ao = attacker.category_inversion(qs)
        else:
            raise ValueError(at)
        sc = scorer.score(ao.reconstruction, s, party.get(s.doc_title, set()))
        extra = dict(ao.extra)
        if at == "category_inversion":
            cats = extra.get("categories_norm") or []
            extra["cat_top1"] = bool(cats) and cats[0] == s.category
            extra["cat_top3"] = s.category in cats[:3]
        extra["refusal"] = bool(REFUSAL_RE.search(ao.reconstruction[:400]))
        return {"sample_id": sid, "arm": a.arm, "attack": ao.attack, "observation": ao.observation, "model": a.model,
                "reconstruction": ao.reconstruction, "extra": extra, "score": sc.to_dict()}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(out_p, "a") as f:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                sid, at = futs[fut]
                row = {"sample_id": sid, "arm": a.arm, "attack": at, "model": a.model, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ({time.time()-t0:.0f}s, cli calls {client.calls}, failures {client.failures})", flush=True)

    # ── comparison on the same (sample, attack) pairs ────────────────────────
    fable = {(r["sample_id"], r["attack"]): r for r in jsonl(out_p) if not r.get("error")}
    titles_of = {sid: meta[sid]["doc_title"] for sid in meta}
    lines = [f"# Fable calibration — {run_dir.name}, arm {a.arm}", "",
             f"Same (sample, attack) pairs; attacker = Gemma-4-31B (local) vs {a.model} (`claude -p`). "
             f"n = {len(subset)} samples over {len({meta[s]['doc_title'] for s in subset})} contracts. 95% cluster-bootstrap CIs by contract.", "",
             "| Attack | n | Gemma SVR | Fable SVR | Gemma any-gold | Fable any-gold | Gemma party | Fable party | Gemma cat. named | Fable cat. named | Fable refusals |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for at in attacks:
        pairs = [(sid, at) for sid in subset if (sid, at) in fable and (sid, at) in gemma]
        if not pairs:
            continue
        def ci(rows, key):
            vals, cl = [], []
            for r in rows:
                v = r["score"][key] if key in r["score"] else r["extra"].get(key)
                if v is None:
                    continue
                vals.append(float(v)); cl.append(titles_of[r["sample_id"]])
            return cluster_bootstrap_mean(vals, cl, seed=a.seed)
        g = [gemma[p] for p in pairs]; fb = [fable[p] for p in pairs]
        row = {"n": len(pairs)}
        cells = [at, str(len(pairs))]
        for key in ["svr_exact", "any_gold", "party_hit", "category_label_hit"]:
            cg, cf = ci(g, key), ci(fb, key)
            row[key] = {"gemma": cg.__dict__, "fable": cf.__dict__}
            cells += [cg.fmt(), cf.fmt()]
        refusals = sum(1 for r in fb if r["extra"].get("refusal"))
        row["refusals"] = refusals
        cells.append(str(refusals))
        if at == "category_inversion":
            cg, cf = ci(g, "cat_top1"), ci(fb, "cat_top1")
            row["cat_top1"] = {"gemma": cg.__dict__, "fable": cf.__dict__}
            cells[0] = f"{at} (category top-1: Gemma {cg.fmt()}, Fable {cf.fmt()})"
        summary[at] = row
        lines.append("| " + " | ".join(cells) + " |")
    (run_dir / "fable_calibration.md").write_text("\n".join(lines) + "\n")
    (run_dir / "fable_calibration.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
