#!/usr/bin/env python
"""
Summarise an adaptive-attack run (experiments/run_adaptive_eval.py output) into
markdown + LaTeX tables. Every interval is a cluster bootstrap by CONTRACT
(src/evaluation/ci.py), 2000 resamples, seed 42.

  .venv/bin/python experiments/analyze_adaptive.py --run-dir results/adaptive/main
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.evaluation.ci import cluster_bootstrap_mean, BootstrapCI
from src.attacks.scoring import party_hit

ARM_LABEL = {
    "plain": "Topic bottleneck",
    "vocab": "Topic bottleneck + vocab exclusion",
    "entity_sub": "Entity substitution",
    "santext_e1": "SanText+ ($\\varepsilon$=1)",
    "santext_e3": "SanText+ ($\\varepsilon$=3)",
}
ATTACK_LABEL = {
    "direct": ("a", "Direct (paper prompt)", "Q"),
    "expert": ("a", "Expert persona (paper prompt)", "Q"),
    "icl3": ("a", "ICL, 3 solved examples", "Q"),
    "icl20": ("a", "ICL, 20 solved examples", "Q"),
    "icl_same": ("a", "ICL, examples from the same contract", "Q"),
    "category_inversion": ("b1", "Category inversion (pipeline-aware)", "Q+"),
    "absence_q": ("b2", "Absence fingerprint on Q", "Q+"),
    "absence_s0": ("b2", "Absence fingerprint on S0 text (upper bound)", "S0"),
    "sub_direct": ("a", "Placeholder re-identification", "SUB-text"),
    "dp_direct": ("a", "Direct reconstruction", "DP-text"),
    "dp_icl": ("a", "ICL reconstruction (Pang et al. recipe)", "DP/SUB-text"),
}
GROUP_LABEL = {"party": "Parties", "date": "Dates", "name": "Document name", "clause": "Clause text"}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def md_ci(c: BootstrapCI, pct: bool = True) -> str:
    return c.fmt(pct=pct) if c.n else "–"


def tex_ci(c: BootstrapCI, pct: bool = True) -> str:
    if not c.n:
        return "--"
    if pct:
        return f"{100*c.mean:.1f} [{100*c.lo:.1f}, {100*c.hi:.1f}]"
    return f"{c.mean:.3f} [{c.lo:.3f}, {c.hi:.3f}]"


class Analyzer:
    def __init__(self, run_dir: Path, n_boot: int, seed: int):
        self.run_dir = run_dir
        self.n_boot = n_boot
        self.seed = seed
        data = json.loads((run_dir / "samples.json").read_text())
        self.meta = {r["id"]: r for r in data["samples"]}
        self.config = data["config"]
        self.titles = {sid: r["doc_title"] for sid, r in self.meta.items()}
        self.tokens = {t: set(v) for t, v in data["party_tokens"].items()}
        self.party_known = {t for t, toks in self.tokens.items() if toks}
        self.arms = [a for a in ["plain", "vocab"] if (run_dir / f"pipeline_{a}.jsonl").exists()]
        self.arms += sorted(p.stem[4:] for p in run_dir.glob("arm_*.jsonl"))
        self.md: list[str] = []
        self.tex: list[str] = []
        self.summary: dict = {"config": self.config, "n_samples": len(self.meta),
                              "n_contracts": len(set(self.titles.values())), "arms": self.arms}

    # ── helpers ──────────────────────────────────────────────────────────────
    def ci(self, rows: list[dict], value, cluster_key="sample_id") -> BootstrapCI:
        vals, clus = [], []
        for r in rows:
            v = value(r)
            if v is None:
                continue
            vals.append(float(v))
            clus.append(self.titles[r[cluster_key]] if cluster_key == "sample_id" else r[cluster_key])
        return cluster_bootstrap_mean(vals, clus, n_boot=self.n_boot, seed=self.seed)

    def table(self, title: str, headers: list[str], rows: list[list[str]], tex_rows: list[list[str]] | None, label: str, caption: str):
        self.md.append(f"\n### {title}\n")
        self.md.append("| " + " | ".join(headers) + " |")
        self.md.append("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            self.md.append("| " + " | ".join(r) + " |")
        tex_rows = tex_rows or rows
        cols = "l" + "r" * (len(headers) - 1)
        self.tex.append(f"\\begin{{table}}[t]\n\\centering\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule")
        self.tex.append(" & ".join(headers) + " \\\\\n\\midrule")
        for r in tex_rows:
            self.tex.append(" & ".join(r) + " \\\\")
        self.tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ── sections ─────────────────────────────────────────────────────────────
    def sample_stats(self):
        by_type = defaultdict(int)
        by_group = defaultdict(int)
        from src.attacks.scoring import category_group
        for r in self.meta.values():
            by_type[r["task_type"]] += 1
            by_group[category_group(r["category"])] += 1
        n_win = sum(1 for r in self.meta.values() if r["window_contains_span"])
        self.summary["task_types"] = dict(by_type)
        self.summary["groups"] = dict(by_group)
        self.summary["window_contains_span"] = n_win
        self.md.append(f"# Adaptive-attack evaluation — {self.run_dir.name}\n")
        self.md.append(f"- samples: **{len(self.meta)}** questions from **{self.summary['n_contracts']}** CUAD dev contracts "
                       f"(seed {self.config['seed']}, min {self.config['min_per_contract']} answerable questions per contract)")
        self.md.append(f"- task types: {dict(by_type)}; category groups: {dict(by_group)}")
        self.md.append(f"- the sensitive document per query is a {self.config['window_chars']}-char window of the contract around the first gold span "
                       f"(window contains a gold span for {n_win}/{len(self.meta)})")
        self.md.append(f"- all intervals: cluster bootstrap by contract, {self.n_boot} resamples, 95%")

    def leak(self):
        headers = ["Arm", "n", "Gold span in transmitted text", "Party name", "Any date", "Any amount"]
        rows, tex_rows = [], []
        self.summary["leak"] = {}
        for arm in self.arms:
            data = jsonl(self.run_dir / f"leak_{arm}.jsonl")
            if not data:
                continue
            g = self.ci(data, lambda r: r["gold_hit"])
            p = self.ci(data, lambda r: r["party_hit"])
            d = self.ci(data, lambda r: r["date_hit"])
            a = self.ci(data, lambda r: r["amount_hit"])
            self.summary["leak"][arm] = {"n": g.n, "gold_hit": g.__dict__, "party_hit": p.__dict__, "date_hit": d.__dict__, "amount_hit": a.__dict__}
            rows.append([ARM_LABEL.get(arm, arm), str(g.n), md_ci(g), md_ci(p), md_ci(d), md_ci(a)])
            tex_rows.append([ARM_LABEL.get(arm, arm), str(g.n), tex_ci(g), tex_ci(p), tex_ci(d), tex_ci(a)])
        self.table("Rung 0 — leak scan of the transmitted content (no attacker model)", headers, rows, tex_rows,
                   "tab:leak", "Rung 0: does the transmitted content itself contain the sensitive values? Percent of queries, 95\\% cluster-bootstrap CI by contract.")

    def attacks(self):
        headers = ["Arm", "Rung", "Attack", "Obs.", "n", "SVR", "Any gold span", "Party name", "Category named", "Situation"]
        rows, tex_rows = [], []
        self.summary["attacks"] = {}
        self.best: dict[str, tuple[str, list[dict]]] = {}
        for arm in self.arms:
            data, seen = [], set()
            for r in jsonl(self.run_dir / f"attacks_{arm}.jsonl"):   # parallel writers may duplicate a pair
                if r.get("error") or (r["sample_id"], r["attack"]) in seen:
                    continue
                seen.add((r["sample_id"], r["attack"]))
                data.append(r)
            for r in data:   # party tokens may have been recomputed since the attack ran
                r["score"]["party_hit"] = party_hit(r["reconstruction"], self.tokens.get(self.titles[r["sample_id"]], set()))
            errs = [r for r in jsonl(self.run_dir / f"attacks_{arm}.jsonl") if r.get("error")]
            by_attack = defaultdict(list)
            for r in data:
                by_attack[r["attack"]].append(r)
            self.summary["attacks"][arm] = {"errors": len(errs)}
            best_svr, best_name = -1.0, None
            for name in ATTACK_LABEL:
                rs = by_attack.get(name)
                if not rs:
                    continue
                svr = self.ci(rs, lambda r: r["score"]["svr_exact"])
                anyg = self.ci(rs, lambda r: r["score"]["any_gold"])
                party = self.ci([r for r in rs if r["score"]["party_hit"] is not None], lambda r: r["score"]["party_hit"])
                cat = self.ci(rs, lambda r: r["score"]["category_label_hit"])
                sit = self.ci(rs, lambda r: r["score"]["situation_score"])
                rung, label, obs = ATTACK_LABEL[name]
                entry = {"n": svr.n, "svr": svr.__dict__, "any_gold": anyg.__dict__, "party_hit": party.__dict__,
                         "category_named": cat.__dict__, "situation": sit.__dict__}
                if name == "category_inversion":
                    t1 = self.ci(rs, lambda r: r["extra"].get("cat_top1", False))
                    t3 = self.ci(rs, lambda r: r["extra"].get("cat_top3", False))
                    entry["cat_top1"] = t1.__dict__
                    entry["cat_top3"] = t3.__dict__
                    label += f"; category top-1 {md_ci(t1)}, top-3 {md_ci(t3)}"
                if name.startswith("absence"):
                    rec = self.ci(rs, lambda r: r["extra"].get("excl_recall"))
                    entry["excl_recall"] = rec.__dict__
                    label += f"; excluded-word recall {md_ci(rec)}"
                self.summary["attacks"][arm][name] = entry
                if svr.mean > best_svr and obs != "S0":
                    best_svr, best_name = svr.mean, name
                rows.append([ARM_LABEL.get(arm, arm), rung, label, obs, str(svr.n), md_ci(svr), md_ci(anyg), md_ci(party), md_ci(cat), md_ci(sit, pct=False)])
                tex_rows.append([ARM_LABEL.get(arm, arm), rung, ATTACK_LABEL[name][1], obs, str(svr.n), tex_ci(svr), tex_ci(anyg), tex_ci(party), tex_ci(cat), tex_ci(sit, pct=False)])
            if best_name:
                self.best[arm] = (best_name, by_attack[best_name])
        self.table("Rungs a, b — reconstruction attacks (per query)", headers, rows, tex_rows, "tab:attacks",
                   "Reconstruction attacks. SVR = specific value recovery (paper criterion: first 50 characters of a gold span appear verbatim in the attacker's output), "
                   "\\emph{any gold span} = share of queries with SVR $>0$, \\emph{party name} = a distinctive party token appears, \\emph{category named} = the CUAD category "
                   "is named, \\emph{situation} = MiniLM cosine between reconstruction and contract. Obs.\\ = observation set (Q transmitted questions; Q+ with pipeline "
                   "knowledge; S0 the never-transmitted local text, upper bound). 95\\% cluster-bootstrap CIs by contract.")

    def typed(self):
        from src.attacks.scoring import category_group
        headers = ["Arm (strongest realistic attack)"] + [GROUP_LABEL[g] for g in ["party", "date", "name", "clause"]]
        rows, tex_rows = [], []
        self.summary["typed"] = {}
        for arm, (name, rs) in self.best.items():
            by_group = defaultdict(list)
            for r in rs:
                by_group[category_group(self.meta[r["sample_id"]]["category"])].append(r)
            cells, tcells = [], []
            self.summary["typed"][arm] = {"attack": name}
            for g in ["party", "date", "name", "clause"]:
                c = self.ci(by_group.get(g, []), lambda r: r["score"]["svr_exact"])
                self.summary["typed"][arm][g] = c.__dict__
                cells.append(f"{md_ci(c)} (n={c.n})")
                tcells.append(f"{tex_ci(c)} ({c.n})")
            rows.append([f"{ARM_LABEL.get(arm, arm)} — {ATTACK_LABEL[name][1]}"] + cells)
            tex_rows.append([f"{ARM_LABEL.get(arm, arm)} ({ATTACK_LABEL[name][1]})"] + tcells)
        self.table("SVR by value type (strongest realistic attack per arm)", headers, rows, tex_rows, "tab:typed",
                   "SVR of the strongest realistic attack per arm, split by the type of the sensitive value (n in parentheses).")

    def linkage(self):
        headers = ["Arm", "Category top-1", "Category top-5", "Contract top-1", "Contract top-5", "Contract top-1, k=3", "k=5", "k=all"]
        rows, tex_rows = [], []
        self.summary["linkage"] = {}
        chance = None
        for arm in [a for a in self.arms if a in ("plain", "vocab")]:
            p = self.run_dir / f"linkage_{arm}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            chance = (d["n_categories"], d["n_contracts"])
            c1 = self.ci(d["category_rows"], lambda r: r["top1"])
            c5 = self.ci(d["category_rows"], lambda r: r["top5"])
            k1 = self.ci(d["contract_rows"], lambda r: r["top1"])
            k5 = self.ci(d["contract_rows"], lambda r: r["top5"])
            multi = defaultdict(list)
            for r in d["multi_rows"]:
                multi[r["k"]].append(r)
            # k=all is the draw whose k equals the contract's query count: mark by sample_ids == all
            ks = sorted(k for k in multi if k in (3, 5))
            mk = {}
            for k in (3, 5):
                rs = multi.get(k, [])
                mk[k] = self.ci(rs, lambda r: r["top1"], cluster_key="doc_title") if rs else cluster_bootstrap_mean([], [])
            # 'all': rows produced by k="all" are those with draw index 0 and k == number of queries of the contract; identify via max k per contract
            per_title = defaultdict(list)
            for r in d["multi_rows"]:
                per_title[r["doc_title"]].append(r)
            all_rows = [max(rs, key=lambda r: r["k"]) for rs in per_title.values()]
            mk["all"] = self.ci(all_rows, lambda r: r["top1"], cluster_key="doc_title")
            self.summary["linkage"][arm] = {"category_top1": c1.__dict__, "category_top5": c5.__dict__, "contract_top1": k1.__dict__,
                                            "contract_top5": k5.__dict__, "multi_top1": {str(k): v.__dict__ for k, v in mk.items()},
                                            "n_categories": d["n_categories"], "n_contracts": d["n_contracts"]}
            rows.append([ARM_LABEL.get(arm, arm), md_ci(c1), md_ci(c5), md_ci(k1), md_ci(k5), md_ci(mk[3]), md_ci(mk[5]), md_ci(mk["all"])])
            tex_rows.append([ARM_LABEL.get(arm, arm), tex_ci(c1), tex_ci(c5), tex_ci(k1), tex_ci(k5), tex_ci(mk[3]), tex_ci(mk[5]), tex_ci(mk["all"])])
        if chance:
            c1, c5 = 100 / chance[0], min(100.0, 500 / chance[0])
            k1, k5 = 100 / chance[1], min(100.0, 500 / chance[1])
            rows.append(["chance", f"{c1:.1f}%", f"{c5:.1f}%", f"{k1:.1f}%", f"{k5:.1f}%", f"{k1:.1f}%", f"{k1:.1f}%", f"{k1:.1f}%"])
            tex_rows.append(["chance", f"{c1:.1f}", f"{c5:.1f}", f"{k1:.1f}", f"{k5:.1f}", f"{k1:.1f}", f"{k1:.1f}", f"{k1:.1f}"])
        self.table("Rungs b3, c — embedding linkage (no attacker LLM)", headers, rows, tex_rows, "tab:linkage",
                   "Linkage by embedding similarity of the transmitted questions against the 41 CUAD category questions and against the full text of the sampled contracts; "
                   "k = number of queries from the same contract pooled by the attacker (rung c). Percent, 95\\% cluster-bootstrap CIs by contract.")

    def multiquery(self):
        headers = ["Arm", "k", "n contracts", "Any gold span of the contract", "SVR over the contract's spans", "Party name"]
        rows, tex_rows = [], []
        self.summary["multiquery"] = {}
        for arm in [a for a in self.arms if a in ("plain", "vocab")]:
            data, seen = [], set()
            for r in jsonl(self.run_dir / f"multiquery_{arm}.jsonl"):
                if r.get("error") or (r["doc_title"], r["k"]) in seen:
                    continue
                seen.add((r["doc_title"], r["k"]))
                data.append(r)
            if not data:
                continue
            for r in data:
                r["party_hit"] = party_hit(r["reconstruction"], self.tokens.get(r["doc_title"], set()))
            by_k = defaultdict(list)
            per_title_max = defaultdict(int)
            for r in data:
                per_title_max[r["doc_title"]] = max(per_title_max[r["doc_title"]], r["k"])
            for r in data:
                key = "all" if r["k"] == per_title_max[r["doc_title"]] and r["k"] not in (3, 5) else r["k"]
                by_k[key].append(r)
                if r["k"] == per_title_max[r["doc_title"]] and r["k"] in (3, 5):
                    by_k["all"].append(r)
            self.summary["multiquery"][arm] = {}
            for k in [3, 5, "all"]:
                rs = by_k.get(k, [])
                if not rs:
                    continue
                anyg = self.ci(rs, lambda r: r["any_gold"], cluster_key="doc_title")
                svr = self.ci(rs, lambda r: r["svr_union"], cluster_key="doc_title")
                party = self.ci([r for r in rs if r["party_hit"] is not None], lambda r: r["party_hit"], cluster_key="doc_title")
                self.summary["multiquery"][arm][str(k)] = {"n": anyg.n, "any_gold": anyg.__dict__, "svr": svr.__dict__, "party_hit": party.__dict__}
                rows.append([ARM_LABEL.get(arm, arm), str(k), str(anyg.n), md_ci(anyg), md_ci(svr), md_ci(party)])
                tex_rows.append([ARM_LABEL.get(arm, arm), str(k), str(anyg.n), tex_ci(anyg), tex_ci(svr), tex_ci(party)])
        self.table("Rung c — multi-query aggregation with the LLM attacker", headers, rows, tex_rows, "tab:multiquery",
                   "Rung c: the attacker pools the transmitted questions of $k$ queries from the same contract and reconstructs the contract. "
                   "One draw per contract and $k$; intervals are bootstraps over contracts.")

    def utility(self, skip: bool):
        oracle = {r["sample_id"]: r["answer"] for r in jsonl(self.run_dir / "utility_oracle.jsonl") if not r.get("error")}
        if not oracle:
            return
        cache_p = self.run_dir / "utility_scores.json"
        cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
        scorer = None
        headers = ["Arm", "n", "BERTScore F1 vs oracle", "Semantic similarity", "Gold-span overlap"]
        rows, tex_rows = [], []
        self.summary["utility"] = {}
        for arm in self.arms:
            data = [r for r in jsonl(self.run_dir / f"utility_{arm}.jsonl") if not r.get("error") and r["sample_id"] in oracle]
            if not data:
                continue
            todo = [r for r in data if f"{arm}:{r['sample_id']}" not in cache]
            if todo and not skip:
                if scorer is None:
                    from sentence_transformers import SentenceTransformer
                    from src.evaluation.utility import BatchedQualityScorer
                    scorer = BatchedQualityScorer(SentenceTransformer("all-MiniLM-L6-v2", device="cpu"))
                scored = scorer.score([r["answer"] for r in todo], [oracle[r["sample_id"]] for r in todo],
                                      [self.meta[r["sample_id"]]["answers_in_window"] for r in todo])
                for r, s in zip(todo, scored):
                    cache[f"{arm}:{r['sample_id']}"] = s
                cache_p.write_text(json.dumps(cache))
            rs = [dict(cache[f"{arm}:{r['sample_id']}"], sample_id=r["sample_id"]) for r in data if f"{arm}:{r['sample_id']}" in cache]
            if not rs:
                continue
            b = self.ci(rs, lambda r: r["bertscore_f1"])
            s = self.ci(rs, lambda r: r["semantic_sim"])
            g = self.ci(rs, lambda r: r["gold_span_overlap"])
            self.summary["utility"][arm] = {"n": b.n, "bertscore_f1": b.__dict__, "semantic_sim": s.__dict__, "gold_span_overlap": g.__dict__}
            rows.append([ARM_LABEL.get(arm, arm), str(b.n), md_ci(b, pct=False), md_ci(s, pct=False), md_ci(g)])
            tex_rows.append([ARM_LABEL.get(arm, arm), str(b.n), tex_ci(b, pct=False), tex_ci(s, pct=False), tex_ci(g)])
        self.table("Utility — answer quality against the oracle (same model answers the plain question with the full window)", headers, rows, tex_rows, "tab:utility",
                   "Utility of each arm: final answer vs.\\ the oracle answer (BERTScore F1 with roberta-large; MiniLM cosine) and share of gold spans reproduced. 95\\% cluster-bootstrap CIs by contract.")

    def write(self):
        (self.run_dir / "summary.md").write_text("\n".join(self.md) + "\n")
        (self.run_dir / "tables.tex").write_text("\n".join(self.tex) + "\n")
        (self.run_dir / "summary.json").write_text(json.dumps(self.summary, indent=1, default=str))
        print("\n".join(self.md))
        print(f"\nwrote {self.run_dir/'summary.md'}, {self.run_dir/'tables.tex'}, {self.run_dir/'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-utility-scoring", action="store_true", help="only use cached utility scores")
    a = ap.parse_args()
    an = Analyzer(Path(a.run_dir), a.n_boot, a.seed)
    an.sample_stats()
    an.leak()
    an.attacks()
    an.typed()
    an.linkage()
    an.multiquery()
    an.utility(a.skip_utility_scoring)
    an.write()


if __name__ == "__main__":
    main()
