#!/usr/bin/env python
"""Copy control for the same-contract ICL attack: how much of its SVR is already in its examples?

  .venv/bin/python experiments/icl_same_copy_control.py --run-dir results/adaptive/main
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_adaptive_eval import _icl_examples
from src.attacks.scoring import svr_exact
from src.data.cuad_loader import CUADSample, Split
from src.evaluation.ci import cluster_bootstrap_mean

ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
run = Path(a.run_dir)
meta = {r["id"]: r for r in json.loads((run / "samples.json").read_text())["samples"]}
by_id = {k: CUADSample(id=k, question=m["question"], raw_question="", context=m["window"], answers=m["answers_in_window"], category=m["category"],
                       task_type=m["task_type"], doc_title=m["doc_title"], is_impossible=False, split=Split.DEV) for k, m in meta.items()}
class _C:
    args = a
out = {}
for arm in ["plain", "vocab"]:
    data = {r["sample_id"]: dict(r, transmitted=r["selected_questions"]) for r in map(json.loads, open(run / f"pipeline_{arm}.jsonl")) if r.get("selected_questions")}
    att = {r["sample_id"]: r for r in map(json.loads, open(run / f"attacks_{arm}.jsonl")) if r["attack"] == "icl_same" and not r.get("error")}
    copy, model, beyond, cl = [], [], [], []
    for sid, r in att.items():
        ex = _icl_examples(_C, data, by_id, sid, 5, True)
        c = svr_exact("\n".join(" ".join(e.gold) + " " + e.context_excerpt for e in ex), by_id[sid].answers)
        m = r["score"]["svr_exact"]
        copy.append(c); model.append(m); beyond.append(float(m > 0 and c == 0)); cl.append(meta[sid]["doc_title"])
    res = {k: cluster_bootstrap_mean(v, cl, seed=a.seed) for k, v in [("copy_floor", copy), ("model", model), ("beyond_copy", beyond)]}
    print(arm, {k: v.fmt() for k, v in res.items()})
    out[arm] = {k: v.__dict__ for k, v in res.items()}
(run / "icl_same_copy_control.json").write_text(json.dumps(out, indent=1))
