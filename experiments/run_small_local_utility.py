#!/usr/bin/env python
"""
Does the cloud add anything? Small-local-model utility check (paper setting: weak local, strong cloud).

On the same utility subset as run_adaptive_eval.py --stage utility, with a SMALL local model (Ollama):
  small_local_only   small model answers the question from the full window, no cloud
  small_pipeline     small model synthesizes from the window + the cloud's (Gemma) answers to the
                     3 transmitted proxy questions, reused from utility_<arm>.jsonl
Both are scored by analyze_adaptive.py against the same Gemma full-window oracle.

  .venv/bin/python experiments/run_small_local_utility.py --run-dir results/adaptive/main
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_adaptive_eval import JsonlWriter, jsonl_read, run_parallel, ANSWER_PROMPT, ANSWER_SYSTEM, WINDOW_CHARS
from src.llm.ollama_client import OllamaClient
from src.pipeline.base import S0Answer
from src.pipeline.synthesis import LLMSynthesizer
import json

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--model", default="qwen2.5:3b-instruct")
ap.add_argument("--ollama-url", default="http://localhost:11434")
ap.add_argument("--arms", default="plain", help="pipeline arms whose cloud answers to reuse")
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()
run = Path(a.run_dir)
meta = {r["id"]: r for r in json.loads((run / "samples.json").read_text())["samples"]}
oracle_ids = [r["sample_id"] for r in jsonl_read(run / "utility_oracle.jsonl") if not r.get("error")]
client = OllamaClient(a.ollama_url, a.model, timeout=900)
tag = a.model.split(":")[0].replace(".", "")

out = run / f"utility_small_local_only.jsonl"
done = {r["sample_id"] for r in jsonl_read(out) if not r.get("error")}
run_parallel([s for s in oracle_ids if s not in done],
             lambda sid: [{"sample_id": sid, "arm": "small_local_only", "model": a.model,
                           "answer": client.generate(ANSWER_PROMPT.format(question=meta[sid]["question"], context=meta[sid]["window"]), system=ANSWER_SYSTEM)}],
             JsonlWriter(out), a.workers, "small:local_only")

synth = LLMSynthesizer(client, max_context_chars=WINDOW_CHARS)
for arm in a.arms.split(","):
    cloud = {r["sample_id"]: r for r in jsonl_read(run / f"utility_{arm}.jsonl") if not r.get("error")}
    out = run / f"utility_small_pipeline_{arm}.jsonl"
    done = {r["sample_id"] for r in jsonl_read(out) if not r.get("error")}

    def one(sid, cloud=cloud, arm=arm):
        s0 = [S0Answer(question=x["q"], answer=x["a"]) for x in cloud[sid]["s0_answers"]]
        return [{"sample_id": sid, "arm": f"small_pipeline_{arm}", "model": a.model,
                 "answer": synth.synthesize(meta[sid]["question"], meta[sid]["window"], s0)}]

    run_parallel([s for s in oracle_ids if s in cloud and s not in done], one, JsonlWriter(out), a.workers, f"small:pipeline_{arm}")
print("SMALL DONE")
