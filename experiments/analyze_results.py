"""
Load evaluation JSON results and produce summary tables.

Usage:
  python experiments/analyze_results.py results/phi3_20260605_120000.json
  python experiments/analyze_results.py results/          # aggregate all JSON files
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def mean_of(rows: list[dict | None], key: str) -> float | None:
    vals = [r[key] for r in rows if r is not None and r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def fmt(v: float | None, precision: int = 3) -> str:
    if v is None:
        return "  n/a  "
    return f"{v:.{precision}f}"


def print_quality_table(data: dict):
    qp = data.get("quality_pipeline", [])
    ql = data.get("quality_local_only", [])
    qe = data.get("quality_entity_substitution", [])

    print("\n── Answer Quality (vs. Oracle) ─────────────────────────────")
    print(f"  {'Metric':<28} {'Pipeline':>9} {'Local':>9} {'Entity-sub':>10}")
    print(f"  {'─'*28} {'─'*9} {'─'*9} {'─'*10}")

    for key, label in [
        ("bertscore_f1", "BERTScore F1"),
        ("semantic_sim", "Semantic similarity"),
        ("gold_span_overlap", "Gold span overlap"),
    ]:
        mp = mean_of(qp, key)
        ml = mean_of(ql, key)
        me = mean_of(qe, key)
        print(f"  {label:<28} {fmt(mp):>9} {fmt(ml):>9} {fmt(me):>10}")

    n_valid = sum(1 for r in qp if r is not None)
    print(f"  (n={n_valid} valid samples)")


def print_fidelity_table(data: dict):
    fid = data.get("fidelity", [])
    valid = [f for f in fid if f is not None]
    if not valid:
        print("\n── Bottleneck Fidelity ─── (no data)")
        return

    print("\n── Bottleneck Fidelity ─────────────────────────────────────")
    for key, label in [
        ("max_sim", "Max cosine sim (best proxy)"),
        ("mean_sim", "Mean cosine sim (all proxies)"),
        ("coverage", "Coverage (orig vs. centroid)"),
    ]:
        val = mean_of(valid, key)
        print(f"  {label:<30} {fmt(val)}")
    print(f"  (n={len(valid)} samples)")


def print_recovery_table(data: dict):
    rec = data.get("recovery", [])
    if not rec:
        print("\n── Sensitive Data Recovery ─── (skipped)")
        return

    valid = [r for r in rec if r is not None]
    svr_all = mean_of(valid, "mean_specific_value_rate")

    print("\n── Sensitive Data Recovery ─────────────────────────────────")
    print(f"  Mean specific value recovery rate:  {fmt(svr_all)}  (target: ≈0.00)")

    direct_svr = mean_of([r.get("direct") for r in valid if r.get("direct")], "specific_value_rate")
    expert_svr = mean_of([r.get("expert") for r in valid if r.get("expert")], "specific_value_rate")
    direct_dom = mean_of([r.get("direct") for r in valid if r.get("direct")], "domain_recovered")

    print(f"  Direct attack SVR:                  {fmt(direct_svr)}")
    print(f"  Expert attack SVR:                  {fmt(expert_svr)}")
    print(f"  Domain recovery (direct):           {fmt(direct_dom)}")
    print(f"  (n={len(valid)} samples)")


def print_task_sensitivity(data: dict):
    ts = data.get("task_sensitivity", {})
    if not ts:
        print("\n── Task Sensitivity ─── (no data)")
        return

    print("\n── Task Sensitivity ────────────────────────────────────────")
    print(f"  {'Task type':<28} {'n':>4} {'BERTScore':>9} {'Sem sim':>8} {'SVR':>7}")
    print(f"  {'─'*28} {'─'*4} {'─'*9} {'─'*8} {'─'*7}")

    for tt, g in sorted(ts.items()):
        print(
            f"  {tt:<28} {g['n']:>4} "
            f"{fmt(g.get('mean_bertscore_f1')):>9} "
            f"{fmt(g.get('mean_semantic_sim')):>8} "
            f"{fmt(g.get('mean_specific_value_recovery')):>7}"
        )


def print_run_header(data: dict):
    cfg = data.get("config", {})
    ts = data.get("timestamp", "?")
    n = cfg.get("num_samples", "?")
    model = cfg.get("local_model", "?")
    cloud = cfg.get("cloud_provider", "?")
    excl = cfg.get("use_vocab_exclusion", False)
    sel = cfg.get("question_selector", "?")
    t_total = data.get("total_pipeline_time_s")

    print(f"\n{'═'*60}")
    print(f"  Run: {data.get('run_id', '?')}")
    print(f"  Timestamp: {ts} | n={n} | model={model} | cloud={cloud}")
    print(f"  Vocab exclusion: {excl} | Selector: {sel}")
    if t_total:
        print(f"  Total pipeline time: {t_total:.1f}s ({t_total/max(n,1):.1f}s/sample)")
    print(f"{'═'*60}")


def analyze_file(path: Path):
    data = load_result(path)
    print_run_header(data)
    print_quality_table(data)
    print_fidelity_table(data)
    print_recovery_table(data)
    print_task_sensitivity(data)
    print()


def main():
    p = argparse.ArgumentParser(description="Analyze pipeline evaluation results")
    p.add_argument("path", help="Path to a results JSON file or directory of JSON files")
    args = p.parse_args()

    target = Path(args.path)
    if target.is_dir():
        files = sorted(target.glob("*.json"))
        if not files:
            print(f"No JSON files found in {target}")
            sys.exit(1)
        for f in files:
            analyze_file(f)
    elif target.is_file():
        analyze_file(target)
    else:
        print(f"Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
