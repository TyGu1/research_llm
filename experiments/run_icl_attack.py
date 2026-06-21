"""Run ICL attack on existing pipeline results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PipelineConfig
from src.data.cuad_loader import CUADDataset, Split
from src.llm.cloud_client import CloudClient
from src.evaluation.sensitive_recovery import SensitiveRecoveryEvaluator


def main():
    cfg = PipelineConfig()

    # Load existing results
    results_path = Path("results/phi3_20260621_160315.json")
    with open(results_path) as f:
        existing = json.load(f)

    # Load same samples
    dataset = CUADDataset()
    samples = dataset.sample(n=48, seed=42, answerable_only=True, split=Split.DEV)

    # Build cloud client
    cloud_client = CloudClient(
        provider=cfg.cloud_provider,
        model=cfg.cloud_model,
        api_key=cfg.cloud_api_key,
        base_url=cfg.cloud_base_url,
        api_version=cfg.cloud_api_version,
    )

    embed_model = SentenceTransformer(cfg.similarity_model)

    # Build ICL examples from first 3 samples' proxy questions
    icl_examples = []
    for i, (sample, pr) in enumerate(zip(samples, existing["pipeline_results"])):
        if not pr.get("error") and pr.get("selected_questions") and len(icl_examples) < 3:
            icl_examples.append((sample.context[:200], pr["selected_questions"]))

    print(f"Built {len(icl_examples)} ICL examples")

    # Create evaluator WITH ICL examples
    evaluator = SensitiveRecoveryEvaluator(
        attacker_client=cloud_client,
        embed_model=embed_model,
        icl_examples=icl_examples,
    )

    # Run ICL attack on all samples (skip first 3 used as examples)
    from src.pipeline.base import PipelineResult, TopicDistribution, S0Answer

    icl_results = []
    for i, (sample, pr) in enumerate(zip(samples, existing["pipeline_results"])):
        if pr.get("error") or not pr.get("selected_questions"):
            icl_results.append(None)
            continue

        # Reconstruct minimal PipelineResult for the evaluator
        topics = TopicDistribution(topics=pr["topics"]) if pr.get("topics") else None
        result = PipelineResult(
            sample_id=pr["sample_id"],
            topics=topics,
            s0_text="",
            s0_text_excluded="",
            generated_questions=[],
            selected_questions=pr["selected_questions"],
            s0_answers=[],
            final_answer="",
            step_timings={},
            error="",
        )

        rr = evaluator.evaluate(sample, result)
        icl_data = None
        if rr.icl:
            icl_data = {
                "domain_recovered": rr.icl.domain_recovered,
                "situation_score": rr.icl.situation_score,
                "specific_value_rate": rr.icl.specific_value_rate,
            }
        icl_results.append({
            "sample_id": sample.id,
            "icl": icl_data,
        })
        svr = rr.icl.specific_value_rate if rr.icl else 0.0
        print(f"[{i+1}/48] SVR={svr:.3f} domain={rr.icl.domain_recovered if rr.icl else 'N/A'} — {sample.category}")

    # Summary
    svrs = [r["icl"]["specific_value_rate"] for r in icl_results if r and r.get("icl")]
    print(f"\n=== ICL Attack Results ===")
    print(f"Samples evaluated: {len(svrs)}")
    print(f"Mean SVR: {np.mean(svrs):.4f} ({np.mean(svrs)*100:.1f}%)")
    print(f"Max SVR: {np.max(svrs):.4f}")
    print(f"Samples with SVR > 0: {sum(1 for s in svrs if s > 0)}")

    # Save
    out_path = Path("results/icl_attack_results.json")
    with open(out_path, "w") as f:
        json.dump({"icl_results": icl_results, "mean_svr": float(np.mean(svrs))}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
