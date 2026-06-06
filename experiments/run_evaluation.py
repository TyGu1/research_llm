"""
Main evaluation entry point for the topic-bottleneck pipeline.

Runs all four evaluation tests on CUAD samples and saves results to JSON.

Usage:
  python experiments/run_evaluation.py                          # defaults
  python experiments/run_evaluation.py --max-samples 5         # smoke test
  python experiments/run_evaluation.py --local-model gemma4:26b --max-samples 50
  python experiments/run_evaluation.py --no-recovery           # skip attack sims
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make sure src/ is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PipelineConfig
from src.data.cuad_loader import CUADDataset
from src.llm.ollama_client import OllamaClient
from src.llm.cloud_client import CloudClient
from src.llm.mock_client import MockOllamaClient
from src.pipeline.runner import PipelineRunner
from src.pipeline.topic_extraction import LLMTopicExtractor
from src.pipeline.text_generation import LLMTextGenerator
from src.pipeline.vocab_exclusion import RegexVocabExcluder
from src.pipeline.question_generation import LLMQuestionGenerator
from src.pipeline.question_selection import EmbeddingQuestionSelector, LLMQuestionSelector
from src.pipeline.cloud_answering import StandardCloudAnswerer
from src.pipeline.synthesis import LLMSynthesizer
from src.baselines.oracle import OracleBaseline
from src.baselines.local_only import LocalOnlyBaseline
from src.baselines.entity_substitution import EntitySubstitutionBaseline
from src.evaluation.sensitive_recovery import SensitiveRecoveryEvaluator
from src.evaluation.answer_quality import AnswerQualityEvaluator
from src.evaluation.bottleneck_fidelity import BottleneckFidelityEvaluator
from src.evaluation.task_sensitivity import TaskSensitivityEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate topic-bottleneck pipeline on CUAD")
    p.add_argument("--max-samples", type=int, default=50, help="Number of CUAD samples (default: 50)")
    p.add_argument("--local-model", default=None, help="Override local model (default: config)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-vocab-exclusion", action="store_true", help="Enable vocab exclusion step")
    p.add_argument("--question-selector", choices=["embedding", "llm"], default="embedding")
    p.add_argument("--no-recovery", action="store_true", help="Skip sensitive recovery attacks (faster)")
    p.add_argument("--cloud-provider", default=None, help="Cloud provider: mock | openai | anthropic")
    p.add_argument("--out-dir", default=None, help="Output directory (default: results/)")
    return p.parse_args()


def build_runner(cfg: PipelineConfig, local_client: OllamaClient, cloud_client) -> PipelineRunner:
    selector = (
        EmbeddingQuestionSelector(ollama_client=local_client, embed_model=cfg.embed_model)
        if cfg.question_selector == "embedding"
        else LLMQuestionSelector(client=local_client)
    )
    return PipelineRunner(
        topic_extractor=LLMTopicExtractor(client=local_client, max_context_tokens=cfg.local_model_max_tokens),
        text_generator=LLMTextGenerator(client=local_client, min_words=cfg.s0_text_min_words, max_words=cfg.s0_text_max_words),
        vocab_excluder=RegexVocabExcluder() if cfg.use_vocab_exclusion else None,
        question_generator=LLMQuestionGenerator(client=local_client, n=cfg.num_proxy_questions),
        question_selector=selector,
        cloud_answerer=StandardCloudAnswerer(cloud_client=cloud_client),
        synthesizer=LLMSynthesizer(client=local_client),
        num_proxy_questions=cfg.num_proxy_questions,
        num_selected_questions=cfg.num_selected_questions,
    )


def result_to_dict(r) -> dict:
    return {
        "sample_id": r.sample_id,
        "error": r.error,
        "topics": r.topics.topics if r.topics else None,
        "s0_text_length": len(r.s0_text) if r.s0_text else 0,
        "generated_questions_count": len(r.generated_questions) if r.generated_questions else 0,
        "selected_questions": r.selected_questions,
        "s0_answers_count": len(r.s0_answers) if r.s0_answers else 0,
        "final_answer_length": len(r.final_answer) if r.final_answer else 0,
        "step_timings": r.step_timings,
    }


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = PipelineConfig()
    if args.local_model:
        cfg.local_model = args.local_model
    if args.cloud_provider:
        cfg.cloud_provider = args.cloud_provider
    if args.use_vocab_exclusion:
        cfg.use_vocab_exclusion = True
    cfg.question_selector = args.question_selector
    cfg.cuad_max_samples = args.max_samples
    cfg.cuad_seed = args.seed

    out_dir = Path(args.out_dir or cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Clients ───────────────────────────────────────────────────────────────
    local_client = OllamaClient(
        base_url=cfg.ollama_base_url,
        model=cfg.local_model,
        embed_model=cfg.embed_model,
    )
    if cfg.cloud_provider == "mock":
        cloud_client = MockOllamaClient()
    else:
        cloud_client = CloudClient(
            provider=cfg.cloud_provider,
            model=cfg.cloud_model,
            api_key=cfg.cloud_api_key,
            base_url=cfg.cloud_base_url,
        )

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading CUAD (up to {cfg.cuad_max_samples} samples)...")
    dataset = CUADDataset()
    samples = dataset.sample(
        n=cfg.cuad_max_samples,
        seed=cfg.cuad_seed,
        answerable_only=cfg.cuad_answerable_only,
    )
    print(f"  Loaded {len(samples)} samples")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    runner = build_runner(cfg, local_client, cloud_client)

    print(f"\nRunning pipeline ({cfg.local_model}, vocab_exclusion={cfg.use_vocab_exclusion})...")
    t_pipeline_start = time.monotonic()
    pipeline_results = runner.run_batch(samples, verbose=True)
    t_pipeline = time.monotonic() - t_pipeline_start
    print(f"Pipeline done in {t_pipeline:.1f}s")

    errors = [r for r in pipeline_results if r.error]
    if errors:
        print(f"  Warnings: {len(errors)} samples had errors (see results JSON)")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\nRunning baselines...")
    oracle = OracleBaseline(client=cloud_client)
    local_only = LocalOnlyBaseline(client=local_client)
    entity_sub = EntitySubstitutionBaseline(local_client=local_client, cloud_client=cloud_client)

    oracle_answers = []
    local_answers = []
    entity_answers = []
    for s in samples:
        oracle_answers.append(oracle.answer(s))
        local_answers.append(local_only.answer(s))
        entity_answers.append(entity_sub.answer(s))
    print("  Baselines done")

    # ── Evaluation ────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(cfg.similarity_model)

    # Test 3: Bottleneck fidelity
    print("\nTest 3: Bottleneck fidelity...")
    fidelity_eval = BottleneckFidelityEvaluator(embed_model=embed_model)
    fidelity_results = []
    for sample, result in zip(samples, pipeline_results):
        if result.error or not result.selected_questions:
            fidelity_results.append(None)
            continue
        fr = fidelity_eval.evaluate(sample.question, result.selected_questions)
        fidelity_results.append({"max_sim": fr.max_sim, "mean_sim": fr.mean_sim, "coverage": fr.coverage})

    # Test 2: Answer quality
    print("Test 2: Answer quality...")
    quality_eval = AnswerQualityEvaluator(embed_model=embed_model, device="cpu")
    quality_results_pipeline = []
    quality_results_local = []
    quality_results_entity = []
    for sample, pr, oa, la, ea in zip(samples, pipeline_results, oracle_answers, local_answers, entity_answers):
        oracle_text = oa or ""
        qr = quality_eval.evaluate(pr.final_answer or "", oracle_text, sample.answers)
        quality_results_pipeline.append({
            "bertscore_f1": qr.bertscore_f1,
            "semantic_sim": qr.semantic_sim_vs_oracle,
            "gold_span_overlap": qr.gold_span_overlap,
            "length_ratio": qr.answer_length_ratio,
        })
        qr_l = quality_eval.evaluate(la or "", oracle_text, sample.answers)
        quality_results_local.append({
            "bertscore_f1": qr_l.bertscore_f1,
            "semantic_sim": qr_l.semantic_sim_vs_oracle,
            "gold_span_overlap": qr_l.gold_span_overlap,
        })
        qr_e = quality_eval.evaluate(ea or "", oracle_text, sample.answers)
        quality_results_entity.append({
            "bertscore_f1": qr_e.bertscore_f1,
            "semantic_sim": qr_e.semantic_sim_vs_oracle,
            "gold_span_overlap": qr_e.gold_span_overlap,
        })

    # Test 1: Sensitive data recovery
    recovery_results = []
    if not args.no_recovery:
        print("Test 1: Sensitive data recovery (3 attack variants)...")
        recovery_eval = SensitiveRecoveryEvaluator(
            attacker_client=cloud_client,
            embed_model=embed_model,
        )
        for sample, result in zip(samples, pipeline_results):
            if result.error or not result.selected_questions:
                recovery_results.append(None)
                continue
            rr = recovery_eval.evaluate(sample, result)
            recovery_results.append({
                "sample_id": rr.sample_id,
                "mean_specific_value_rate": rr.mean_specific_value_rate,
                "direct": {
                    "domain_recovered": rr.direct.domain_recovered if rr.direct else None,
                    "situation_score": rr.direct.situation_score if rr.direct else None,
                    "specific_value_rate": rr.direct.specific_value_rate if rr.direct else None,
                } if rr.direct else None,
                "expert": {
                    "domain_recovered": rr.expert.domain_recovered if rr.expert else None,
                    "specific_value_rate": rr.expert.specific_value_rate if rr.expert else None,
                } if rr.expert else None,
            })
    else:
        print("Test 1: Skipped (--no-recovery)")

    # Test 4: Task sensitivity
    print("Test 4: Task type sensitivity...")
    from src.evaluation.answer_quality import QualityResult
    quality_obj_list = []
    svr_list = []
    for i, (sample, result) in enumerate(zip(samples, pipeline_results)):
        if result.error:
            continue
        qr_dict = quality_results_pipeline[i]
        qr_obj = QualityResult(
            bertscore_f1=qr_dict["bertscore_f1"],
            semantic_sim_vs_oracle=qr_dict["semantic_sim"],
            gold_span_overlap=qr_dict["gold_span_overlap"],
            answer_length_ratio=qr_dict.get("length_ratio", 1.0),
        )
        quality_obj_list.append(qr_obj)
        svr = recovery_results[i]["mean_specific_value_rate"] if (recovery_results and recovery_results[i]) else 0.0
        svr_list.append(svr)

    valid_samples = [s for s, r in zip(samples, pipeline_results) if not r.error]
    task_eval = TaskSensitivityEvaluator()
    task_result = task_eval.evaluate(valid_samples, quality_obj_list, svr_list)
    task_groups = {
        tt: {
            "n": g.n,
            "mean_bertscore_f1": g.mean_bertscore_f1,
            "mean_semantic_sim": g.mean_semantic_sim,
            "mean_gold_span_overlap": g.mean_gold_span_overlap,
            "mean_specific_value_recovery": g.mean_specific_value_recovery,
        }
        for tt, g in task_result.groups.items()
    }

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.local_model.replace(':', '-')}_{timestamp}"
    out_path = out_dir / f"{run_id}.json"

    output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": {
            "local_model": cfg.local_model,
            "cloud_provider": cfg.cloud_provider,
            "num_samples": len(samples),
            "use_vocab_exclusion": cfg.use_vocab_exclusion,
            "question_selector": cfg.question_selector,
            "num_proxy_questions": cfg.num_proxy_questions,
            "num_selected_questions": cfg.num_selected_questions,
        },
        "pipeline_results": [result_to_dict(r) for r in pipeline_results],
        "fidelity": fidelity_results,
        "quality_pipeline": quality_results_pipeline,
        "quality_local_only": quality_results_local,
        "quality_entity_substitution": quality_results_entity,
        "recovery": recovery_results,
        "task_sensitivity": task_groups,
        "total_pipeline_time_s": t_pipeline,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("\n── Summary ──────────────────────────────────────────────────")
    if quality_results_pipeline:
        import statistics
        bs_scores = [q["bertscore_f1"] for q in quality_results_pipeline if q["bertscore_f1"] > 0]
        sim_scores = [q["semantic_sim"] for q in quality_results_pipeline if q["semantic_sim"] > 0]
        if bs_scores:
            print(f"  BERTScore F1 (pipeline vs oracle): {statistics.mean(bs_scores):.3f}")
        if sim_scores:
            print(f"  Semantic sim (pipeline vs oracle): {statistics.mean(sim_scores):.3f}")
    if recovery_results:
        svrs = [r["mean_specific_value_rate"] for r in recovery_results if r is not None]
        if svrs:
            print(f"  Specific value recovery rate:      {statistics.mean(svrs):.3f}  (target: near 0)")
    print("────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
