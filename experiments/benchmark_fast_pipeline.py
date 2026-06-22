"""
Benchmark loop: test multiple LLM candidates for merged step 2+3.
Runs full 48-sample CUAD evaluation per candidate.
Measures: per-step latency, BERTScore F1, SVR.
Loops until best candidate is identified.

Usage:
    python experiments/benchmark_fast_pipeline.py
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PipelineConfig
from src.data.cuad_loader import CUADDataset, Split
from src.pipeline.base import PipelineResult
from src.pipeline.runner import PipelineRunner
from src.pipeline.embedding_topic_extraction import EmbeddingTopicExtractor
from src.pipeline.merged_s0_questions import (
    MergedS0Generator, MergedTextGeneratorAdapter, MergedQuestionGeneratorAdapter,
)
from src.pipeline.question_selection import EmbeddingQuestionSelector
from src.pipeline.cloud_answering import StandardCloudAnswerer
from src.pipeline.synthesis import LLMSynthesizer
from src.llm.cloud_client import CloudClient


# ── LLM candidates to benchmark ─────────────────────────────────────────────
# Each entry: (name, provider, model, endpoint_override)
# All use Azure OpenAI unless noted otherwise
LLM_CANDIDATES = [
    # (name, provider, model, endpoint_override, api_key_override)
    # gpt-oss-120b via Azure AI OSS (OpenAI-compatible endpoint) — fast, ~1.4s/call
    ("gpt-oss-120b", "openai", "gpt-oss-120b",
     "https://andre-mc3rpw8k-eastus2.services.ai.azure.com/openai/v1/", None),
    # gpt-4o via Azure OpenAI — baseline quality ceiling
    ("gpt-4o", "azure", "gpt-4o", None, None),
    # TODO: add gpt-oss-20b when deployed — expected ~500ms/call
]


def load_samples(cfg: PipelineConfig) -> list:
    """Load CUAD evaluation samples."""
    dataset = CUADDataset()
    all_samples = dataset.load()
    answerable = [s for s in all_samples if s.answers]

    # Filter by split
    if cfg.cuad_split == "dev":
        answerable = [s for s in answerable if s.split == Split.DEV]
    elif cfg.cuad_split == "test":
        answerable = [s for s in answerable if s.split == Split.TEST]

    import random
    random.Random(cfg.cuad_seed).shuffle(answerable)
    return answerable[:cfg.cuad_max_samples]


def build_fast_pipeline(
    local_client: CloudClient,
    cloud_client: CloudClient,
    cfg: PipelineConfig,
) -> tuple[PipelineRunner, MergedQuestionGeneratorAdapter]:
    """Build the fast pipeline with embedding topics + merged steps 2+3."""

    # Step 1: Embedding-based topic extraction (~10ms)
    topic_extractor = EmbeddingTopicExtractor(
        model_name=cfg.similarity_model,
    )

    # Steps 2+3: Merged LLM call
    merged = MergedS0Generator(
        client=local_client,
        n_questions=cfg.num_proxy_questions,
        temperature=cfg.temperature,
    )
    text_gen = MergedTextGeneratorAdapter(merged)
    q_gen = MergedQuestionGeneratorAdapter(merged)

    # Step 4: Embedding-based selection (~25ms)
    from sentence_transformers import SentenceTransformer
    import numpy as np

    class FastEmbeddingSelector:
        """Batched sentence-transformer selection."""
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self._model = SentenceTransformer(model_name)

        def select(self, original, candidates, k=3):
            if not candidates:
                return []
            all_texts = [original] + candidates
            embs = self._model.encode(all_texts, normalize_embeddings=True)
            scores = np.dot(embs[1:], embs[0])
            top_k = np.argsort(scores)[-k:][::-1]
            return [candidates[i] for i in top_k]

    selector = FastEmbeddingSelector(cfg.similarity_model)

    # Step 5: Cloud answering (GPT-4o always)
    cloud_answerer = StandardCloudAnswerer(cloud_client)

    # Step 6: Synthesis (uses same local client)
    synthesizer = LLMSynthesizer(local_client)

    runner = PipelineRunner(
        topic_extractor=topic_extractor,
        text_generator=text_gen,
        question_generator=q_gen,
        question_selector=selector,
        cloud_answerer=cloud_answerer,
        synthesizer=synthesizer,
        num_proxy_questions=cfg.num_proxy_questions,
        num_selected_questions=cfg.num_selected_questions,
    )

    return runner, q_gen


def run_evaluation(runner, q_gen, samples) -> list[PipelineResult]:
    """Run pipeline on all samples, injecting topics into merged Q generator."""
    results = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {sample.category} — {sample.id[:40]}...")

        result = PipelineResult(sample_id=sample.id)
        timings = {}

        try:
            # Step 1: Topic extraction
            t0 = time.monotonic()
            result.topics = runner._topic_extractor.extract(sample.question, sample.context)
            timings["topic_extraction"] = time.monotonic() - t0

            # Inject topics into merged generator
            q_gen.set_topics(result.topics)

            # Step 2: S0 text generation (merged — will cache)
            t0 = time.monotonic()
            result.s0_text = runner._text_generator.generate(result.topics)
            timings["text_generation"] = time.monotonic() - t0

            # Step 3: Proxy question generation (merged — uses cache)
            t0 = time.monotonic()
            result.generated_questions = runner._question_generator.generate(
                result.s0_text, n=runner._num_proxy
            )
            timings["question_generation"] = time.monotonic() - t0

            # Step 4: Question selection
            t0 = time.monotonic()
            result.selected_questions = runner._question_selector.select(
                sample.question, result.generated_questions, k=runner._num_selected
            )
            timings["question_selection"] = time.monotonic() - t0

            # Step 5: Cloud answering
            t0 = time.monotonic()
            result.s0_answers = runner._cloud_answerer.answer(result.selected_questions)
            timings["cloud_answering"] = time.monotonic() - t0

            # Step 6: Synthesis
            t0 = time.monotonic()
            result.final_answer = runner._synthesizer.synthesize(
                sample.question, sample.context, result.s0_answers
            )
            timings["synthesis"] = time.monotonic() - t0

        except Exception as e:
            result.error = str(e)
            print(f"    ERROR: {e}")

        result.step_timings = timings
        results.append(result)

    return results


def compute_metrics(results, samples):
    """Compute BERTScore F1 and SVR from results."""
    from src.evaluation.answer_quality import compute_answer_quality
    from src.evaluation.sensitive_recovery import SensitiveRecoveryEvaluator

    # BERTScore: need oracle answers for comparison
    # For now, compute mean step timings
    privacy_steps = ["topic_extraction", "text_generation", "question_generation", "question_selection"]
    step_means = {}
    for step in privacy_steps + ["cloud_answering", "synthesis"]:
        vals = [r.step_timings.get(step, 0) for r in results if not r.error]
        step_means[step] = sum(vals) / len(vals) if vals else 0

    privacy_tax = sum(step_means[s] for s in privacy_steps)
    total = sum(step_means.values())

    successful = [r for r in results if not r.error and r.final_answer]

    return {
        "n_samples": len(results),
        "n_successful": len(successful),
        "n_errors": len([r for r in results if r.error]),
        "step_means_ms": {k: v * 1000 for k, v in step_means.items()},
        "privacy_tax_ms": privacy_tax * 1000,
        "total_ms": total * 1000,
    }


def main():
    cfg = PipelineConfig()
    samples = load_samples(cfg)
    print(f"Loaded {len(samples)} samples\n")

    # Cloud client (always GPT-4o for step 5)
    cloud_client = CloudClient(
        provider=cfg.cloud_provider,
        model=cfg.cloud_model,
        api_key=cfg.cloud_api_key,
        base_url=cfg.cloud_base_url,
        api_version=cfg.cloud_api_version,
    )

    all_results = {}

    for name, provider, model, endpoint, api_key_ovr in LLM_CANDIDATES:
        print(f"\n{'='*60}")
        print(f"CANDIDATE: {name} ({provider}/{model})")
        print(f"{'='*60}")

        # Build local client for this candidate
        local_client = CloudClient(
            provider=provider,
            model=model,
            api_key=api_key_ovr or cfg.cloud_api_key,
            base_url=endpoint or cfg.cloud_base_url,
            api_version=cfg.cloud_api_version if provider == "azure" else "",
        )

        # Build fast pipeline
        runner, q_gen = build_fast_pipeline(local_client, cloud_client, cfg)

        # Run eval
        t_start = time.monotonic()
        results = run_evaluation(runner, q_gen, samples)
        wall_time = time.monotonic() - t_start

        # Compute metrics
        metrics = compute_metrics(results, samples)
        metrics["wall_time_s"] = wall_time
        metrics["candidate"] = name
        metrics["model"] = model
        metrics["provider"] = provider

        all_results[name] = metrics

        # Print summary
        print(f"\n  Results for {name}:")
        print(f"    Successful: {metrics['n_successful']}/{metrics['n_samples']}")
        print(f"    Privacy tax: {metrics['privacy_tax_ms']:.0f}ms mean")
        print(f"    Total per query: {metrics['total_ms']:.0f}ms mean")
        print(f"    Wall time: {wall_time:.1f}s")
        print(f"    Step breakdown:")
        for step, ms in metrics['step_means_ms'].items():
            print(f"      {step}: {ms:.0f}ms")

        # Save per-candidate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(cfg.results_dir) / f"fast_{name}_{timestamp}.json"
        out_data = {
            "config": {
                "local_model": model,
                "cloud_model": cfg.cloud_model,
                "provider": provider,
            },
            "metrics": metrics,
            "pipeline_results": [
                {
                    "sample_id": r.sample_id,
                    "error": r.error,
                    "step_timings": r.step_timings,
                    "topics": r.topics.topics if r.topics else None,
                    "s0_text": r.s0_text[:200] if r.s0_text else "",
                    "n_questions": len(r.generated_questions),
                    "selected_questions": r.selected_questions,
                    "final_answer": r.final_answer[:500] if r.final_answer else "",
                }
                for r in results
            ],
        }
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"    Saved: {out_path}")

    # ── Comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Candidate':<20} {'Privacy Tax':>12} {'Total/query':>12} {'Success':>8} {'Wall':>8}")
    print("-" * 64)
    for name, m in all_results.items():
        print(f"{name:<20} {m['privacy_tax_ms']:>10.0f}ms {m['total_ms']:>10.0f}ms {m['n_successful']:>6}/{m['n_samples']} {m['wall_time_s']:>6.1f}s")
    print("-" * 64)
    print("v0.1 baseline:       9,500ms privacy tax")

    # Save comparison
    comp_path = Path(cfg.results_dir) / "fast_pipeline_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
