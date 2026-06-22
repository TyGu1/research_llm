"""
Benchmark latency of different step implementations.
Tests each step independently on 5 CUAD samples, measures wall time.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cuad_loader import CUADDataset, Split


def load_samples(n=5):
    """Load n CUAD samples for benchmarking."""
    dataset = CUADDataset()
    all_samples = dataset.load()
    answerable = [s for s in all_samples if s.answers and s.split == Split.DEV]
    import random
    random.Random(42).shuffle(answerable)
    samples = answerable[:n]
    print(f"Loaded {len(samples)} samples")
    return samples


def benchmark_step(name, fn, samples, runs=1):
    """Run fn on each sample, return mean time."""
    times = []
    results = []
    for s in samples:
        t0 = time.monotonic()
        for _ in range(runs):
            r = fn(s)
        elapsed = (time.monotonic() - t0) / runs
        times.append(elapsed)
        results.append(r)
    mean_t = sum(times) / len(times)
    print(f"  {name}: {mean_t*1000:.1f}ms mean ({min(times)*1000:.1f}–{max(times)*1000:.1f}ms range)")
    return mean_t, results


def bench_topic_extraction(samples):
    print("\n=== Step 1: Topic Extraction ===")

    # Variant A: Keyword-based (no model)
    from src.pipeline.fast_topic_extraction import KeywordTopicExtractor
    kw = KeywordTopicExtractor()
    t_kw, res_kw = benchmark_step("Keyword", lambda s: kw.extract(s.question, s.context), samples, runs=10)

    # Variant B: Zero-shot classifier (BART-large-mnli)
    try:
        from src.pipeline.fast_topic_extraction import ZeroShotTopicExtractor
        print("  Loading zero-shot model...")
        zs = ZeroShotTopicExtractor(device="cpu")
        # Warmup
        zs.extract(samples[0].question, samples[0].context)
        t_zs, res_zs = benchmark_step("ZeroShot", lambda s: zs.extract(s.question, s.context), samples)
    except Exception as e:
        print(f"  ZeroShot: FAILED ({e})")
        t_zs, res_zs = None, None

    # Show example output
    print(f"\n  Keyword topics (sample 0): {res_kw[0].topics}")
    if res_zs:
        print(f"  ZeroShot topics (sample 0): {res_zs[0].topics}")

    return {"keyword": t_kw, "zero_shot": t_zs}


def bench_text_generation(samples, topic_results):
    print("\n=== Step 2: S0 Text Generation ===")

    from src.pipeline.fast_text_generation import TemplateTextGenerator
    tg = TemplateTextGenerator()

    def gen(s):
        idx = samples.index(s)
        return tg.generate(topic_results[idx])

    t_tmpl, res = benchmark_step("Template", gen, samples, runs=100)

    # Show word count
    for i, r in enumerate(res[:2]):
        print(f"  Sample {i}: {len(r.split())} words")

    return {"template": t_tmpl}


def bench_question_generation(samples, text_results):
    print("\n=== Step 3: Proxy Question Generation ===")

    from src.pipeline.fast_question_generation import TemplateQuestionGenerator
    qg = TemplateQuestionGenerator(seed=42)

    def gen(s):
        idx = samples.index(s)
        return qg.generate(text_results[idx], n=10)

    t_tmpl, res = benchmark_step("Template", gen, samples, runs=100)

    # Show count
    for i, r in enumerate(res[:2]):
        print(f"  Sample {i}: {len(r)} questions generated")

    return {"template": t_tmpl}


def bench_question_selection(samples):
    print("\n=== Step 4: Question Selection ===")

    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Variant: all-MiniLM-L6-v2 via sentence-transformers (batch)
    print("  Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    dummy_questions = [
        "What are standard due diligence procedures?",
        "How do valuation methodologies differ?",
        "What regulatory approvals are required?",
        "What risk allocation mechanisms exist?",
        "How is post-merger integration structured?",
        "What representations and warranties are standard?",
        "What indemnification provisions are typical?",
        "How do competition authorities assess concentration?",
        "What remedies are available to regulators?",
        "What factors determine anti-competitive behavior?",
    ]

    def select(s):
        all_texts = [s.question] + dummy_questions
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        orig_emb = embeddings[0]
        cand_embs = embeddings[1:]
        scores = np.dot(cand_embs, orig_emb)
        top_k = np.argsort(scores)[-3:][::-1]
        return [dummy_questions[i] for i in top_k]

    # Warmup
    select(samples[0])
    t_st, _ = benchmark_step("SentenceTransformers (batch)", select, samples, runs=3)

    return {"sentence_transformers_batch": t_st}


def main():
    samples = load_samples(5)

    # Step 1
    topic_times = bench_topic_extraction(samples)

    # Use keyword topics for steps 2-3 (fastest)
    from src.pipeline.fast_topic_extraction import KeywordTopicExtractor
    kw = KeywordTopicExtractor()
    topic_results = [kw.extract(s.question, s.context) for s in samples]

    # Step 2
    from src.pipeline.fast_text_generation import TemplateTextGenerator
    tg = TemplateTextGenerator()
    text_results = [tg.generate(t) for t in topic_results]
    text_times = bench_text_generation(samples, topic_results)

    # Step 3
    q_times = bench_question_generation(samples, text_results)

    # Step 4
    sel_times = bench_question_selection(samples)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Best per-step latency")
    print("=" * 60)

    best = {
        "Step 1 (topic extraction)": min(v for v in topic_times.values() if v is not None),
        "Step 2 (S0 text generation)": min(text_times.values()),
        "Step 3 (proxy questions)": min(q_times.values()),
        "Step 4 (question selection)": min(sel_times.values()),
    }

    total = 0
    for step, t in best.items():
        print(f"  {step}: {t*1000:.1f}ms")
        total += t

    print(f"\n  TOTAL PRIVACY TAX: {total*1000:.1f}ms")
    print(f"  vs v0.1 baseline: 9,500ms")
    print(f"  Speedup: {9500 / (total * 1000):.0f}x")

    # Save results
    out = {
        "topic_extraction": topic_times,
        "text_generation": text_times,
        "question_generation": q_times,
        "question_selection": sel_times,
        "best_total_ms": total * 1000,
        "v01_baseline_ms": 9500,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/latency_benchmark.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to results/latency_benchmark.json")


if __name__ == "__main__":
    main()
