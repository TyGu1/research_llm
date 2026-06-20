"""
Unit tests for evaluation modules — bottleneck fidelity, answer quality, task sensitivity.

Uses real SentenceTransformer (all-MiniLM-L6-v2) for embedding tests.
BERTScore is tested only with empty inputs (no model download needed).
"""
import numpy as np
import pytest

from src.evaluation.bottleneck_fidelity import BottleneckFidelityEvaluator, FidelityResult
from src.evaluation.answer_quality import AnswerQualityEvaluator, QualityResult
from src.evaluation.task_sensitivity import (
    TaskSensitivityEvaluator, TaskSensitivityResult, _confidence_interval_95,
)


@pytest.fixture(scope="module")
def embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── Bottleneck Fidelity ──────────────────────────────────────────────────────

def test_fidelity_returns_result(embed_model):
    evaluator = BottleneckFidelityEvaluator(embed_model=embed_model)
    result = evaluator.evaluate(
        "What non-compete restrictions apply?",
        ["What are typical non-compete terms?", "How long do restrictions last?"],
    )
    assert isinstance(result, FidelityResult)
    assert 0.0 <= result.max_sim <= 1.0
    assert 0.0 <= result.mean_sim <= 1.0
    assert 0.0 <= result.coverage <= 1.0


def test_fidelity_max_ge_mean(embed_model):
    evaluator = BottleneckFidelityEvaluator(embed_model=embed_model)
    result = evaluator.evaluate(
        "What are the termination conditions?",
        ["How can a contract be terminated?", "What is force majeure?", "What are renewal terms?"],
    )
    assert result.max_sim >= result.mean_sim


def test_fidelity_empty_proxies(embed_model):
    evaluator = BottleneckFidelityEvaluator(embed_model=embed_model)
    result = evaluator.evaluate("What are the terms?", [])
    assert result.max_sim == 0.0
    assert result.mean_sim == 0.0
    assert result.coverage == 0.0


def test_fidelity_identical_question(embed_model):
    evaluator = BottleneckFidelityEvaluator(embed_model=embed_model)
    q = "What are the non-compete restrictions?"
    result = evaluator.evaluate(q, [q])
    assert result.max_sim > 0.95


# ── Answer Quality (unit tests that avoid BERTScore model download) ──────────

def test_quality_semantic_sim_direct(embed_model):
    """Test _semantic_sim directly — avoids BERTScore."""
    evaluator = AnswerQualityEvaluator(embed_model=embed_model)
    sim = evaluator._semantic_sim(
        "The contract restricts competition for two years.",
        "Competition is restricted for a period of two years.",
    )
    assert sim > 0.5


def test_quality_semantic_sim_empty(embed_model):
    evaluator = AnswerQualityEvaluator(embed_model=embed_model)
    assert evaluator._semantic_sim("", "something") == 0.0
    assert evaluator._semantic_sim("something", "") == 0.0


def test_quality_gold_span_overlap_all_found(embed_model):
    evaluator = AnswerQualityEvaluator(embed_model=embed_model)
    overlap = evaluator._gold_span_overlap(
        "The restriction lasts two years in North America.",
        ["two years", "North America"],
    )
    assert overlap == 1.0


def test_quality_gold_span_overlap_none_found(embed_model):
    evaluator = AnswerQualityEvaluator(embed_model=embed_model)
    overlap = evaluator._gold_span_overlap(
        "The contract is governed by Delaware law.",
        ["two years", "North America"],
    )
    assert overlap == 0.0


def test_quality_gold_span_overlap_empty(embed_model):
    evaluator = AnswerQualityEvaluator(embed_model=embed_model)
    assert evaluator._gold_span_overlap("anything", []) == 0.0


def test_quality_empty_prediction_bertscore(embed_model):
    """Empty prediction should short-circuit BERTScore to 0.0."""
    evaluator = AnswerQualityEvaluator(embed_model=embed_model)
    result = evaluator.evaluate(
        prediction="",
        oracle_answer="Some answer.",
        gold_answers=["answer"],
    )
    assert result.semantic_sim_vs_oracle == 0.0
    assert result.bertscore_f1 == 0.0


def test_quality_length_ratio(embed_model):
    evaluator = AnswerQualityEvaluator(embed_model=embed_model)
    result = evaluator.evaluate(
        prediction="",
        oracle_answer="A much longer oracle answer with many words.",
        gold_answers=[],
    )
    assert result.answer_length_ratio == 0.0


# ── Task Sensitivity ─────────────────────────────────────────────────────────

def test_task_sensitivity_groups_by_type(sample_cuad_data):
    evaluator = TaskSensitivityEvaluator()
    quality_results = [
        QualityResult(bertscore_f1=0.8, semantic_sim_vs_oracle=0.7, gold_span_overlap=0.5, answer_length_ratio=1.0),
        QualityResult(bertscore_f1=0.6, semantic_sim_vs_oracle=0.5, gold_span_overlap=0.3, answer_length_ratio=0.9),
        QualityResult(bertscore_f1=0.7, semantic_sim_vs_oracle=0.6, gold_span_overlap=0.4, answer_length_ratio=1.1),
    ]
    svrs = [0.1, 0.2, 0.0]

    result = evaluator.evaluate(sample_cuad_data, quality_results, svrs)
    assert isinstance(result, TaskSensitivityResult)
    assert len(result.groups) > 0
    for group in result.groups.values():
        assert group.n > 0
        # CIs should be populated
        assert len(group.ci95_bertscore_f1) == 2


def test_task_sensitivity_empty_input():
    evaluator = TaskSensitivityEvaluator()
    result = evaluator.evaluate([], [], [])
    assert isinstance(result, TaskSensitivityResult)
    assert len(result.groups) == 0


# ── Confidence interval utility ──────────────────────────────────────────────

def test_ci95_single_value():
    lo, hi = _confidence_interval_95([0.5])
    assert lo == 0.5
    assert hi == 0.5


def test_ci95_bounds_contain_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lo, hi = _confidence_interval_95(values)
    mean = np.mean(values)
    assert lo <= mean <= hi


def test_ci95_narrow_for_constant():
    values = [0.5] * 20
    lo, hi = _confidence_interval_95(values)
    assert abs(hi - lo) < 0.01


def test_ci95_is_deterministic():
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    ci1 = _confidence_interval_95(values)
    ci2 = _confidence_interval_95(values)
    assert ci1 == ci2
