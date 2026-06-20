"""
Unit tests for the SensitiveRecoveryEvaluator — RecoveryResult dataclass,
absence fingerprint attack, and serialization completeness.
"""
import json
import pytest

from src.evaluation.sensitive_recovery import (
    AttackResult,
    RecoveryResult,
    SensitiveRecoveryEvaluator,
)
from src.pipeline.base import PipelineResult, TopicDistribution


# ── RecoveryResult ───────────────────────────────────────────────────────────

def test_recovery_result_includes_absence_in_mean():
    rr = RecoveryResult(
        sample_id="s1",
        direct=AttackResult(reconstruction="r", domain_recovered=True, situation_score=0.5, specific_value_rate=0.2),
        absence_fingerprint=AttackResult(reconstruction="r", domain_recovered=True, situation_score=0.6, specific_value_rate=0.8),
    )
    # mean of 0.2 and 0.8 = 0.5
    assert abs(rr.mean_specific_value_rate - 0.5) < 1e-6


def test_recovery_result_mean_without_absence():
    rr = RecoveryResult(
        sample_id="s1",
        direct=AttackResult(reconstruction="r", domain_recovered=True, situation_score=0.5, specific_value_rate=0.4),
    )
    assert abs(rr.mean_specific_value_rate - 0.4) < 1e-6


def test_recovery_result_absence_fingerprint_default_none():
    rr = RecoveryResult(sample_id="s1")
    assert rr.absence_fingerprint is None


# ── Absence fingerprint attack ───────────────────────────────────────────────

def test_absence_attack_skipped_without_vocab_exclusion(mock_ollama, sample_cuad_data):
    """Attack 4 should not run when s0_text_excluded is empty."""
    from sentence_transformers import SentenceTransformer
    embed = SentenceTransformer("all-MiniLM-L6-v2")

    evaluator = SensitiveRecoveryEvaluator(
        attacker_client=mock_ollama,
        embed_model=embed,
    )
    result = PipelineResult(
        sample_id="test-001",
        topics=TopicDistribution(topics={"contract_law": 0.5, "non_compete": 0.5}),
        s0_text="Some S0 text about agreements.",
        s0_text_excluded="",  # empty — no vocab exclusion
        selected_questions=["What are typical non-compete terms?"],
    )
    rr = evaluator.evaluate(sample_cuad_data[0], result)
    assert rr.absence_fingerprint is None


def test_absence_attack_runs_with_vocab_exclusion(mock_ollama, sample_cuad_data):
    """Attack 4 should run when s0_text_excluded is non-empty."""
    from sentence_transformers import SentenceTransformer
    embed = SentenceTransformer("all-MiniLM-L6-v2")

    evaluator = SensitiveRecoveryEvaluator(
        attacker_client=mock_ollama,
        embed_model=embed,
    )
    result = PipelineResult(
        sample_id="test-001",
        topics=TopicDistribution(topics={"contract_law": 0.5, "non_compete": 0.5}),
        s0_text="Some S0 text about agreements.",
        s0_text_excluded="Some text with words removed.",
        selected_questions=["What are typical non-compete terms?"],
    )
    rr = evaluator.evaluate(sample_cuad_data[0], result)
    assert rr.absence_fingerprint is not None
    assert isinstance(rr.absence_fingerprint, AttackResult)
    assert isinstance(rr.absence_fingerprint.specific_value_rate, float)


# ── Temperature wiring ───────────────────────────────────────────────────────

def test_topic_extractor_accepts_temperature(mock_ollama):
    from src.pipeline.topic_extraction import LLMTopicExtractor
    ext = LLMTopicExtractor(client=mock_ollama, temperature=0.7)
    assert ext._temperature == 0.7


def test_text_generator_accepts_temperature(mock_ollama):
    from src.pipeline.text_generation import LLMTextGenerator
    gen = LLMTextGenerator(client=mock_ollama, temperature=0.5)
    assert gen._temperature == 0.5


def test_question_generator_accepts_temperature(mock_ollama):
    from src.pipeline.question_generation import LLMQuestionGenerator
    gen = LLMQuestionGenerator(client=mock_ollama, temperature=0.3)
    assert gen._temperature == 0.3


def test_synthesizer_accepts_temperature(mock_ollama):
    from src.pipeline.synthesis import LLMSynthesizer
    syn = LLMSynthesizer(client=mock_ollama, temperature=0.9)
    assert syn._temperature == 0.9


def test_temperature_defaults_to_zero(mock_ollama):
    from src.pipeline.topic_extraction import LLMTopicExtractor
    from src.pipeline.text_generation import LLMTextGenerator
    from src.pipeline.question_generation import LLMQuestionGenerator
    from src.pipeline.synthesis import LLMSynthesizer

    assert LLMTopicExtractor(client=mock_ollama)._temperature == 0.0
    assert LLMTextGenerator(client=mock_ollama)._temperature == 0.0
    assert LLMQuestionGenerator(client=mock_ollama)._temperature == 0.0
    assert LLMSynthesizer(client=mock_ollama)._temperature == 0.0
