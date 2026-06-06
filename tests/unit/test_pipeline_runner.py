"""
Unit tests for PipelineRunner — verifies orchestration logic with all mocks.

The runner wires 6+ components together; these tests confirm that:
- The happy path produces a populated PipelineResult
- Exceptions in any step surface in result.error without crashing
- Vocab exclusion step is optional and toggleable
- Timings are captured for each step
"""
import pytest

from src.data.cuad_loader import CUADSample
from src.pipeline.base import (
    PipelineResult, TopicDistribution, S0Answer,
    TopicExtractor, S0TextGenerator, VocabExcluder,
    ProxyQuestionGenerator, QuestionSelector, CloudAnswerer, Synthesizer,
)
from src.pipeline.runner import PipelineRunner
from src.pipeline.topic_extraction import LLMTopicExtractor
from src.pipeline.text_generation import LLMTextGenerator
from src.pipeline.vocab_exclusion import RegexVocabExcluder
from src.pipeline.question_generation import LLMQuestionGenerator
from src.pipeline.question_selection import EmbeddingQuestionSelector
from src.pipeline.cloud_answering import StandardCloudAnswerer
from src.pipeline.synthesis import LLMSynthesizer


@pytest.fixture
def runner(mock_ollama) -> PipelineRunner:
    return PipelineRunner(
        topic_extractor=LLMTopicExtractor(client=mock_ollama),
        text_generator=LLMTextGenerator(client=mock_ollama),
        question_generator=LLMQuestionGenerator(client=mock_ollama, n=8),
        question_selector=EmbeddingQuestionSelector(ollama_client=mock_ollama),
        cloud_answerer=StandardCloudAnswerer(cloud_client=mock_ollama),
        synthesizer=LLMSynthesizer(client=mock_ollama),
        num_proxy_questions=8,
        num_selected_questions=3,
    )


@pytest.fixture
def runner_with_exclusion(mock_ollama) -> PipelineRunner:
    return PipelineRunner(
        topic_extractor=LLMTopicExtractor(client=mock_ollama),
        text_generator=LLMTextGenerator(client=mock_ollama),
        vocab_excluder=RegexVocabExcluder(),
        question_generator=LLMQuestionGenerator(client=mock_ollama, n=8),
        question_selector=EmbeddingQuestionSelector(ollama_client=mock_ollama),
        cloud_answerer=StandardCloudAnswerer(cloud_client=mock_ollama),
        synthesizer=LLMSynthesizer(client=mock_ollama),
        num_proxy_questions=8,
        num_selected_questions=3,
    )


# ── Happy path ────────────────────────────────────────────────────────────────

def test_run_returns_pipeline_result(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    assert isinstance(result, PipelineResult)


def test_run_no_error(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    # fix: wrong — PipelineResult.error defaults to "" (empty str sentinel), not None
    assert result.error == ""


def test_run_populates_topics(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    assert result.topics is not None
    assert len(result.topics.topics) >= 1


def test_run_populates_s0_text(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    assert result.s0_text is not None
    assert len(result.s0_text) > 0


def test_run_populates_selected_questions(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    assert result.selected_questions is not None
    assert 1 <= len(result.selected_questions) <= 3


def test_run_populates_final_answer(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    assert result.final_answer is not None
    assert len(result.final_answer) > 0


def test_run_captures_timings(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    assert "topic_extraction" in result.step_timings
    assert "text_generation" in result.step_timings
    assert "synthesis" in result.step_timings
    for v in result.step_timings.values():
        assert v >= 0.0


# ── Vocab exclusion ───────────────────────────────────────────────────────────

def test_run_without_exclusion_leaves_s0_text_excluded_empty(runner, sample_cuad_data):
    result = runner.run(sample_cuad_data[0])
    # fix: wrong — PipelineResult.s0_text_excluded defaults to "" (empty str sentinel), not None
    assert result.s0_text_excluded == ""


def test_run_with_exclusion_populates_s0_text_excluded(runner_with_exclusion, sample_cuad_data):
    result = runner_with_exclusion.run(sample_cuad_data[0])
    assert result.s0_text_excluded is not None


def test_run_with_exclusion_captures_vocab_timing(runner_with_exclusion, sample_cuad_data):
    result = runner_with_exclusion.run(sample_cuad_data[0])
    assert "vocab_exclusion" in result.step_timings


# ── Batch run ─────────────────────────────────────────────────────────────────

def test_run_batch_returns_all_results(runner, sample_cuad_data):
    results = runner.run_batch(sample_cuad_data, verbose=False)
    assert len(results) == len(sample_cuad_data)
    assert all(isinstance(r, PipelineResult) for r in results)


def test_run_batch_preserves_sample_ids(runner, sample_cuad_data):
    results = runner.run_batch(sample_cuad_data, verbose=False)
    ids = [r.sample_id for r in results]
    assert ids == [s.id for s in sample_cuad_data]


# ── Error isolation ───────────────────────────────────────────────────────────

class FailingExtractor(TopicExtractor):
    def extract(self, question: str, context: str) -> TopicDistribution:
        raise RuntimeError("Simulated extractor failure")


def test_run_captures_step_exception(mock_ollama, sample_cuad_data):
    broken_runner = PipelineRunner(
        topic_extractor=FailingExtractor(),
        text_generator=LLMTextGenerator(client=mock_ollama),
        question_generator=LLMQuestionGenerator(client=mock_ollama),
        question_selector=EmbeddingQuestionSelector(ollama_client=mock_ollama),
        cloud_answerer=StandardCloudAnswerer(cloud_client=mock_ollama),
        synthesizer=LLMSynthesizer(client=mock_ollama),
    )
    result = broken_runner.run(sample_cuad_data[0])
    assert result.error != ""
    assert "Simulated extractor failure" in result.error
