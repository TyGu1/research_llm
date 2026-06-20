"""
Integration tests: end-to-end pipeline on real samples with Ollama.

Runs the full 6-step pipeline on 2 hardcoded samples.
Validates structural correctness of PipelineResult, not answer content
(LLM outputs are non-deterministic even at temperature=0 across versions).

Skip if Ollama is not running.
"""
import pytest
import httpx

from src.pipeline.base import PipelineResult
from src.pipeline.runner import PipelineRunner
from src.pipeline.topic_extraction import LLMTopicExtractor
from src.pipeline.text_generation import LLMTextGenerator
from src.pipeline.vocab_exclusion import RegexVocabExcluder
from src.pipeline.question_generation import LLMQuestionGenerator
from src.pipeline.question_selection import EmbeddingQuestionSelector
from src.pipeline.cloud_answering import StandardCloudAnswerer
from src.pipeline.synthesis import LLMSynthesizer
from src.llm.ollama_client import OllamaClient
from src.llm.mock_client import MockOllamaClient


OLLAMA_URL = "http://localhost:11434"


def ollama_available() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.integration
skip_no_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running at localhost:11434",
)


@pytest.fixture
def live_runner():
    # fix: wrong — OllamaClient has no embed_model constructor arg; selector gets model via config
    client = OllamaClient(base_url=OLLAMA_URL, model="phi3")
    # Use MockOllamaClient as cloud answerer to avoid needing a real API key
    cloud = MockOllamaClient()
    return PipelineRunner(
        topic_extractor=LLMTopicExtractor(client=client, max_context_tokens=1500),
        text_generator=LLMTextGenerator(client=client),
        question_generator=LLMQuestionGenerator(client=client, n=8),
        question_selector=EmbeddingQuestionSelector(ollama_client=client),
        cloud_answerer=StandardCloudAnswerer(cloud_client=cloud),
        synthesizer=LLMSynthesizer(client=client),
        num_proxy_questions=8,
        num_selected_questions=3,
    )


@pytest.fixture
def live_runner_with_exclusion():
    # fix: wrong — OllamaClient has no embed_model constructor arg; selector gets model via config
    client = OllamaClient(base_url=OLLAMA_URL, model="phi3")
    cloud = MockOllamaClient()
    return PipelineRunner(
        topic_extractor=LLMTopicExtractor(client=client, max_context_tokens=1500),
        text_generator=LLMTextGenerator(client=client),
        vocab_excluder=RegexVocabExcluder(),
        question_generator=LLMQuestionGenerator(client=client, n=8),
        question_selector=EmbeddingQuestionSelector(ollama_client=client),
        cloud_answerer=StandardCloudAnswerer(cloud_client=cloud),
        synthesizer=LLMSynthesizer(client=client),
        num_proxy_questions=8,
        num_selected_questions=3,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@skip_no_ollama
def test_e2e_pipeline_no_error(live_runner, sample_cuad_data):
    result = live_runner.run(sample_cuad_data[0])
    assert result.error == "", f"Pipeline error: {result.error}"


@skip_no_ollama
def test_e2e_pipeline_all_steps_populated(live_runner, sample_cuad_data):
    result = live_runner.run(sample_cuad_data[0])
    assert result.topics is not None
    assert result.s0_text is not None and len(result.s0_text) > 50
    assert result.generated_questions is not None and len(result.generated_questions) >= 1
    assert result.selected_questions is not None and len(result.selected_questions) >= 1
    assert result.s0_answers is not None and len(result.s0_answers) >= 1
    assert result.final_answer is not None and len(result.final_answer) > 0


@skip_no_ollama
def test_e2e_pipeline_selected_within_generated(live_runner, sample_cuad_data):
    result = live_runner.run(sample_cuad_data[0])
    for q in result.selected_questions:
        assert q in result.generated_questions


@skip_no_ollama
def test_e2e_pipeline_timings_all_positive(live_runner, sample_cuad_data):
    result = live_runner.run(sample_cuad_data[0])
    required_steps = ["topic_extraction", "text_generation", "question_generation",
                      "question_selection", "cloud_answering", "synthesis"]
    for step in required_steps:
        assert step in result.step_timings, f"Missing timing for {step}"
        assert result.step_timings[step] > 0, f"Zero timing for {step}"


@skip_no_ollama
def test_e2e_batch_two_samples(live_runner, sample_cuad_data):
    results = live_runner.run_batch(sample_cuad_data[:2], verbose=False)
    assert len(results) == 2
    assert all(r.error == "" for r in results)


@skip_no_ollama
def test_e2e_with_vocab_exclusion(live_runner_with_exclusion, sample_cuad_data):
    result = live_runner_with_exclusion.run(sample_cuad_data[0])
    assert result.error == ""
    assert result.s0_text_excluded is not None
    assert len(result.s0_text_excluded) > 0
