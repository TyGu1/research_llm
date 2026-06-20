"""
Unit tests for PipelineRunner checkpoint saving.
"""
import json
from pathlib import Path

from src.pipeline.runner import PipelineRunner


def test_checkpoint_saves_progress(mock_ollama, sample_cuad_data, tmp_path):
    from src.pipeline.topic_extraction import LLMTopicExtractor
    from src.pipeline.text_generation import LLMTextGenerator
    from src.pipeline.question_generation import LLMQuestionGenerator
    from src.pipeline.question_selection import EmbeddingQuestionSelector
    from src.pipeline.cloud_answering import StandardCloudAnswerer
    from src.pipeline.synthesis import LLMSynthesizer

    runner = PipelineRunner(
        topic_extractor=LLMTopicExtractor(client=mock_ollama),
        text_generator=LLMTextGenerator(client=mock_ollama),
        question_generator=LLMQuestionGenerator(client=mock_ollama),
        question_selector=EmbeddingQuestionSelector(ollama_client=mock_ollama, embed_model="mock"),
        cloud_answerer=StandardCloudAnswerer(cloud_client=mock_ollama),
        synthesizer=LLMSynthesizer(client=mock_ollama),
    )

    results = runner.run_batch(
        sample_cuad_data,
        verbose=False,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    checkpoint_path = tmp_path / "checkpoints" / "checkpoint.json"
    assert checkpoint_path.exists()

    with open(checkpoint_path) as f:
        saved = json.load(f)

    assert len(saved) == len(results)
    assert all("sample_id" in entry for entry in saved)


def test_no_checkpoint_without_dir(mock_ollama, sample_cuad_data, tmp_path):
    from src.pipeline.topic_extraction import LLMTopicExtractor
    from src.pipeline.text_generation import LLMTextGenerator
    from src.pipeline.question_generation import LLMQuestionGenerator
    from src.pipeline.question_selection import EmbeddingQuestionSelector
    from src.pipeline.cloud_answering import StandardCloudAnswerer
    from src.pipeline.synthesis import LLMSynthesizer

    runner = PipelineRunner(
        topic_extractor=LLMTopicExtractor(client=mock_ollama),
        text_generator=LLMTextGenerator(client=mock_ollama),
        question_generator=LLMQuestionGenerator(client=mock_ollama),
        question_selector=EmbeddingQuestionSelector(ollama_client=mock_ollama, embed_model="mock"),
        cloud_answerer=StandardCloudAnswerer(cloud_client=mock_ollama),
        synthesizer=LLMSynthesizer(client=mock_ollama),
    )

    # Should work fine without checkpoint_dir (backward compat)
    results = runner.run_batch(sample_cuad_data, verbose=False)
    assert len(results) == len(sample_cuad_data)
