"""Tests for fast pipeline components: embedding topic extraction, merged S0+questions, templates."""
import pytest
from src.pipeline.base import TopicDistribution


# ── Embedding Topic Extraction ───────────────────────────────────────────────

class TestEmbeddingTopicExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        from src.pipeline.embedding_topic_extraction import EmbeddingTopicExtractor
        return EmbeddingTopicExtractor(top_k=5, min_score=0.1)

    def test_returns_topic_distribution(self, extractor, sample_cuad_data):
        result = extractor.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        assert isinstance(result, TopicDistribution)
        assert len(result.topics) > 0

    def test_weights_sum_to_one(self, extractor, sample_cuad_data):
        result = extractor.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        total = sum(result.topics.values())
        assert abs(total - 1.0) < 0.01

    def test_respects_top_k(self, extractor, sample_cuad_data):
        result = extractor.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        assert len(result.topics) <= 5

    def test_noncompete_gets_relevant_topics(self, extractor, sample_cuad_data):
        result = extractor.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        topic_names = " ".join(result.topics.keys()).lower()
        assert any(kw in topic_names for kw in ["non-compete", "employment", "contract", "compete"])

    def test_ip_query_gets_ip_topic(self, extractor, sample_cuad_data):
        result = extractor.extract(sample_cuad_data[1].question, sample_cuad_data[1].context)
        topic_names = " ".join(result.topics.keys()).lower()
        assert "intellectual property" in topic_names

    def test_custom_topic_descriptions(self):
        from src.pipeline.embedding_topic_extraction import EmbeddingTopicExtractor
        custom = {"healthcare": "medical records, patient data, HIPAA, clinical trials"}
        ext = EmbeddingTopicExtractor(topic_descriptions=custom, top_k=1)
        result = ext.extract("What are the HIPAA requirements?", "Patient data handling.")
        assert "healthcare" in result.topics

    def test_fast_execution(self, extractor, sample_cuad_data):
        import time
        # Warmup
        extractor.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        # Measure
        t0 = time.monotonic()
        for _ in range(10):
            extractor.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        elapsed = (time.monotonic() - t0) / 10
        assert elapsed < 0.5  # should be ~25ms, allow generous margin


# ── Keyword Topic Extraction ─────────────────────────────────────────────────

class TestKeywordTopicExtractor:
    def test_returns_topics(self, sample_cuad_data):
        from src.pipeline.fast_topic_extraction import KeywordTopicExtractor
        ext = KeywordTopicExtractor()
        result = ext.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        assert isinstance(result, TopicDistribution)
        assert len(result.topics) > 0

    def test_weights_normalize(self, sample_cuad_data):
        from src.pipeline.fast_topic_extraction import KeywordTopicExtractor
        ext = KeywordTopicExtractor()
        result = ext.extract(sample_cuad_data[0].question, sample_cuad_data[0].context)
        assert abs(sum(result.topics.values()) - 1.0) < 0.01

    def test_fallback_on_empty(self):
        from src.pipeline.fast_topic_extraction import KeywordTopicExtractor
        ext = KeywordTopicExtractor()
        result = ext.extract("xyz", "abc")
        assert "contract terms and conditions" in result.topics


# ── Template Text Generation ─────────────────────────────────────────────────

class TestTemplateTextGenerator:
    def test_generates_text(self):
        from src.pipeline.fast_text_generation import TemplateTextGenerator
        gen = TemplateTextGenerator()
        topics = TopicDistribution(topics={"mergers and acquisitions": 0.6, "antitrust and competition": 0.4})
        text = gen.generate(topics)
        assert len(text.split()) >= 50
        assert "acquisition" in text.lower() or "merger" in text.lower()

    def test_fallback_for_unknown_topic(self):
        from src.pipeline.fast_text_generation import TemplateTextGenerator
        gen = TemplateTextGenerator()
        topics = TopicDistribution(topics={"quantum computing": 1.0})
        text = gen.generate(topics)
        assert len(text) > 0

    def test_trims_to_300_words(self):
        from src.pipeline.fast_text_generation import TemplateTextGenerator
        gen = TemplateTextGenerator()
        topics = TopicDistribution(topics={
            "mergers and acquisitions": 0.25,
            "antitrust and competition": 0.25,
            "intellectual property": 0.25,
            "employment and labor": 0.25,
        })
        text = gen.generate(topics)
        assert len(text.split()) <= 310


# ── Template Question Generation ─────────────────────────────────────────────

class TestTemplateQuestionGenerator:
    def test_generates_questions(self):
        from src.pipeline.fast_question_generation import TemplateQuestionGenerator
        gen = TemplateQuestionGenerator(seed=42)
        questions = gen.generate("Corporate acquisitions involve due diligence...", n=10)
        assert len(questions) == 10
        assert all("?" in q for q in questions)

    def test_fallback_questions(self):
        from src.pipeline.fast_question_generation import TemplateQuestionGenerator
        gen = TemplateQuestionGenerator(seed=42)
        questions = gen.generate("totally unrelated text about cooking", n=10)
        assert len(questions) >= 8  # should get fallbacks


# ── Merged S0 Generator ──────────────────────────────────────────────────────

class TestMergedS0Generator:
    def test_parse_merged_response(self):
        from src.pipeline.merged_s0_questions import _parse_merged_response
        response = """TEXT:
This is the generated S0 text about legal topics.

QUESTIONS:
1. What are the key considerations?
2. How do parties allocate risk?
3. What regulatory requirements apply?"""
        text, questions = _parse_merged_response(response, 10)
        assert "S0 text" in text
        assert len(questions) == 3

    def test_parse_fallback_no_markers(self):
        from src.pipeline.merged_s0_questions import _parse_merged_response
        response = """Legal topics involve complex considerations.
1. What are the key factors?
2. How is risk managed?"""
        text, questions = _parse_merged_response(response, 10)
        assert len(text) > 0
        assert len(questions) == 2

    def test_generates_text_and_questions(self):
        from unittest.mock import MagicMock
        from src.pipeline.merged_s0_questions import MergedS0Generator

        mock_client = MagicMock()
        mock_client._async_client = None  # force merged mode
        mock_client.generate.return_value = """TEXT:
Test text about legal topics.

QUESTIONS:
1. Question one here?
2. Question two here?"""

        gen = MergedS0Generator(client=mock_client, n_questions=5)
        topics = TopicDistribution(topics={"test": 1.0})

        text = gen.generate_text(topics)
        assert "Test text" in text
        assert mock_client.generate.call_count == 1

    def test_retry_on_empty_response(self):
        from unittest.mock import MagicMock
        from src.pipeline.merged_s0_questions import MergedS0Generator

        mock_client = MagicMock()
        mock_client._async_client = None
        # First call returns empty, second returns valid
        mock_client.generate.side_effect = [
            "",
            "TEXT:\nRetried text.\n\nQUESTIONS:\n1. Retried question?",
        ]

        gen = MergedS0Generator(client=mock_client, n_questions=5)
        topics = TopicDistribution(topics={"test": 1.0})
        text = gen.generate_text(topics)
        assert "Retried" in text
        assert mock_client.generate.call_count == 2

    def test_adapters(self):
        from unittest.mock import MagicMock
        from src.pipeline.merged_s0_questions import (
            ParallelS0Generator, MergedTextGeneratorAdapter, MergedQuestionGeneratorAdapter,
        )

        mock_client = MagicMock()
        mock_client._async_client = None  # force merged mode
        mock_client.generate.return_value = """TEXT:
Adapter test text.

QUESTIONS:
1. Adapter question one?
2. Adapter question two?"""

        merged = ParallelS0Generator(client=mock_client, n_questions=5)
        text_adapter = MergedTextGeneratorAdapter(merged)
        q_adapter = MergedQuestionGeneratorAdapter(merged)

        topics = TopicDistribution(topics={"contract": 1.0})
        q_adapter.set_topics(topics)

        text = text_adapter.generate(topics)
        assert "Adapter" in text

        questions = q_adapter.generate("", n=5)
        assert len(questions) == 2
