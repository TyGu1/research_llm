"""
Unit tests for LLMTopicExtractor and the JSON parsing utilities.

Key invariants:
- Weights always normalize to sum ≈ 1.0
- Fallback parser handles malformed JSON gracefully
- Context truncation respects max_context_tokens
"""
import json
import pytest
from src.pipeline.base import TopicDistribution
from src.pipeline.topic_extraction import LLMTopicExtractor, _parse_topic_json


# ── _parse_topic_json ─────────────────────────────────────────────────────────

def test_parse_valid_json():
    text = json.dumps({"topics": {"contract_law": 0.5, "business": 0.5}})
    result = _parse_topic_json(text)
    assert "contract_law" in result
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_parse_json_with_extra_whitespace():
    text = '   {"topics": {"ip_rights": 0.7, "licensing": 0.3}}   '
    result = _parse_topic_json(text)
    assert len(result) == 2


def test_parse_json_missing_topics_key():
    # Falls through to regex fallback
    text = '{"contract_law": 0.6, "liability": 0.4}'
    result = _parse_topic_json(text)
    # Regex fallback still finds the key-value pairs
    assert isinstance(result, dict)
    assert len(result) >= 1


def test_parse_malformed_json_regex_fallback():
    text = '"contract_law": 0.5, "liability": 0.3, "termination": 0.2'
    result = _parse_topic_json(text)
    assert "contract_law" in result
    # Fallback normalizes weights
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_parse_empty_string_last_resort():
    result = _parse_topic_json("")
    assert result == {"legal_contract": 1.0}


def test_parse_gibberish_last_resort():
    result = _parse_topic_json("This is not JSON at all!!!")
    assert result == {"legal_contract": 1.0}


# ── LLMTopicExtractor ─────────────────────────────────────────────────────────

@pytest.fixture
def extractor(mock_ollama) -> LLMTopicExtractor:
    return LLMTopicExtractor(client=mock_ollama, max_context_tokens=2000)


def test_extract_returns_topic_distribution(extractor):
    result = extractor.extract("What are the termination conditions?", "Some contract text here.")
    assert isinstance(result, TopicDistribution)
    assert isinstance(result.topics, dict)
    assert len(result.topics) >= 1


def test_extract_weights_sum_to_one(extractor):
    result = extractor.extract("What are the IP ownership provisions?", "Contract context.")
    total = sum(result.topics.values())
    assert abs(total - 1.0) < 1e-6


def test_extract_truncates_context(mock_ollama):
    """With max_context_tokens=10 (~40 chars), long context should be truncated."""
    extractor = LLMTopicExtractor(client=mock_ollama, max_context_tokens=10)
    long_context = "A" * 10000
    # Should not raise — truncation prevents sending huge prompts
    result = extractor.extract("Some question?", long_context)
    assert isinstance(result, TopicDistribution)


def test_extract_no_empty_topics(extractor):
    """Even on odd inputs, at least one topic is returned."""
    result = extractor.extract("", "")
    assert len(result.topics) >= 1
