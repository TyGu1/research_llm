"""
Unit tests for topic extraction fallback logging.
"""
import logging
from src.pipeline.topic_extraction import _parse_topic_json


def test_fallback_logs_warning_on_garbage_input(caplog):
    with caplog.at_level(logging.WARNING, logger="src.pipeline.topic_extraction"):
        result = _parse_topic_json("this is not json at all")
    assert result == {"legal_contract": 1.0}
    assert "Topic extraction failed" in caplog.text
    assert "falling back to generic topic" in caplog.text


def test_no_warning_on_valid_json(caplog):
    with caplog.at_level(logging.WARNING, logger="src.pipeline.topic_extraction"):
        result = _parse_topic_json('{"topics": {"law": 0.5, "contracts": 0.5}}')
    assert result == {"law": 0.5, "contracts": 0.5}
    assert "Topic extraction failed" not in caplog.text


def test_no_warning_on_regex_fallback(caplog):
    """Regex fallback should succeed silently — only last-resort triggers the warning."""
    with caplog.at_level(logging.WARNING, logger="src.pipeline.topic_extraction"):
        result = _parse_topic_json('Here are topics: "law": 0.6, "finance": 0.4')
    assert "law" in result
    assert "Topic extraction failed" not in caplog.text
