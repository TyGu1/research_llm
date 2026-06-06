"""
Unit tests for LLMQuestionGenerator and the numbered-list parser.
"""
import pytest
from src.pipeline.question_generation import LLMQuestionGenerator, _parse_numbered_list


# ── _parse_numbered_list ──────────────────────────────────────────────────────

def test_parse_standard_numbered_list():
    text = (
        "1. What are the key obligations under this type of agreement?\n"
        "2. How are disputes typically resolved in similar contracts?\n"
        "3. What termination conditions are standard for this clause type?\n"
    )
    result = _parse_numbered_list(text)
    assert len(result) == 3
    assert all("?" in q for q in result)


def test_parse_parenthesis_numbering():
    text = "1) What are the liability limitations?\n2) How is notice defined?"
    result = _parse_numbered_list(text)
    assert len(result) == 2


def test_parse_dash_numbering():
    text = "1 - What are renewal terms?\n2 - What is the warranty duration?"
    result = _parse_numbered_list(text)
    assert len(result) == 2


def test_parse_skips_lines_without_question_mark():
    text = (
        "1. What are the key obligations?\n"
        "2. This line has no question mark\n"
        "3. How are disputes resolved?\n"
    )
    result = _parse_numbered_list(text)
    assert len(result) == 2  # line 2 skipped


def test_parse_skips_short_lines():
    text = "1. Q?\n2. What are the key obligations under this type of agreement?\n"
    result = _parse_numbered_list(text)
    assert len(result) == 1  # "Q?" is too short (< 10 chars)


def test_parse_empty_string():
    assert _parse_numbered_list("") == []


# ── LLMQuestionGenerator ──────────────────────────────────────────────────────

@pytest.fixture
def generator(mock_ollama) -> LLMQuestionGenerator:
    return LLMQuestionGenerator(client=mock_ollama, n=8)


def test_generate_returns_list(generator):
    result = generator.generate("Some generic text about contract termination clauses.")
    assert isinstance(result, list)
    assert len(result) >= 1


def test_generate_respects_n_limit(mock_ollama):
    gen = LLMQuestionGenerator(client=mock_ollama, n=3)
    result = gen.generate("Some generic text about contract clauses.", n=3)
    assert len(result) <= 3


def test_generate_n_override(generator):
    result = generator.generate("Some text.", n=2)
    assert len(result) <= 2


def test_generate_all_are_strings(generator):
    result = generator.generate("Generic legal text about liability limitations.")
    assert all(isinstance(q, str) for q in result)
