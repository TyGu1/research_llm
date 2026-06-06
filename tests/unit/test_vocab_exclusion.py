"""
Unit tests for RegexVocabExcluder (pure Python, zero external deps).

This is the most self-contained module in the codebase — ideal for verifying
the vocabulary exclusion logic without any LLM or embedding infrastructure.
"""
import pytest
from src.pipeline.base import TopicDistribution
from src.pipeline.vocab_exclusion import RegexVocabExcluder, _topic_vocab_words, STOP_WORDS


@pytest.fixture
def excluder() -> RegexVocabExcluder:
    return RegexVocabExcluder()


def make_topics(**kwargs: float) -> TopicDistribution:
    return TopicDistribution(topics=kwargs)


# ── _topic_vocab_words ────────────────────────────────────────────────────────

def test_vocab_words_splits_underscores():
    topics = make_topics(contract_law=0.5, business_agreements=0.5)
    words = _topic_vocab_words(topics)
    assert "contract" not in words  # "contract" is a STOP_WORD
    assert "law" not in words       # "law" is a STOP_WORD
    assert "business" in words
    # fix: wrong — "agreements" (plural) is NOT in STOP_WORDS; only "agreement" (singular) is
    assert "agreements" in words


def test_vocab_words_filters_stop_words():
    topics = make_topics(legal_obligations=1.0)
    words = _topic_vocab_words(topics)
    assert "legal" not in words    # STOP_WORD
    assert "obligations" in words


def test_vocab_words_filters_short_words():
    topics = make_topics(ip_rights=1.0)
    words = _topic_vocab_words(topics)
    assert "ip" not in words   # len("ip") == 2, filtered


def test_vocab_words_empty_topics():
    assert _topic_vocab_words(make_topics()) == set()


# ── RegexVocabExcluder.exclude ────────────────────────────────────────────────

def test_exclude_removes_topic_word(excluder):
    topics = make_topics(termination=1.0)
    text = "The termination clause allows either party to exit the arrangement."
    result = excluder.exclude(text, topics)
    assert "termination" not in result.lower()
    assert "clause" in result  # non-topic word preserved


def test_exclude_is_case_insensitive(excluder):
    topics = make_topics(termination=1.0)
    text = "TERMINATION is defined in Section 12."
    result = excluder.exclude(text, topics)
    assert "TERMINATION" not in result
    assert "Section" in result


def test_exclude_whole_word_only(excluder):
    topics = make_topics(term=1.0)
    # "term" should not be stripped from "termination" (whole-word match)
    # but "term" as a standalone word should be stripped
    text = "The term of this termination clause is one year."
    result = excluder.exclude(text, topics)
    # "termination" should be preserved (whole-word match protects substrings)
    assert "termination" in result.lower()


def test_exclude_empty_topics_returns_unchanged(excluder):
    text = "This is a contract about termination clauses."
    result = excluder.exclude(text, make_topics())
    assert result == text


def test_exclude_cleans_extra_whitespace(excluder):
    topics = make_topics(obligations=1.0)
    text = "The   obligations  of each party  are listed below."
    result = excluder.exclude(text, topics)
    # No double spaces in result
    assert "  " not in result


def test_exclude_stop_words_never_removed(excluder):
    """STOP_WORDS (contract, legal, parties) should never be stripped,
    even if they appear as topic names."""
    topics = TopicDistribution(topics={w: 0.1 for w in list(STOP_WORDS)[:5]})
    text = "The contract between the parties is governed by law."
    result = excluder.exclude(text, topics)
    # Key domain words from STOP_WORDS preserved
    assert "contract" in result
    assert "parties" in result
