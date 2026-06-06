"""
Unit tests for EmbeddingQuestionSelector and LLMQuestionSelector.

EmbeddingQuestionSelector uses MockOllamaClient.embed() which returns
deterministic pseudo-embeddings — sufficient for testing the ranking logic.
"""
import pytest
from src.pipeline.question_selection import (
    EmbeddingQuestionSelector, LLMQuestionSelector, _cosine_similarity,
)


CANDIDATES = [
    "What are the key obligations under this type of agreement?",
    "How are disputes typically resolved in similar contracts?",
    "What termination conditions are standard for this clause type?",
    "What liability limitations are commonly included?",
    "How is notice typically defined in these agreements?",
]

ORIGINAL = "What are the termination conditions in this contract?"


# ── _cosine_similarity ────────────────────────────────────────────────────────

def test_cosine_same_vector():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_cosine_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ── EmbeddingQuestionSelector ─────────────────────────────────────────────────

@pytest.fixture
def embed_selector(mock_ollama) -> EmbeddingQuestionSelector:
    return EmbeddingQuestionSelector(ollama_client=mock_ollama)


def test_embedding_selector_returns_k(embed_selector):
    result = embed_selector.select(ORIGINAL, CANDIDATES, k=3)
    assert len(result) == 3


def test_embedding_selector_returns_subset(embed_selector):
    result = embed_selector.select(ORIGINAL, CANDIDATES, k=3)
    for q in result:
        assert q in CANDIDATES


def test_embedding_selector_empty_candidates(embed_selector):
    result = embed_selector.select(ORIGINAL, [], k=3)
    assert result == []


def test_embedding_selector_k_larger_than_candidates(embed_selector):
    short = CANDIDATES[:2]
    result = embed_selector.select(ORIGINAL, short, k=5)
    assert len(result) == 2  # can't return more than available


def test_embedding_selector_k_one(embed_selector):
    result = embed_selector.select(ORIGINAL, CANDIDATES, k=1)
    assert len(result) == 1


# ── LLMQuestionSelector ───────────────────────────────────────────────────────

@pytest.fixture
def llm_selector(mock_ollama) -> LLMQuestionSelector:
    return LLMQuestionSelector(client=mock_ollama)


def test_llm_selector_returns_candidates_subset(llm_selector):
    result = llm_selector.select(ORIGINAL, CANDIDATES, k=3)
    # Either parsed from mock ("1, 2, 3") or fallback to first k
    assert len(result) <= 3
    for q in result:
        assert q in CANDIDATES


def test_llm_selector_empty_candidates(llm_selector):
    result = llm_selector.select(ORIGINAL, [], k=3)
    assert result == []
