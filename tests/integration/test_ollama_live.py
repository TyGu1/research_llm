"""
Integration tests: Ollama smoke tests.

Marked with @pytest.mark.integration so they can be excluded from fast CI:
  pytest tests/unit/          # fast, no Ollama needed
  pytest tests/integration/   # requires Ollama at localhost:11434

All tests are auto-skipped if Ollama is unreachable.
"""
import pytest
import httpx

from src.llm.ollama_client import OllamaClient


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


@skip_no_ollama
def test_ollama_generate_returns_string():
    client = OllamaClient(base_url=OLLAMA_URL, model="phi3")
    result = client.generate("Say the word 'hello' and nothing else.", temperature=0.0)
    assert isinstance(result, str)
    assert len(result) > 0


@skip_no_ollama
def test_ollama_generate_with_system_prompt():
    client = OllamaClient(base_url=OLLAMA_URL, model="phi3")
    result = client.generate(
        "What is 2+2?",
        system="You are a math tutor. Respond with only the number.",
        temperature=0.0,
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@skip_no_ollama
def test_ollama_embed_returns_list_of_floats():
    client = OllamaClient(base_url=OLLAMA_URL, model="phi3", embed_model="nomic-embed-text")
    result = client.embed("This is a test sentence for embedding.")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)


@skip_no_ollama
def test_ollama_embed_consistent_dimensions():
    """Same model should always return the same embedding dimension."""
    client = OllamaClient(base_url=OLLAMA_URL, model="phi3", embed_model="nomic-embed-text")
    e1 = client.embed("First sentence.")
    e2 = client.embed("Second sentence, quite different from the first one.")
    assert len(e1) == len(e2)


@skip_no_ollama
def test_ollama_embed_different_texts_differ():
    """Embeddings for different texts should not be identical."""
    client = OllamaClient(base_url=OLLAMA_URL, model="phi3", embed_model="nomic-embed-text")
    e1 = client.embed("Contract termination clause.")
    e2 = client.embed("Governing law and jurisdiction selection.")
    assert e1 != e2
