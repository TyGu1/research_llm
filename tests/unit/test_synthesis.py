"""
Unit tests for the synthesis pipeline step.
"""
from src.pipeline.synthesis import LLMSynthesizer
from src.pipeline.base import S0Answer


def test_synthesize_returns_string(mock_ollama):
    synth = LLMSynthesizer(client=mock_ollama)
    result = synth.synthesize(
        original_question="What non-compete terms apply?",
        context="Neither party shall compete for two years.",
        s0_answers=[S0Answer(question="What are typical terms?", answer="Usually 1-2 years.")],
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_synthesize_truncates_context(mock_ollama):
    synth = LLMSynthesizer(client=mock_ollama, max_context_chars=50)
    long_context = "A" * 10000
    result = synth.synthesize(
        original_question="Q?",
        context=long_context,
        s0_answers=[S0Answer(question="Q?", answer="A.")],
    )
    assert isinstance(result, str)


def test_synthesize_handles_empty_s0_answers(mock_ollama):
    synth = LLMSynthesizer(client=mock_ollama)
    result = synth.synthesize(
        original_question="Q?",
        context="Some contract text.",
        s0_answers=[],
    )
    assert isinstance(result, str)


def test_synthesize_passes_temperature(mock_ollama):
    synth = LLMSynthesizer(client=mock_ollama, temperature=0.7)
    assert synth._temperature == 0.7
