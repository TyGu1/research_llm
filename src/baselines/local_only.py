"""
Local-only baseline: local LLM answers the S2 question without cloud access.
This is the quality floor — what's achievable with full privacy but weaker model.
"""
import time

from src.data.cuad_loader import CUADSample
from src.baselines.oracle import BaselineResult

SYSTEM_PROMPT = """You are a legal expert specializing in contract analysis.
Answer questions about contracts accurately and thoroughly."""

PROMPT_TEMPLATE = """Answer the following question about this contract.

Question: {question}

Contract:
{context}

Provide a thorough, accurate answer based on the contract text."""


class LocalOnlyBaseline:
    def __init__(self, local_client, max_context_chars: int = 8000):
        self._client = local_client
        self._max_context_chars = max_context_chars

    def run(self, sample: CUADSample) -> BaselineResult:
        t0 = time.monotonic()
        prompt = PROMPT_TEMPLATE.format(
            question=sample.question,
            context=sample.context[:self._max_context_chars],
        )
        answer = self._client.generate(prompt, system=SYSTEM_PROMPT)
        return BaselineResult(answer=answer, timing=time.monotonic() - t0)
