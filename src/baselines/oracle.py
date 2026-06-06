"""
Oracle baseline: cloud LLM given the full S2 question + context directly.
This is the quality ceiling — what we'd get with no privacy constraints.
"""
import time
from dataclasses import dataclass

from src.data.cuad_loader import CUADSample

SYSTEM_PROMPT = """You are a legal expert specializing in contract analysis.
Answer questions about contracts accurately and thoroughly."""

PROMPT_TEMPLATE = """Answer the following question about this contract.

Question: {question}

Contract:
{context}

Provide a thorough, accurate answer based on the contract text."""


@dataclass
class BaselineResult:
    answer: str
    timing: float


class OracleBaseline:
    def __init__(self, cloud_client, max_context_chars: int = 8000):
        self._client = cloud_client
        self._max_context_chars = max_context_chars

    def run(self, sample: CUADSample) -> BaselineResult:
        t0 = time.monotonic()
        prompt = PROMPT_TEMPLATE.format(
            question=sample.question,
            context=sample.context[:self._max_context_chars],
        )
        answer = self._client.generate(prompt, system=SYSTEM_PROMPT)
        return BaselineResult(answer=answer, timing=time.monotonic() - t0)
