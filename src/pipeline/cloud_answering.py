"""
Step 5: Cloud SOTA LLM answers the selected S0 proxy questions.
Cloud LLM receives ONLY the S0 questions — no sensitive data.
"""
import logging
import time

from src.pipeline.base import CloudAnswerer, S0Answer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable legal and business expert. Answer questions
about general legal concepts, contract law, and business practices. Provide thorough,
accurate answers based on established legal principles."""


class StandardCloudAnswerer(CloudAnswerer):
    def __init__(self, cloud_client, temperature: float = 0.0, max_retries: int = 3):
        self._client = cloud_client
        self._temperature = temperature
        self._max_retries = max_retries

    def _generate_with_retry(self, prompt: str) -> str:
        for attempt in range(self._max_retries):
            try:
                return self._client.generate(
                    prompt,
                    system=SYSTEM_PROMPT,
                    temperature=self._temperature,
                )
            except Exception:
                if attempt == self._max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("Cloud API failed (attempt %d/%d), retrying in %ds", attempt + 1, self._max_retries, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")  # pragma: no cover

    def answer(self, questions: list[str]) -> list[S0Answer]:
        results: list[S0Answer] = []
        for q in questions:
            answer = self._generate_with_retry(q)
            results.append(S0Answer(question=q, answer=answer))
        return results
