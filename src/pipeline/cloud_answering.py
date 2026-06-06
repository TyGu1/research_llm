"""
Step 5: Cloud SOTA LLM answers the selected S0 proxy questions.
Cloud LLM receives ONLY the S0 questions — no sensitive data.
"""
from src.pipeline.base import CloudAnswerer, S0Answer

SYSTEM_PROMPT = """You are a knowledgeable legal and business expert. Answer questions
about general legal concepts, contract law, and business practices. Provide thorough,
accurate answers based on established legal principles."""


class StandardCloudAnswerer(CloudAnswerer):
    def __init__(self, cloud_client, temperature: float = 0.0):
        self._client = cloud_client
        self._temperature = temperature

    def answer(self, questions: list[str]) -> list[S0Answer]:
        results: list[S0Answer] = []
        for q in questions:
            answer = self._client.generate(
                q,
                system=SYSTEM_PROMPT,
                temperature=self._temperature,
            )
            results.append(S0Answer(question=q, answer=answer))
        return results
