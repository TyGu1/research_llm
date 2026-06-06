"""
Step 4: Select the K proxy questions closest to the original S2 question.
The original S2 question is seen ONLY by the local LLM — never transmitted.

Two implementations behind the same ABC:
- EmbeddingQuestionSelector: cosine similarity via nomic-embed-text (default, ~0.1s)
- LLMQuestionSelector: local LLM ranks candidates (~5s, potentially better quality)
"""
import re
import numpy as np

from src.pipeline.base import QuestionSelector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class EmbeddingQuestionSelector(QuestionSelector):
    """Selects questions by cosine similarity of nomic-embed-text embeddings."""

    def __init__(self, ollama_client, embed_model: str = "nomic-embed-text"):
        self._client = ollama_client
        self._embed_model = embed_model

    def select(self, original: str, candidates: list[str], k: int = 3) -> list[str]:
        if not candidates:
            return []

        orig_emb = self._client.embed(original, model=self._embed_model)
        scored: list[tuple[float, str]] = []

        for q in candidates:
            emb = self._client.embed(q, model=self._embed_model)
            score = _cosine_similarity(orig_emb, emb)
            scored.append((score, q))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [q for _, q in scored[:k]]


class LLMQuestionSelector(QuestionSelector):
    """Selects questions using a local LLM to reason about relevance."""

    SYSTEM_PROMPT = "You are a question relevance analyst. Given an original question and candidate questions, identify the most relevant candidates."

    PROMPT_TEMPLATE = """Original question: {original}

Candidate questions:
{candidates}

Select the {k} candidates most relevant to the original question.
Respond with ONLY the numbers of the selected questions, comma-separated (e.g., "1, 3, 7")."""

    def __init__(self, client):
        self._client = client

    def select(self, original: str, candidates: list[str], k: int = 3) -> list[str]:
        if not candidates:
            return []

        candidate_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(candidates))
        prompt = self.PROMPT_TEMPLATE.format(
            original=original,
            candidates=candidate_list,
            k=k,
        )
        response = self._client.generate(prompt, system=self.SYSTEM_PROMPT)

        # Parse comma-separated indices
        indices = [int(m) - 1 for m in re.findall(r"\d+", response)]
        selected = [candidates[i] for i in indices if 0 <= i < len(candidates)]

        # Fallback to first K if parsing failed
        if not selected:
            selected = candidates[:k]

        return selected[:k]
