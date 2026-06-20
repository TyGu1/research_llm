"""
Step 3: Generate 8-12 proxy S0 questions from the generated S0 text.
Questions must be answerable from public knowledge — no sensitive specifics.
"""
import re

from src.pipeline.base import ProxyQuestionGenerator

SYSTEM_PROMPT = """You are a legal research assistant. Generate clear, answerable questions
about general legal and business concepts. Questions must be generic — no company names,
person names, dollar amounts, or identifying details."""

PROMPT_TEMPLATE = """Based on the following text about legal and business topics, generate
exactly {n} questions that could be answered by a general legal expert without needing
any confidential or proprietary information.

Text:
{s0_text}

Generate exactly {n} questions, one per line, numbered 1 through {n}.
Each question must be self-contained and answerable from general legal knowledge."""


def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered list from LLM output."""
    lines = text.strip().split("\n")
    questions: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading number + punctuation (e.g., "1.", "1)", "1 -")
        cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line).strip()
        if cleaned and len(cleaned) > 10 and "?" in cleaned:
            questions.append(cleaned)

    return questions


class LLMQuestionGenerator(ProxyQuestionGenerator):
    def __init__(self, client, n: int = 10, temperature: float = 0.0):
        self._client = client
        self._default_n = n
        self._temperature = temperature

    def generate(self, s0_text: str, n: int | None = None) -> list[str]:
        n = n or self._default_n
        prompt = PROMPT_TEMPLATE.format(n=n, s0_text=s0_text)
        response = self._client.generate(prompt, system=SYSTEM_PROMPT, temperature=self._temperature)

        questions = _parse_numbered_list(response)

        # Fallback: if parsing failed, split on "?" and clean up
        if len(questions) < 3:
            raw_questions = [q.strip() + "?" for q in response.split("?") if len(q.strip()) > 10]
            questions = raw_questions

        return questions[:n]
