"""
Steps 2+3: Generate S0 text AND proxy questions from topics.
Security model preserved — only sees the topic distribution, never the original query.

Two modes:
- Parallel (default): Two concurrent async API calls for text and questions.
- Merged: Single call that produces both (fallback for non-async clients).
"""
import asyncio
import re
from src.pipeline.base import TopicDistribution, S0TextGenerator, ProxyQuestionGenerator


# ── Prompts ──────────────────────────────────────────────────────────────────

TEXT_SYSTEM = ("You are a legal and business knowledge writer. Write generic educational text. "
               "Do NOT include any company names, person names, dollar amounts, or dates.")

TEXT_PROMPT = """Write a short paragraph (50-80 words) of generic, educational text about these topics.
Focus on general legal principles and frameworks only.

Topics:
{topic_list}"""

QUESTIONS_SYSTEM = ("You are a legal research assistant. Generate clear, answerable questions "
                    "about general legal and business concepts. No company names, person names, "
                    "dollar amounts, or identifying details.")

QUESTIONS_PROMPT = """Generate exactly {n} research questions about these legal/business topics.
Questions must be answerable from general knowledge. One per line, numbered.

Topics:
{topic_list}"""

MERGED_SYSTEM = ("You are a legal and business knowledge writer. Given topic labels, write "
                 "generic educational text and generate research questions. Do NOT include any "
                 "company names, person names, dollar amounts, specific dates, or identifying information.")

MERGED_PROMPT = """Given these topics:
{topic_list}

1. Write 50-80 words of generic educational text about these topics.
2. Generate exactly {n} research questions, numbered.

TEXT:
[paragraph]

QUESTIONS:
1. [question]
..."""


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_questions(text: str, n: int) -> list[str]:
    """Parse numbered questions from LLM output."""
    questions = []
    for line in text.strip().split("\n"):
        cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line.strip()).strip()
        if cleaned and len(cleaned) > 10:
            questions.append(cleaned)
    return questions[:n]


def _parse_merged_response(response: str, n: int) -> tuple[str, list[str]]:
    """Parse the combined TEXT + QUESTIONS response."""
    parts = re.split(r"QUESTIONS:\s*\n", response, maxsplit=1)
    if len(parts) == 2:
        text = re.sub(r"^TEXT:\s*\n?", "", parts[0]).strip()
        questions = _parse_questions(parts[1], n)
        return text, questions

    # Fallback
    lines = response.strip().split("\n")
    text_lines, questions = [], []
    for line in lines:
        if re.match(r"^\d+[\.\)]\s", line.strip()):
            cleaned = re.sub(r"^\d+[\.\)\-]\s*", "", line.strip()).strip()
            if cleaned and len(cleaned) > 10:
                questions.append(cleaned)
        elif line.strip() and line.strip() not in ("TEXT:", "QUESTIONS:"):
            text_lines.append(line)
    return " ".join(text_lines).strip(), questions[:n]


# ── Formatters ───────────────────────────────────────────────────────────────

def _format_topics(topics: TopicDistribution) -> str:
    return "\n".join(
        f"- {t} ({w:.2f})" for t, w in sorted(topics.topics.items(), key=lambda x: -x[1])
    )


# ── Parallel Generator ───────────────────────────────────────────────────────

class ParallelS0Generator:
    """Two concurrent async API calls: one for text, one for questions.
    Each call is smaller (~75 tokens output) and they run in parallel.
    """

    def __init__(self, client, n_questions: int = 5, temperature: float = 0.0):
        self._client = client
        self._n_questions = n_questions
        self._temperature = temperature
        self._last_text: str = ""
        self._last_questions: list[str] = []

    def _generate(self, topics: TopicDistribution) -> tuple[str, list[str]]:
        topic_list = _format_topics(topics)

        # Check if client supports async
        if hasattr(self._client, "agenerate") and self._client._async_client is not None:
            text, questions = self._parallel_generate(topic_list)
        else:
            text, questions = self._merged_generate(topic_list)

        self._last_text = text
        self._last_questions = questions
        return text, questions

    def _parallel_generate(self, topic_list: str) -> tuple[str, list[str]]:
        """Fire two async calls in parallel."""
        text_prompt = TEXT_PROMPT.format(topic_list=topic_list)
        q_prompt = QUESTIONS_PROMPT.format(topic_list=topic_list, n=self._n_questions)

        async def _run():
            text_task = self._client.agenerate(
                text_prompt, system=TEXT_SYSTEM,
                temperature=self._temperature,
            )
            q_task = self._client.agenerate(
                q_prompt, system=QUESTIONS_SYSTEM,
                temperature=self._temperature,
            )
            return await asyncio.gather(text_task, q_task)

        # Run in event loop (create one if needed)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — use nest_asyncio or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                text_raw, q_raw = pool.submit(lambda: asyncio.run(_run())).result()
        else:
            text_raw, q_raw = asyncio.run(_run())

        return text_raw.strip(), _parse_questions(q_raw, self._n_questions)

    def _merged_generate(self, topic_list: str) -> tuple[str, list[str]]:
        """Fallback: single merged call. Retries on empty response (rate limiting)."""
        prompt = MERGED_PROMPT.format(topic_list=topic_list, n=self._n_questions)

        import time as _time
        for attempt in range(3):
            response = self._client.generate(
                prompt, system=MERGED_SYSTEM,
                temperature=self._temperature,
            )
            text, questions = _parse_merged_response(response, self._n_questions)
            if text or questions:
                return text, questions
            # Empty response — likely rate limited, back off and retry
            _time.sleep(0.5 * (attempt + 1))

        return text, questions

    def generate_text(self, topics: TopicDistribution) -> str:
        text, _ = self._generate(topics)
        return text

    def generate_questions(self, s0_text: str, topics: TopicDistribution, n: int = 5) -> list[str]:
        _, questions = self._generate(topics)
        return questions[:n]


# ── Adapters (plug into existing PipelineRunner) ─────────────────────────────

class MergedTextGeneratorAdapter(S0TextGenerator):
    def __init__(self, merged: ParallelS0Generator):
        self._merged = merged

    def generate(self, topics: TopicDistribution) -> str:
        return self._merged.generate_text(topics)


class MergedQuestionGeneratorAdapter(ProxyQuestionGenerator):
    def __init__(self, merged: ParallelS0Generator):
        self._merged = merged
        self._last_topics: TopicDistribution | None = None

    def set_topics(self, topics: TopicDistribution):
        self._last_topics = topics

    def generate(self, s0_text: str, n: int = 5) -> list[str]:
        if self._last_topics is None:
            return []
        return self._merged.generate_questions(s0_text, self._last_topics, n)


# ── Backward compat alias ────────────────────────────────────────────────────
MergedS0Generator = ParallelS0Generator
