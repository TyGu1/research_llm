"""
Steps 2+3 merged: Single LLM call that generates S0 text AND proxy questions from topics.
Security model preserved — only sees the topic distribution, never the original query.
"""
import re
from src.pipeline.base import TopicDistribution, S0TextGenerator, ProxyQuestionGenerator


SYSTEM_PROMPT = """You are a legal and business knowledge writer. Given topic labels, write
generic educational text and generate research questions. Do NOT include any company names,
person names, dollar amounts, specific dates, or any identifying information."""

PROMPT_TEMPLATE = """Given these topics and their relative importance:
{topic_list}

Do two things:

1. Write a short paragraph (100-150 words) of generic, educational text about these topics.
   Focus on general legal principles and frameworks only.

2. Then generate exactly {n} research questions that a legal expert could answer from
   general knowledge. One question per line, numbered.

Format your response exactly as:

TEXT:
[your paragraph here]

QUESTIONS:
1. [question]
2. [question]
..."""


def _parse_merged_response(response: str, n: int) -> tuple[str, list[str]]:
    """Parse the combined TEXT + QUESTIONS response."""
    text = ""
    questions = []

    # Split on QUESTIONS: marker
    parts = re.split(r'\nQUESTIONS:\s*\n', response, maxsplit=1)
    if len(parts) == 2:
        text_part, q_part = parts
        # Clean text part (remove TEXT: prefix)
        text = re.sub(r'^TEXT:\s*\n?', '', text_part).strip()
        # Parse numbered questions
        for line in q_part.strip().split('\n'):
            line = line.strip()
            cleaned = re.sub(r'^\d+[\.\)\-]\s*', '', line).strip()
            if cleaned and len(cleaned) > 10:
                questions.append(cleaned)
    else:
        # Fallback: try to split on numbered list
        lines = response.strip().split('\n')
        text_lines = []
        for line in lines:
            if re.match(r'^\d+[\.\)]\s', line.strip()):
                cleaned = re.sub(r'^\d+[\.\)\-]\s*', '', line.strip()).strip()
                if cleaned and len(cleaned) > 10:
                    questions.append(cleaned)
            elif line.strip() and line.strip() not in ('TEXT:', 'QUESTIONS:'):
                text_lines.append(line)
        text = ' '.join(text_lines).strip()

    return text, questions[:n]


class MergedS0Generator:
    """Single LLM call that produces both S0 text and proxy questions.

    Implements both S0TextGenerator and ProxyQuestionGenerator interfaces
    through a combined call, cached per topic distribution.
    """

    def __init__(self, client, n_questions: int = 10, temperature: float = 0.0):
        self._client = client
        self._n_questions = n_questions
        self._temperature = temperature
        # Cache last result to avoid double calls
        self._last_topics_key: str = ""
        self._last_text: str = ""
        self._last_questions: list[str] = []

    def _generate(self, topics: TopicDistribution) -> tuple[str, list[str]]:
        """Run the merged generation call."""
        topics_key = str(sorted(topics.topics.items()))
        if topics_key == self._last_topics_key:
            return self._last_text, self._last_questions

        topic_list = "\n".join(
            f"- {topic} (importance: {weight:.2f})"
            for topic, weight in sorted(topics.topics.items(), key=lambda x: -x[1])
        )
        prompt = PROMPT_TEMPLATE.format(topic_list=topic_list, n=self._n_questions)
        response = self._client.generate(prompt, system=SYSTEM_PROMPT, temperature=self._temperature)

        text, questions = _parse_merged_response(response, self._n_questions)

        # Cache
        self._last_topics_key = topics_key
        self._last_text = text
        self._last_questions = questions

        return text, questions

    def generate_text(self, topics: TopicDistribution) -> str:
        text, _ = self._generate(topics)
        return text

    def generate_questions(self, s0_text: str, topics: TopicDistribution, n: int = 10) -> list[str]:
        _, questions = self._generate(topics)
        return questions[:n]


class MergedTextGeneratorAdapter(S0TextGenerator):
    """Adapter so MergedS0Generator can plug into the pipeline as a S0TextGenerator."""

    def __init__(self, merged: MergedS0Generator):
        self._merged = merged

    def generate(self, topics: TopicDistribution) -> str:
        return self._merged.generate_text(topics)


class MergedQuestionGeneratorAdapter(ProxyQuestionGenerator):
    """Adapter so MergedS0Generator can plug into the pipeline as a ProxyQuestionGenerator."""

    def __init__(self, merged: MergedS0Generator):
        self._merged = merged
        self._last_topics: TopicDistribution | None = None

    def set_topics(self, topics: TopicDistribution):
        """Set current topics so generate() can access them (called by runner)."""
        self._last_topics = topics

    def generate(self, s0_text: str, n: int = 10) -> list[str]:
        if self._last_topics is None:
            return []
        return self._merged.generate_questions(s0_text, self._last_topics, n)
