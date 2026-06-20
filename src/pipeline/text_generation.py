"""
Step 2: Generate 200-300 words of S0 (public-safe) text about extracted topics.
The text is generic — no entity names, no specific values, no original question content.
"""
from src.pipeline.base import TopicDistribution, S0TextGenerator

SYSTEM_PROMPT = """You are a legal and business knowledge writer. Write clear, generic,
educational text about legal and business topics. Do NOT include any company names,
person names, dollar amounts, specific dates, or any identifying information.
Write only about general concepts, frameworks, and principles."""

PROMPT_TEMPLATE = """Write {min_words}-{max_words} words about the following legal and business topics.
Be educational and general. Do not reference any specific companies, people, or transactions.
Focus only on general legal principles, frameworks, and considerations.

Topics and their relative importance:
{topic_list}

Write the text now:"""


class LLMTextGenerator(S0TextGenerator):
    def __init__(self, client, min_words: int = 200, max_words: int = 300, temperature: float = 0.0):
        self._client = client
        self._min_words = min_words
        self._max_words = max_words
        self._temperature = temperature

    def generate(self, topics: TopicDistribution) -> str:
        topic_list = "\n".join(
            f"- {topic} (importance: {weight:.2f})"
            for topic, weight in sorted(topics.topics.items(), key=lambda x: -x[1])
        )
        prompt = PROMPT_TEMPLATE.format(
            min_words=self._min_words,
            max_words=self._max_words,
            topic_list=topic_list,
        )

        text = self._client.generate(prompt, system=SYSTEM_PROMPT, temperature=self._temperature)

        # Validate word count; retry once if out of range
        word_count = len(text.split())
        if word_count < self._min_words or word_count > self._max_words * 1.5:
            retry_prompt = (
                f"Rewrite the following text so it is between {self._min_words} "
                f"and {self._max_words} words. Keep the same topics and tone.\n\n{text}"
            )
            text = self._client.generate(retry_prompt, system=SYSTEM_PROMPT, temperature=self._temperature)

        return text.strip()
