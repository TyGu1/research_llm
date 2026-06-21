"""
Step 1: Extract topic distribution from a question + context using a local LLM.
Outputs a TopicDistribution (dict of topic_name → weight).
"""
import json
import logging
import re
import time

from src.pipeline.base import TopicDistribution, TopicExtractor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a topic modeling assistant. Given a legal question and contract context,
identify the main legal/business topics present. Respond ONLY with valid JSON in this exact format:
{"topics": {"topic_name": weight, ...}}
where weights are floats that sum to 1.0. Use 3-6 topics. Topic names should be generic
legal/business concepts, NOT company names, person names, or specific entities."""

PROMPT_TEMPLATE = """Identify the main legal and business topics in this question about a contract.

Question: {question}

Contract context (excerpt):
{context}

Respond with JSON only: {{"topics": {{"topic_name": weight, ...}}}}"""


def _parse_topic_json(text: str) -> dict[str, float]:
    """Parse LLM output into topic dict. Falls back to regex if JSON is malformed."""
    # Try direct JSON parse first
    try:
        data = json.loads(text.strip())
        if "topics" in data and isinstance(data["topics"], dict):
            return {str(k): float(v) for k, v in data["topics"].items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex fallback: find "word": number patterns
    matches = re.findall(r'"([^"]+)"\s*:\s*([\d.]+)', text)
    if matches:
        raw = {k: float(v) for k, v in matches if k != "topics"}
        if raw:
            total = sum(raw.values())
            return {k: v / total for k, v in raw.items()} if total > 0 else raw

    # Last resort: return a generic topic
    logger.warning("Topic extraction failed — LLM output could not be parsed, falling back to generic topic. Raw output: %.200s", text)
    return {"legal_contract": 1.0}


class LLMTopicExtractor(TopicExtractor):
    def __init__(self, client, max_context_tokens: int = 2000, temperature: float = 0.0):
        self._client = client
        self._max_context_tokens = max_context_tokens
        self._temperature = temperature

    def extract(self, question: str, context: str) -> TopicDistribution:
        # Rough truncation: ~4 chars per token; 0 = no truncation
        if self._max_context_tokens > 0:
            max_chars = self._max_context_tokens * 4
            truncated_ctx = context[:max_chars]
        else:
            truncated_ctx = context

        prompt = PROMPT_TEMPLATE.format(question=question, context=truncated_ctx)
        response = self._client.generate(prompt, system=SYSTEM_PROMPT, temperature=self._temperature)
        topics = _parse_topic_json(response)

        # Normalize weights
        total = sum(topics.values())
        if total > 0:
            topics = {k: v / total for k, v in topics.items()}

        return TopicDistribution(topics=topics)
