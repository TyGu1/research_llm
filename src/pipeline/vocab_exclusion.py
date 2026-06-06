"""
Step 2.5 (optional hardening): Strip topic vocabulary words from the generated S0 text.
What remains: semantic structure without the domain-specific lexicon.
This is the vocabulary exclusion variant — not found in prior literature.

Pure Python regex — no LLM call needed.
"""
import re

from src.pipeline.base import TopicDistribution, VocabExcluder

# Words too generic to strip (would destroy readability)
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "and", "or", "but", "if",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "that", "this", "these", "those", "it", "its", "their", "they",
    "not", "no", "any", "all", "each", "both", "between", "such",
    "law", "legal", "contract", "agreement", "party", "parties",
}


def _topic_vocab_words(topics: TopicDistribution) -> set[str]:
    """Extract individual words from topic names (split on underscores and spaces)."""
    words: set[str] = set()
    for topic_name in topics.topics:
        for word in re.split(r"[_\s/]+", topic_name.lower()):
            word = word.strip()
            if word and word not in STOP_WORDS and len(word) > 2:
                words.add(word)
    return words


class RegexVocabExcluder(VocabExcluder):
    """Removes topic vocabulary words from text using whole-word regex substitution."""

    def exclude(self, text: str, topics: TopicDistribution) -> str:
        vocab = _topic_vocab_words(topics)
        if not vocab:
            return text

        # Sort by length descending so longer phrases match before substrings
        sorted_vocab = sorted(vocab, key=len, reverse=True)
        result = text

        for word in sorted_vocab:
            # Whole-word match, case-insensitive
            pattern = r"\b" + re.escape(word) + r"\b"
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # Clean up extra whitespace from removals
        result = re.sub(r" {2,}", " ", result)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()
