"""
Entity substitution baseline: regex-based entity replacement → cloud LLM → re-map.
CUAD entities follow predictable patterns (ALL-CAPS company names, dates, dollar amounts).
No external NER dependency — keeps the baseline simple and reproducible.
"""
import re
import time
from dataclasses import dataclass, field

from src.data.cuad_loader import CUADSample
from src.baselines.oracle import BaselineResult

# Entity patterns specific to CUAD contract format
ENTITY_PATTERNS: list[tuple[str, str]] = [
    # Dollar amounts → AMOUNT_N
    (r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?", "AMOUNT"),
    # Dates → DATE_N
    (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
     r"\s+\d{1,2},?\s+\d{4}\b", "DATE"),
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "DATE"),
    # Years in isolation → YEAR_N
    (r"\b(?:19|20)\d{2}\b", "YEAR"),
    # ALL CAPS company/entity names (2+ words of 2+ chars each)
    (r"\b[A-Z][A-Z\s&,\.]{4,}\b", "ENTITY"),
]

SYSTEM_PROMPT = """You are a legal expert specializing in contract analysis.
Answer questions about contracts accurately and thoroughly."""


@dataclass
class EntitySubstitutionResult(BaselineResult):
    substitution_map: dict[str, str] = field(default_factory=dict)
    substituted_question: str = ""
    substituted_context: str = ""


class EntitySubstitutionBaseline:
    def __init__(self, cloud_client, max_context_chars: int = 8000):
        self._client = cloud_client
        self._max_context_chars = max_context_chars

    def _substitute(self, text: str, entity_map: dict[str, str], counters: dict[str, int]) -> str:
        result = text
        for pattern, label in ENTITY_PATTERNS:
            for match in re.finditer(pattern, result, re.IGNORECASE):
                original = match.group(0)
                if original not in entity_map:
                    counters[label] = counters.get(label, 0) + 1
                    entity_map[original] = f"{label}_{counters[label]}"
            result = re.sub(
                pattern,
                lambda m, em=entity_map: em.get(m.group(0), m.group(0)),
                result,
                flags=re.IGNORECASE,
            )
        return result

    def _restore(self, text: str, entity_map: dict[str, str]) -> str:
        inverse = {v: k for k, v in entity_map.items()}
        result = text
        for placeholder, original in inverse.items():
            result = result.replace(placeholder, original)
        return result

    def run(self, sample: CUADSample) -> EntitySubstitutionResult:
        t0 = time.monotonic()
        entity_map: dict[str, str] = {}
        counters: dict[str, int] = {}

        sub_question = self._substitute(sample.question, entity_map, counters)
        sub_context = self._substitute(
            sample.context[:self._max_context_chars], entity_map, counters
        )

        prompt = f"Answer the following question about this contract.\n\nQuestion: {sub_question}\n\nContract:\n{sub_context}\n\nProvide a thorough answer."
        raw_answer = self._client.generate(prompt, system=SYSTEM_PROMPT)
        restored_answer = self._restore(raw_answer, entity_map)

        return EntitySubstitutionResult(
            answer=restored_answer,
            timing=time.monotonic() - t0,
            substitution_map=entity_map,
            substituted_question=sub_question,
            substituted_context=sub_context,
        )
