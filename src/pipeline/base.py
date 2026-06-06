"""
Abstract base classes and shared dataclasses for each pipeline step.
All pipeline step implementations inherit from these ABCs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopicDistribution:
    topics: dict[str, float]    # topic_name → weight (weights sum to ~1.0)

    def top_words(self, n: int = 10) -> list[str]:
        """Return the n highest-weight topic names."""
        return sorted(self.topics, key=lambda k: self.topics[k], reverse=True)[:n]


@dataclass
class S0Answer:
    question: str
    answer: str


@dataclass
class PipelineResult:
    sample_id: str
    topics: TopicDistribution | None = None
    s0_text: str = ""
    s0_text_excluded: str = ""          # after vocab exclusion (if enabled)
    generated_questions: list[str] = field(default_factory=list)
    selected_questions: list[str] = field(default_factory=list)
    s0_answers: list[S0Answer] = field(default_factory=list)
    final_answer: str = ""
    step_timings: dict[str, float] = field(default_factory=dict)
    error: str = ""                     # non-empty if pipeline failed


class TopicExtractor(ABC):
    @abstractmethod
    def extract(self, question: str, context: str) -> TopicDistribution: ...


class S0TextGenerator(ABC):
    @abstractmethod
    def generate(self, topics: TopicDistribution) -> str: ...


class VocabExcluder(ABC):
    @abstractmethod
    def exclude(self, text: str, topics: TopicDistribution) -> str: ...


class ProxyQuestionGenerator(ABC):
    @abstractmethod
    def generate(self, s0_text: str, n: int) -> list[str]: ...


class QuestionSelector(ABC):
    @abstractmethod
    def select(self, original: str, candidates: list[str], k: int) -> list[str]: ...


class CloudAnswerer(ABC):
    @abstractmethod
    def answer(self, questions: list[str]) -> list[S0Answer]: ...


class Synthesizer(ABC):
    @abstractmethod
    def synthesize(
        self,
        original_question: str,
        context: str,
        s0_answers: list[S0Answer],
    ) -> str: ...
