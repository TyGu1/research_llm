"""
Rung c — multi-query aggregation. The attacker observes the proxy-question sets of
k queries that the same user sent about ONE contract and tries to identify the
contract (linkage) or recover its sensitive values (reconstruction).
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class QueryDraw:
    doc_title: str
    k: int
    draw: int
    sample_ids: list[str]
    question_sets: list[list[str]]

    @property
    def union_questions(self) -> list[str]:
        out: list[str] = []
        for qs in self.question_sets:
            out.extend(qs)
        return out


def group_by_contract(sample_titles: dict[str, str], questions_by_sample: dict[str, list[str]]) -> dict[str, list[str]]:
    """doc_title -> sorted sample ids that have transmitted questions."""
    by_title: dict[str, list[str]] = defaultdict(list)
    for sid, title in sample_titles.items():
        if questions_by_sample.get(sid):
            by_title[title].append(sid)
    return {t: sorted(ids) for t, ids in by_title.items()}


def make_draws(
    by_title: dict[str, list[str]],
    questions_by_sample: dict[str, list[str]],
    ks: list[int | str],
    n_draws: int = 3,
    seed: int = 42,
) -> list[QueryDraw]:
    """For each contract and k (int or "all"), draw n_draws random subsets of its queries.

    Contracts with fewer than k queries are skipped for that k (so k is exact, not "up to").
    k="all" uses every query once (a single draw).
    """
    rng = random.Random(seed)
    draws: list[QueryDraw] = []
    for title in sorted(by_title):
        ids = by_title[title]
        for k in ks:
            if k == "all":
                chosen_sets = [ids]
            else:
                if len(ids) < k:
                    continue
                chosen_sets = []
                for _ in range(n_draws):
                    chosen_sets.append(sorted(rng.sample(ids, k)))
            for d, chosen in enumerate(chosen_sets):
                draws.append(QueryDraw(
                    doc_title=title,
                    k=len(chosen),
                    draw=d,
                    sample_ids=chosen,
                    question_sets=[questions_by_sample[s] for s in chosen],
                ))
    return draws
