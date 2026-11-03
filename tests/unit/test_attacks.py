"""
Unit tests for the adaptive-attack modules (no LLM, no network).
"""
import numpy as np
import pytest

from src.attacks.leak_scan import scan
from src.attacks.scoring import svr_exact, party_hit, category_label_hit, normalise_category_guess, category_group
from src.attacks.reconstruction import LLMAttacker, ICLExample, _first_json, _excluded_word_scores
from src.attacks.linkage import EmbeddingLinker, rank_of
from src.attacks.multi_query import group_by_contract, make_draws
from src.data.contract_sampling import party_tokens, party_tokens_by_contract, sample_by_contract
from src.data.cuad_loader import CUADSample, Split
from src.pipeline.base import TopicDistribution


# ── scoring ──────────────────────────────────────────────────────────────────

def test_svr_exact_matches_paper_criterion():
    assert svr_exact("The term is two (2) years in North America.", ["two (2) years", "North America"]) == 1.0
    assert svr_exact("nothing here", ["two (2) years", "North America"]) == 0.0
    assert svr_exact("only North America", ["two (2) years", "North America"]) == 0.5
    assert svr_exact("x", []) == 0.0


def test_svr_uses_first_50_chars_case_insensitive():
    long_span = "A" * 60
    assert svr_exact(("a" * 50) + " tail", [long_span]) == 1.0


def test_party_hit_whole_word_and_none_without_tokens():
    assert party_hit("the Rubio restaurant chain", {"rubio"}) is True
    assert party_hit("rubios is not rubio", {"rubio"}) is True        # 'rubio' present as a whole word
    assert party_hit("rubios only", {"rubio"}) is False
    assert party_hit("anything", set()) is None


def test_category_label_hit_and_normalisation():
    assert category_label_hit("there is a right of first refusal here", "Rofr/Rofo/Rofn")
    assert not category_label_hit("the governing law is Delaware", "Non-Compete")
    assert normalise_category_guess("non-compete") == "Non-Compete"
    assert normalise_category_guess("Change of Control") == "Change Of Control"
    assert normalise_category_guess("something else") is None
    assert category_group("Parties") == "party" and category_group("Insurance") == "clause"


# ── leak scan ────────────────────────────────────────────────────────────────

def test_leak_scan_detects_gold_party_date_amount():
    qs = ["What did Acme Holdings agree on January 5, 2019?", "Is $2,500,000 a typical cap?"]
    r = scan(qs, ["January 5, 2019"], {"acme"})
    assert r.gold_hit and r.gold_rate == 1.0
    assert r.party_hit is True
    assert r.date_hit and r.amount_hit
    assert "Acme Holdings" in r.proper_nouns


def test_leak_scan_clean_questions():
    qs = ["What are typical notice periods for renewal?", "How is a non-compete usually scoped?"]
    r = scan(qs, ["sixty (60) days"], {"acme"})
    assert not r.gold_hit and r.party_hit is False and not r.date_hit and not r.amount_hit
    assert r.proper_nouns == []


# ── reconstruction attacks (mock client) ─────────────────────────────────────

class EchoClient:
    """Returns a canned answer; records prompts."""
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt, system="", temperature=0.0, max_tokens=None):
        self.prompts.append((system, prompt))
        return self.answer


def test_attack_observation_labels_and_prompts():
    c = EchoClient("Acme Corp, January 5, 2019")
    a = LLMAttacker(c)
    qs = ["What is a typical renewal term?"]
    assert a.direct(qs).observation == "Q"
    assert a.expert(qs).observation == "Q"
    ex = [ICLExample(["q1"], "orig?", ["gold"], "excerpt")]
    out = a.icl(qs, ex, "icl3")
    assert out.observation == "Q" and out.extra["k"] == 1 and "gold" in c.prompts[-1][1]
    assert a.category_inversion(qs).observation == "Q+"
    assert a.absence_q(qs, None).observation == "Q+"
    assert a.absence_s0("stripped text", TopicDistribution({"renewal_terms": 1.0})).observation == "S0"
    assert a.multi_query([qs, qs]).observation == "Qxk"
    assert a.dp_direct("sanitized").observation == "DP-text"
    assert a.sub_direct("ENTITY_1 text").observation == "SUB-text"


def test_category_inversion_parses_json_and_normalises():
    c = EchoClient('{"categories": ["non-compete", "Exclusivity", "Nope"], "topics": ["x"], "reconstruction": "two (2) years"}')
    out = LLMAttacker(c).category_inversion(["q"])
    assert out.extra["parsed"] is True
    assert out.extra["categories_norm"] == ["Non-Compete", "Exclusivity", None]
    assert "two (2) years" in out.reconstruction


def test_json_fallback_keeps_raw_text():
    c = EchoClient("no json here, but two (2) years")
    out = LLMAttacker(c).category_inversion(["q"])
    assert out.extra["parsed"] is False and "two (2) years" in out.reconstruction
    assert _first_json("prefix {\"a\": 1,} suffix") == {"a": 1}


def test_excluded_word_scores():
    topics = TopicDistribution({"renewal terms": 0.5, "notice period": 0.5})
    s = _excluded_word_scores(["renewal", "notice", "banana"], topics)
    assert s["excl_truth_n"] == 4 and s["excl_recall"] == 0.5 and abs(s["excl_precision"] - 2 / 3) < 1e-9
    assert _excluded_word_scores(None, topics)["excl_recall"] == 0.0
    assert _excluded_word_scores(["x"], None) == {}


# ── linkage ──────────────────────────────────────────────────────────────────

class ToyEmbedder:
    """Deterministic bag-of-words embedding so similarity tests are exact."""
    VOCAB = ["renewal", "notice", "compete", "license", "insurance", "escrow"]

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        out = []
        for t in texts:
            v = np.array([t.lower().count(w) for w in self.VOCAB], dtype=float) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def test_embedding_linker_ranks_true_candidate_first():
    linker = EmbeddingLinker(ToyEmbedder(), chunk_chars=20, max_chunks=3)
    idx = linker.build_index({"A": "renewal renewal notice", "B": "compete compete license", "C": "insurance escrow"})
    ranked = linker.rank(["What is a renewal notice?"], idx)
    assert ranked[0][0] == "A" and rank_of("A", ranked) == 1 and rank_of("C", ranked) == 3
    assert rank_of("missing", ranked) == 4


# ── multi-query draws ────────────────────────────────────────────────────────

def test_make_draws_exact_k_and_all():
    titles = {"s1": "T1", "s2": "T1", "s3": "T1", "s4": "T2"}
    qs = {"s1": ["a"], "s2": ["b"], "s3": ["c"], "s4": ["d"]}
    by_title = group_by_contract(titles, qs)
    draws = make_draws(by_title, qs, ks=[1, 3, "all"], n_draws=2, seed=1)
    t1 = [d for d in draws if d.doc_title == "T1"]
    t2 = [d for d in draws if d.doc_title == "T2"]
    assert sorted({d.k for d in t1}) == [1, 3]           # k=3 and "all" (=3) both present
    assert all(d.k == 1 for d in t2)                       # T2 has one query: k=3 skipped, "all" is k=1
    assert [d.union_questions for d in t1 if d.k == 3][0] == ["a", "b", "c"]


# ── contract sampling ────────────────────────────────────────────────────────

def _mk(i, title, cat="Non-Compete", answers=("two (2) years",), split=Split.DEV):
    return CUADSample(id=f"id{i}", question="q", raw_question="", context="ctx", answers=list(answers),
                      category=cat, task_type="framework_reasoning", doc_title=title, is_impossible=False, split=split)


def test_sample_by_contract_takes_all_questions_of_chosen_contracts():
    samples = [_mk(i, "A") for i in range(6)] + [_mk(10 + i, "B") for i in range(2)] + [_mk(20 + i, "C") for i in range(5)]
    out = sample_by_contract(samples, n_contracts=1, seed=0, min_per_contract=5)
    titles = {s.doc_title for s in out}
    assert len(titles) == 1 and titles <= {"A", "C"}
    assert len(out) in (5, 6)
    out2 = sample_by_contract(samples, n_contracts=5, seed=0, min_per_contract=5)
    assert {s.doc_title for s in out2} == {"A", "C"}   # B has < 5 questions


def test_party_tokens_drop_generic_words():
    toks = party_tokens(["Acme Holdings Inc. (the Company)", "Rubio's Restaurants, Inc."])
    assert "acme" in toks and "rubio" in toks and "restaurants" in toks
    assert "holdings" not in toks and "company" not in toks
    by = party_tokens_by_contract([_mk(1, "A", cat="Parties", answers=("Zorblax Corp",)), _mk(2, "A")])
    assert by == {"A": {"zorblax"}}
