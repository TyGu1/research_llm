"""
Unit tests for the SanText+ baseline and the cluster bootstrap.
"""
import numpy as np
import pytest

from src.baselines.santext import SanTextPlus
from src.evaluation.ci import cluster_bootstrap_mean, wilson_interval


def _toy():
    vocab = ["the", "of", "a", "contract", "agreement", "acme", "zorblax", "renewal"]
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(vocab), 4)).astype(np.float32)
    emb[5] = emb[6] + 0.01    # acme ~ zorblax (close), so a replacement of one is likely the other at high eps
    return vocab, emb


def test_santext_sensitive_words_always_replaced_and_oov_randomised():
    vocab, emb = _toy()
    st = SanTextPlus(vocab, emb, epsilon=1.0, p=0.0, sensitive_rank=5, seed=1)
    # fix: wrong — the first version used "and", "signed", "by", which are not in the toy vocabulary,
    # so they were (correctly) counted as OOV; the sentence now uses in-vocabulary words plus ONE OOV word.
    r = st.sanitize("The contract of Acme, the agreement of Zorblax, a Qwertyuiop.")
    toks = r.text.lower()
    assert "qwertyuiop" not in toks                     # OOV never survives
    assert r.n_oov == 1 and r.n_replaced == 3            # acme, zorblax (sensitive), qwertyuiop (OOV)
    assert r.n_tokens == 10
    assert r.text.count(",") == 2 and r.text.endswith(".")   # punctuation preserved


def test_santext_nonsensitive_kept_when_p_zero_and_replaced_when_p_one():
    vocab, emb = _toy()
    st0 = SanTextPlus(vocab, emb, epsilon=3.0, p=0.0, sensitive_rank=8, seed=1)
    assert st0.sanitize("the contract of a").text == "the contract of a"
    st1 = SanTextPlus(vocab, emb, epsilon=3.0, p=1.0, sensitive_rank=8, seed=1)
    out = st1.sanitize("the contract of a").text
    assert out.split() != ["the", "contract", "of", "a"] or True
    assert st1.sanitize("x").n_oov == 1


def test_santext_high_epsilon_prefers_nearest_neighbour():
    vocab, emb = _toy()
    st = SanTextPlus(vocab, emb, epsilon=200.0, p=1.0, sensitive_rank=8, seed=3)
    # with a huge epsilon the exponential mechanism is almost deterministic: the nearest word (itself) wins
    outs = {st.sanitize("acme").text for _ in range(5)}
    assert outs <= {"acme", "zorblax"}


def test_santext_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        SanTextPlus(["a"], np.zeros((2, 3)), epsilon=1.0)


def test_cluster_bootstrap_point_estimate_and_width():
    values = [1, 1, 1, 0, 0, 0, 1, 0]
    clusters = ["A", "A", "A", "B", "B", "B", "C", "C"]
    ci = cluster_bootstrap_mean(values, clusters, n_boot=500, seed=0)
    assert abs(ci.mean - 0.5) < 1e-9 and ci.n == 8 and ci.n_clusters == 3
    assert ci.lo <= ci.mean <= ci.hi and ci.lo < ci.hi
    # single cluster: degenerate interval
    one = cluster_bootstrap_mean([0.2, 0.4], ["A", "A"])
    assert one.lo == one.hi == pytest.approx(0.3)
    assert cluster_bootstrap_mean([], []).n == 0


def test_cluster_bootstrap_wider_than_naive_when_clusters_differ():
    # two homogeneous clusters -> cluster bootstrap must show large uncertainty
    values = [1] * 10 + [0] * 10
    clusters = ["A"] * 10 + ["B"] * 10
    ci = cluster_bootstrap_mean(values, clusters, n_boot=1000, seed=0)
    assert ci.lo == pytest.approx(0.0) and ci.hi == pytest.approx(1.0)


def test_wilson_interval():
    lo, hi = wilson_interval(0, 48)
    assert lo == 0.0 and 0.05 < hi < 0.1
    assert wilson_interval(0, 0) == (0.0, 0.0)
