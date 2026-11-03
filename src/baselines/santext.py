"""
SanText+ (Yue et al., ACL Findings 2021) — word-level metric local differential
privacy over word embeddings. This is the DP text-sanitization family that Pang et
al. (arXiv:2410.12443) reconstruct at 72–94%; the paper's headline comparison was
against their numbers on other datasets. Here it runs on the SAME CUAD samples.

Mechanism (per token):
  * sensitive words (rank beyond `sensitive_rank` in the frequency-sorted vocabulary,
    plus every out-of-vocabulary token) are ALWAYS replaced; the replacement w' is
    sampled with P(w') ∝ exp(-ε · ||e_w − e_w'|| / 2) over the sensitive vocabulary.
    OOV tokens have no embedding, so they are replaced uniformly at random from the
    sensitive vocabulary (documented deviation; the reference code keeps them, which
    would leak every party name unchanged).
  * non-sensitive words are replaced with probability p (same mechanism over the whole
    vocabulary) and kept otherwise.
Punctuation and whitespace are preserved; replacements are lower-case.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# a period stays inside a token only between alphanumerics ("3.5", "U.S"), never trailing
TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9'\-]|\.(?=[A-Za-z0-9]))*|[^\sA-Za-z0-9]")


@dataclass
class SanTextResult:
    text: str
    n_tokens: int
    n_replaced: int
    n_oov: int


class SanTextPlus:
    def __init__(
        self,
        vocab: list[str],
        embeddings: np.ndarray,
        epsilon: float,
        p: float = 0.3,
        sensitive_rank: int = 20000,
        seed: int = 42,
    ):
        if len(vocab) != embeddings.shape[0]:
            raise ValueError("vocab and embeddings disagree in size")
        self.vocab = vocab
        self.emb = np.asarray(embeddings, dtype=np.float32)
        self.epsilon = float(epsilon)
        self.p = float(p)
        self.sensitive_rank = min(int(sensitive_rank), len(vocab))
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self._rng = np.random.default_rng(seed)
        self._sq = (self.emb * self.emb).sum(axis=1)

    # ── sampling ────────────────────────────────────────────────────────────
    def _distances(self, idx: int, lo: int, hi: int) -> np.ndarray:
        e = self.emb[idx]
        d2 = self._sq[lo:hi] - 2.0 * (self.emb[lo:hi] @ e) + float(e @ e)
        return np.sqrt(np.maximum(d2, 0.0))

    def _sample_replacement(self, idx: int, lo: int, hi: int) -> int:
        d = self._distances(idx, lo, hi)
        logits = -self.epsilon * d / 2.0
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return lo + int(self._rng.choice(hi - lo, p=probs))

    def _random_sensitive(self) -> int:
        lo = self.sensitive_rank
        hi = len(self.vocab)
        if hi <= lo:  # tiny vocab: sample from everything
            lo = 0
        return int(self._rng.integers(lo, hi))

    # ── public ──────────────────────────────────────────────────────────────
    def sanitize(self, text: str) -> SanTextResult:
        out: list[str] = []
        last = 0
        n_tok = n_rep = n_oov = 0
        V = len(self.vocab)
        for m in TOKEN_RE.finditer(text):
            out.append(text[last:m.start()])
            last = m.end()
            tok = m.group(0)
            if not re.search(r"[A-Za-z0-9]", tok):
                out.append(tok)  # punctuation
                continue
            n_tok += 1
            key = tok.lower().strip(".'")
            idx = self.word2idx.get(key)
            if idx is None:
                n_oov += 1
                n_rep += 1
                out.append(self.vocab[self._random_sensitive()])
            elif idx >= self.sensitive_rank:
                n_rep += 1
                out.append(self.vocab[self._sample_replacement(idx, self.sensitive_rank, V)])
            elif self._rng.random() < self.p:
                n_rep += 1
                out.append(self.vocab[self._sample_replacement(idx, 0, V)])
            else:
                out.append(tok)
        out.append(text[last:])
        return SanTextResult(text="".join(out), n_tokens=n_tok, n_replaced=n_rep, n_oov=n_oov)


def load_glove_vocab(max_vocab: int = 50000, model_name: str = "sentence-transformers/average_word_embeddings_glove.6B.300d"):
    """GloVe 6B-300d (frequency-sorted) through the sentence-transformers packaging."""
    from sentence_transformers import SentenceTransformer
    import torch

    st = SentenceTransformer(model_name, device="cpu")
    we = st[0]
    tok = getattr(we, "tokenizer", None)
    vocab = None
    if tok is not None:
        vocab = getattr(tok, "vocab", None)
        if vocab is None and hasattr(tok, "get_vocab"):
            vocab = tok.get_vocab()
    if vocab is None:
        raise RuntimeError("could not read the GloVe vocabulary from the sentence-transformers module")
    if isinstance(vocab, dict):  # word -> idx
        vocab = [w for w, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    weight = we.emb_layer.weight.detach().cpu().numpy() if hasattr(we, "emb_layer") else None
    if weight is None:
        with torch.no_grad():
            weight = we.embeddings_dimension  # pragma: no cover
    n = min(max_vocab, len(vocab), weight.shape[0])
    return list(vocab[:n]), np.asarray(weight[:n], dtype=np.float32)
