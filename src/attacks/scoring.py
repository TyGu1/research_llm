"""
Scoring of attacker reconstructions.

Primary metric stays the paper's exact-substring SVR so numbers are comparable.
Added: party-name recovery, full category-label mention, situation score, and a
typed view (by category group: party / date / name / clause text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np

from src.data.cuad_loader import CUADSample, QUESTION_TEMPLATES

CATEGORY_GROUP: dict[str, str] = {
    "Parties": "party",
    "Agreement Date": "date",
    "Effective Date": "date",
    "Expiration Date": "date",
    "Document Name": "name",
}

# Phrases that count as naming the category (lower-case). Default: the label itself.
CATEGORY_PHRASES: dict[str, list[str]] = {
    "Rofr/Rofo/Rofn": ["right of first refusal", "right of first offer", "right of first negotiation", "rofr", "rofo", "rofn"],
    "Ip Ownership Assignment": ["ip ownership", "intellectual property ownership", "ownership of intellectual property", "ip assignment"],
    "Joint Ip Ownership": ["joint ip", "joint ownership", "jointly own"],
    "Unlimited/All-You-Can-Eat-License": ["unlimited license", "all-you-can-eat", "all you can eat"],
    "Irrevocable Or Perpetual License": ["irrevocable", "perpetual license", "perpetual licence"],
    "Affiliate License-Licensor": ["affiliate", "licensor's affiliates"],
    "Affiliate License-Licensee": ["affiliate", "licensee's affiliates"],
    "Notice Period To Terminate Renewal": ["notice period", "notice to terminate", "non-renewal notice", "notice of non-renewal"],
    "No-Solicit Of Customers": ["solicit", "non-solicitation"],
    "No-Solicit Of Employees": ["solicit", "non-solicitation"],
    "Cap On Liability": ["cap on liability", "liability cap", "limitation of liability", "limit on liability"],
    "Uncapped Liability": ["uncapped liability", "unlimited liability", "no cap on liability"],
    "Covenant Not To Sue": ["covenant not to sue"],
    "Termination For Convenience": ["termination for convenience", "terminate for convenience"],
    "Post-Termination Services": ["post-termination", "after termination", "transition services"],
    "Competitive Restriction Exception": ["exception", "carve-out", "carve out"],
    "Change Of Control": ["change of control", "change in control"],
    "Most Favored Nation": ["most favored nation", "most favoured nation", "mfn"],
    "Source Code Escrow": ["escrow"],
    "Third Party Beneficiary": ["third party beneficiary", "third-party beneficiary"],
    "Revenue/Profit Sharing": ["revenue sharing", "profit sharing", "revenue share", "profit share"],
    "Volume Restriction": ["volume restriction", "volume limit", "minimum volume", "maximum volume"],
    "Minimum Commitment": ["minimum commitment", "minimum purchase", "minimum order"],
    "License Grant": ["license grant", "grant of license", "grants a license", "licence grant"],
    "Non-Transferable License": ["non-transferable", "nontransferable", "not transferable", "sublicens"],
    "Governing Law": ["governing law", "governed by the laws", "laws of the state"],
    "Price Restrictions": ["price restriction", "pricing restriction", "price control", "resale price"],
    "Anti-Assignment": ["anti-assignment", "assignment", "assign"],
    "Warranty Duration": ["warranty"],
    "Non-Disparagement": ["non-disparagement", "disparag"],
    "Audit Rights": ["audit"],
    "Liquidated Damages": ["liquidated damages"],
    "Insurance": ["insurance"],
    "Exclusivity": ["exclusiv"],
    "Non-Compete": ["non-compete", "noncompete", "not compete", "non-competition"],
    "Renewal Term": ["renewal", "renew"],
    "Agreement Date": ["agreement date", "date of the agreement", "dated"],
    "Effective Date": ["effective date"],
    "Expiration Date": ["expiration date", "expires", "expiry", "term of the agreement"],
    "Document Name": ["agreement", "contract"],
    "Parties": ["parties", "between"],
}

ALL_CATEGORIES: list[str] = list(QUESTION_TEMPLATES.keys())


def category_group(category: str) -> str:
    return CATEGORY_GROUP.get(category, "clause")


def svr_exact(reconstruction: str, answers: list[str]) -> float:
    """Paper's criterion: fraction of gold spans whose first 50 chars occur (case-insensitive)."""
    if not answers:
        return 0.0
    r = reconstruction.lower()
    found = sum(1 for a in answers if a.lower()[:50] in r)
    return found / len(answers)


def party_hit(text: str, tokens: set[str]) -> bool | None:
    """True if any distinctive party token appears as a whole word. None when no tokens are known."""
    if not tokens:
        return None
    words = set(re.findall(r"[a-z][a-z\-]{3,}", text.lower()))   # same tokenisation as party_tokens
    return any(t in words for t in tokens)


def category_label_hit(text: str, category: str) -> bool:
    t = text.lower()
    phrases = CATEGORY_PHRASES.get(category)
    if phrases is None:
        phrases = [category.lower().replace("/", " ").replace("-", " ")]
    return any(p in t for p in phrases)


def normalise_category_guess(guess: str) -> str | None:
    """Map a free-text category guess to one of the 41 CUAD labels (exact, case/punct-insensitive)."""
    g = re.sub(r"[^a-z0-9]+", " ", guess.lower()).strip()
    for c in ALL_CATEGORIES:
        if re.sub(r"[^a-z0-9]+", " ", c.lower()).strip() == g:
            return c
    return None


@dataclass
class ReconstructionScore:
    svr_exact: float
    n_gold: int
    any_gold: bool
    party_hit: bool | None
    category_label_hit: bool
    situation_score: float
    group: str

    def to_dict(self) -> dict:
        return asdict(self)


class Scorer:
    def __init__(self, embed_model):
        self._embed = embed_model

    def situation(self, reconstruction: str, context: str) -> float:
        if not reconstruction.strip():
            return 0.0
        embs = self._embed.encode([reconstruction, context[:500]], normalize_embeddings=True)
        return max(0.0, float(np.dot(embs[0], embs[1])))

    def score(self, reconstruction: str, sample: CUADSample, tokens: set[str]) -> ReconstructionScore:
        svr = svr_exact(reconstruction, sample.answers)
        return ReconstructionScore(
            svr_exact=svr,
            n_gold=len(sample.answers),
            any_gold=svr > 0,
            party_hit=party_hit(reconstruction, tokens),
            category_label_hit=category_label_hit(reconstruction, sample.category),
            situation_score=self.situation(reconstruction, sample.context),
            group=category_group(sample.category),
        )
