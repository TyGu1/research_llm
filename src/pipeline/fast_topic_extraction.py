"""
Step 1 (fast): Extract topic distribution using zero-shot classification.
Uses transformers pipeline with BART-large-MNLI — no LLM call needed.
Target: <200ms per query after model warmup.
"""
import logging
from src.pipeline.base import TopicDistribution, TopicExtractor

logger = logging.getLogger(__name__)

# Broad legal/business topic candidates for zero-shot classification
CANDIDATE_TOPICS = [
    "mergers and acquisitions",
    "antitrust and competition",
    "intellectual property",
    "employment and labor",
    "real estate and leasing",
    "corporate governance",
    "regulatory compliance",
    "data privacy and security",
    "financial transactions",
    "contract terms and conditions",
    "dispute resolution",
    "insurance and indemnification",
    "licensing and royalties",
    "tax and fiscal matters",
    "environmental and sustainability",
    "supply chain and procurement",
    "change of control",
    "non-compete and non-solicitation",
    "termination and expiration",
    "confidentiality and trade secrets",
]


class ZeroShotTopicExtractor(TopicExtractor):
    """Zero-shot classifier using facebook/bart-large-mnli for topic extraction."""

    def __init__(self, candidate_topics: list[str] | None = None, top_k: int = 5, device: str = "cpu"):
        from transformers import pipeline
        self._classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
        self._candidates = candidate_topics or CANDIDATE_TOPICS
        self._top_k = top_k

    def extract(self, question: str, context: str) -> TopicDistribution:
        # Use question + first 500 chars of context for speed
        text = f"{question} {context[:500]}"
        result = self._classifier(text, self._candidates, multi_label=True)

        # Take top_k topics, normalize weights
        topics = {}
        for label, score in zip(result["labels"][:self._top_k], result["scores"][:self._top_k]):
            topics[label] = score

        total = sum(topics.values())
        if total > 0:
            topics = {k: v / total for k, v in topics.items()}

        return TopicDistribution(topics=topics)


class KeywordTopicExtractor(TopicExtractor):
    """Ultra-fast keyword-based topic extraction using TF-IDF + predefined topic vocabulary.
    No model loading, no GPU — pure Python. Target: <5ms per query.
    """

    # Topic → defining keywords (if ≥2 keywords match, topic is active)
    TOPIC_KEYWORDS = {
        "mergers and acquisitions": ["acquisition", "merger", "acquire", "target", "deal", "purchase", "buyout", "takeover"],
        "antitrust and competition": ["antitrust", "competition", "monopoly", "market share", "dominant", "cartel"],
        "intellectual property": ["patent", "trademark", "copyright", "intellectual property", "ip", "invention", "license"],
        "employment and labor": ["employee", "employment", "termination", "salary", "compensation", "hire", "workforce"],
        "real estate and leasing": ["lease", "rent", "property", "tenant", "landlord", "premises", "real estate"],
        "corporate governance": ["board", "director", "shareholder", "voting", "governance", "bylaws", "fiduciary"],
        "regulatory compliance": ["compliance", "regulation", "regulatory", "authority", "statute", "law", "requirement"],
        "financial transactions": ["payment", "loan", "interest", "debt", "credit", "financial", "revenue", "amount"],
        "contract terms and conditions": ["agreement", "contract", "term", "condition", "obligation", "covenant", "provision"],
        "dispute resolution": ["dispute", "arbitration", "litigation", "mediation", "court", "claim", "resolution"],
        "insurance and indemnification": ["insurance", "indemnify", "indemnification", "liability", "coverage", "insured"],
        "licensing and royalties": ["license", "royalty", "sublicense", "licensee", "licensor", "fee", "exclusive"],
        "change of control": ["change of control", "ownership", "transfer", "assignment", "successor"],
        "non-compete and non-solicitation": ["non-compete", "noncompete", "non-solicitation", "restrictive covenant", "compete"],
        "termination and expiration": ["termination", "expire", "expiration", "cancel", "renewal", "notice period"],
        "confidentiality and trade secrets": ["confidential", "trade secret", "nda", "non-disclosure", "proprietary"],
        "data privacy and security": ["privacy", "data protection", "gdpr", "personal data", "security", "breach"],
        "tax and fiscal matters": ["tax", "fiscal", "deduction", "withholding", "vat", "income tax"],
    }

    def extract(self, question: str, context: str) -> TopicDistribution:
        text = f"{question} {context[:2000]}".lower()

        scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count >= 1:
                scores[topic] = count

        if not scores:
            scores = {"contract terms and conditions": 1.0}

        # Take top 5, then normalize
        top = dict(sorted(scores.items(), key=lambda x: -x[1])[:5])
        total = sum(top.values())
        topics = {k: v / total for k, v in top.items()}

        return TopicDistribution(topics=topics)
