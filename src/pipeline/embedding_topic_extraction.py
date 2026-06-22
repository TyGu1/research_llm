"""
Step 1 (fast, general-purpose): Topic extraction via embedding similarity.
Encodes the query once, cosine-similarity against pre-embedded topic descriptions.
Single forward pass, ~10ms on CPU. Fully general — swap topic descriptions for any domain.
"""
import numpy as np
from src.pipeline.base import TopicDistribution, TopicExtractor

# Default topic descriptions — general enough for legal/business, but replaceable
DEFAULT_TOPIC_DESCRIPTIONS = {
    "mergers and acquisitions": "corporate acquisitions, mergers, buyouts, deal structuring, due diligence, target valuation",
    "antitrust and competition": "competition law, market concentration, antitrust review, monopoly, cartel, market dominance",
    "intellectual property": "patents, trademarks, copyrights, trade secrets, IP licensing, invention protection",
    "employment and labor": "employment contracts, hiring, termination, compensation, workforce, employee rights",
    "real estate and leasing": "commercial leases, property transactions, rent, tenant, landlord, premises",
    "corporate governance": "board of directors, shareholders, voting rights, fiduciary duties, governance framework",
    "regulatory compliance": "regulatory requirements, compliance programs, enforcement, audits, reporting obligations",
    "data privacy and security": "personal data protection, GDPR, privacy policy, data breach, security measures",
    "financial transactions": "loans, payments, interest rates, debt, credit, collateral, financial covenants",
    "contract terms and conditions": "contractual obligations, warranties, representations, liability, force majeure, terms",
    "dispute resolution": "arbitration, litigation, mediation, court proceedings, dispute settlement, claims",
    "insurance and indemnification": "insurance coverage, indemnification, liability protection, risk transfer",
    "licensing and royalties": "license agreements, royalty payments, sublicensing, exclusivity, technology licensing",
    "change of control": "ownership transfer, change of control provisions, assignment, successor rights",
    "non-compete and non-solicitation": "non-compete agreements, restrictive covenants, non-solicitation, competition restrictions",
    "termination and expiration": "contract termination, expiration, renewal, notice period, cure period",
    "confidentiality and trade secrets": "confidential information, NDA, non-disclosure, proprietary information, secrecy",
    "tax and fiscal matters": "tax obligations, withholding, transfer pricing, tax indemnification, fiscal compliance",
    "supply chain and procurement": "procurement contracts, supply agreements, vendor management, sourcing, delivery terms",
    "environmental and sustainability": "environmental regulations, sustainability requirements, emissions, environmental impact",
}


class EmbeddingTopicExtractor(TopicExtractor):
    """Extract topics by embedding similarity. One forward pass per query.

    Args:
        model_name: sentence-transformers model to use
        topic_descriptions: dict of topic_name → description text. Replace for new domains.
        top_k: number of topics to return
        min_score: minimum cosine similarity to include a topic
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        topic_descriptions: dict[str, str] | None = None,
        top_k: int = 5,
        min_score: float = 0.1,
    ):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._top_k = top_k
        self._min_score = min_score

        # Pre-embed all topic descriptions (done once at init)
        descriptions = topic_descriptions or DEFAULT_TOPIC_DESCRIPTIONS
        self._topic_names = list(descriptions.keys())
        self._topic_embeddings = self._model.encode(
            list(descriptions.values()), normalize_embeddings=True
        )

    def extract(self, question: str, context: str) -> TopicDistribution:
        # Encode query + first 500 chars of context (single forward pass)
        text = f"{question} {context[:500]}"
        query_emb = self._model.encode([text], normalize_embeddings=True)[0]

        # Cosine similarity against all topic embeddings
        scores = np.dot(self._topic_embeddings, query_emb)

        # Take top_k above min_score
        top_indices = np.argsort(scores)[::-1][:self._top_k]
        topics = {}
        for idx in top_indices:
            score = float(scores[idx])
            if score >= self._min_score:
                topics[self._topic_names[idx]] = score

        if not topics:
            topics = {self._topic_names[top_indices[0]]: 1.0}

        # Normalize to sum to 1
        total = sum(topics.values())
        topics = {k: v / total for k, v in topics.items()}

        return TopicDistribution(topics=topics)
