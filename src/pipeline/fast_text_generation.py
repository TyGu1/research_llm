"""
Step 2 (fast): Generate S0 text using templates — no LLM needed.
Maps each topic to a pre-written paragraph, concatenates by weight.
Target: <1ms per query.
"""
from src.pipeline.base import TopicDistribution, S0TextGenerator

# Pre-written generic S0 paragraphs per topic domain
TOPIC_TEMPLATES = {
    "mergers and acquisitions": (
        "Corporate acquisitions involve complex due diligence processes that evaluate "
        "target companies across financial, legal, and operational dimensions. Key "
        "considerations include valuation methodologies such as discounted cash flow and "
        "comparable company analysis, regulatory approval requirements, integration "
        "planning, and risk allocation between buyer and seller. Transaction structures "
        "vary from asset purchases to stock acquisitions, each carrying distinct tax and "
        "liability implications."
    ),
    "antitrust and competition": (
        "Competition law governs market behavior to prevent anti-competitive practices. "
        "Regulators evaluate market concentration using metrics like the Herfindahl-Hirschman "
        "Index, assess barriers to entry, and scrutinize horizontal and vertical agreements. "
        "Merger control regimes require pre-notification for transactions exceeding certain "
        "thresholds, with remedies ranging from behavioral commitments to structural "
        "divestitures."
    ),
    "intellectual property": (
        "Intellectual property protection encompasses patents, trademarks, copyrights, and "
        "trade secrets. Patent prosecution requires demonstrating novelty, non-obviousness, "
        "and utility. Trademark registration protects distinctive marks in commerce. "
        "Licensing agreements define scope, territory, exclusivity, and royalty structures "
        "for IP commercialization."
    ),
    "employment and labor": (
        "Employment relationships are governed by statutory requirements covering minimum "
        "wage, working hours, discrimination protections, and termination procedures. "
        "Executive compensation packages typically include base salary, performance bonuses, "
        "equity grants, and severance provisions. Non-compete and non-solicitation "
        "restrictions must be reasonable in scope, duration, and geography to be enforceable."
    ),
    "real estate and leasing": (
        "Commercial lease agreements establish the terms of property use, including rent "
        "escalation mechanisms, maintenance obligations, permitted use restrictions, and "
        "assignment and subletting rights. Due diligence involves title searches, "
        "environmental assessments, zoning verification, and structural inspections."
    ),
    "corporate governance": (
        "Corporate governance frameworks define the relationships among shareholders, "
        "boards of directors, and management. Key mechanisms include voting rights, "
        "fiduciary duties of care and loyalty, audit committee oversight, and disclosure "
        "requirements. Board composition, independence standards, and executive succession "
        "planning are critical governance considerations."
    ),
    "regulatory compliance": (
        "Regulatory compliance requires organizations to adhere to applicable laws, "
        "regulations, and industry standards. Compliance programs typically include risk "
        "assessments, policies and procedures, training, monitoring, and reporting mechanisms. "
        "Enforcement actions may result in fines, injunctions, or operational restrictions."
    ),
    "financial transactions": (
        "Financial transactions encompass lending, borrowing, payment processing, and "
        "capital markets activities. Key considerations include interest rate structures, "
        "collateral requirements, covenant compliance, default triggers, and cross-default "
        "provisions. Payment terms specify timing, currency, methods, and late payment "
        "penalties."
    ),
    "contract terms and conditions": (
        "Contractual agreements define the rights and obligations of parties through terms "
        "covering scope of work, payment, warranties, representations, liability limitations, "
        "indemnification, force majeure, and dispute resolution. Interpretation principles "
        "include the parol evidence rule, contra proferentem, and the duty to mitigate damages."
    ),
    "dispute resolution": (
        "Dispute resolution mechanisms range from negotiation and mediation to binding "
        "arbitration and litigation. Arbitration clauses specify institutional rules, "
        "seat of arbitration, language, number of arbitrators, and governing law. "
        "Court selection clauses designate jurisdiction and venue for judicial proceedings."
    ),
    "insurance and indemnification": (
        "Indemnification provisions allocate risk between parties by requiring one party "
        "to compensate another for specified losses. Insurance requirements mandate "
        "minimum coverage levels, named insured parties, and policy types including general "
        "liability, professional liability, and property insurance."
    ),
    "licensing and royalties": (
        "Licensing agreements grant rights to use intellectual property, technology, or "
        "other assets under specified conditions. Key terms include exclusivity, territory, "
        "field of use, sublicensing rights, royalty rates and calculation methods, minimum "
        "guarantees, audit rights, and termination triggers."
    ),
    "change of control": (
        "Change of control provisions address the consequences of ownership transitions, "
        "including consent requirements, acceleration of obligations, put and call options, "
        "and termination rights. Anti-assignment clauses may restrict transfers triggered "
        "by change of control events."
    ),
    "non-compete and non-solicitation": (
        "Restrictive covenants including non-compete and non-solicitation agreements "
        "protect legitimate business interests such as trade secrets, customer relationships, "
        "and workforce stability. Enforceability depends on reasonableness of geographic "
        "scope, duration, and activity restrictions."
    ),
    "termination and expiration": (
        "Contract termination provisions define the circumstances and procedures for "
        "ending an agreement, including termination for cause, termination for convenience, "
        "notice periods, cure periods, and post-termination obligations such as return of "
        "materials and survival of certain provisions."
    ),
    "confidentiality and trade secrets": (
        "Confidentiality obligations protect proprietary information shared between parties. "
        "Non-disclosure agreements define the scope of confidential information, permitted "
        "disclosures, duration of obligations, and remedies for breach including injunctive "
        "relief and monetary damages."
    ),
    "data privacy and security": (
        "Data protection regulations such as GDPR and CCPA establish requirements for "
        "collecting, processing, storing, and transferring personal data. Compliance "
        "requires implementing appropriate technical and organizational measures, conducting "
        "privacy impact assessments, and establishing data breach notification procedures."
    ),
    "tax and fiscal matters": (
        "Tax considerations in commercial transactions include income tax implications, "
        "withholding tax obligations, transfer pricing compliance, indirect tax treatment, "
        "and tax indemnification provisions. Structuring decisions may be influenced by "
        "jurisdiction-specific tax incentives and treaty networks."
    ),
}

# Generic fallback for unmatched topics
_FALLBACK = (
    "Legal and business considerations in this area involve careful analysis of "
    "applicable laws, regulations, contractual obligations, and industry standards. "
    "Parties must evaluate risks, allocate responsibilities, and establish clear "
    "procedures for compliance and dispute resolution."
)


class TemplateTextGenerator(S0TextGenerator):
    """Generates S0 text by concatenating pre-written paragraphs per topic.
    Zero LLM calls. Target: <1ms.
    """

    def generate(self, topics: TopicDistribution) -> str:
        paragraphs = []
        for topic in sorted(topics.topics, key=lambda t: -topics.topics[t]):
            # Fuzzy match: find template whose key appears in the topic name
            matched = False
            for key, template in TOPIC_TEMPLATES.items():
                if key in topic.lower() or topic.lower() in key:
                    paragraphs.append(template)
                    matched = True
                    break
            if not matched:
                # Try partial keyword match
                topic_words = set(topic.lower().split())
                for key, template in TOPIC_TEMPLATES.items():
                    key_words = set(key.lower().split())
                    if topic_words & key_words:
                        paragraphs.append(template)
                        matched = True
                        break
            if not matched:
                paragraphs.append(_FALLBACK)

        # Take top 3-4 paragraphs (matching num topics), aim for 200-300 words
        text = " ".join(paragraphs[:4])

        # Trim to ~300 words if too long
        words = text.split()
        if len(words) > 300:
            text = " ".join(words[:300]) + "."

        return text
