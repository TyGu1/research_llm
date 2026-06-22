"""
Step 3 (fast): Generate proxy questions from topic→question bank lookup.
No LLM needed. Target: <1ms per query.
"""
import random
from src.pipeline.base import ProxyQuestionGenerator

# Pre-written generic S0 questions per topic domain
QUESTION_BANK = {
    "mergers and acquisitions": [
        "What are the standard due diligence procedures in corporate acquisitions?",
        "How do valuation methodologies differ between asset purchases and stock acquisitions?",
        "What regulatory approvals are typically required for corporate mergers?",
        "What are common risk allocation mechanisms in acquisition agreements?",
        "How is post-merger integration typically structured and managed?",
        "What representations and warranties are standard in acquisition agreements?",
        "What indemnification provisions are typical in corporate acquisition transactions?",
    ],
    "antitrust and competition": [
        "How do competition authorities assess market concentration in merger reviews?",
        "What remedies are available to regulators in antitrust enforcement actions?",
        "What factors determine whether a business practice constitutes anti-competitive behavior?",
        "How are relevant markets defined for competition law analysis?",
        "What are the notification thresholds for merger control filings?",
        "How do horizontal and vertical merger analyses differ?",
    ],
    "intellectual property": [
        "What are the requirements for patentability of an invention?",
        "How do licensing agreements typically structure royalty payments?",
        "What constitutes trade secret misappropriation under current legal standards?",
        "What are the key differences between exclusive and non-exclusive licenses?",
        "How are intellectual property assets valued in commercial transactions?",
    ],
    "employment and labor": [
        "What are the legal requirements for enforceable non-compete agreements?",
        "How are executive compensation packages typically structured?",
        "What protections exist for employees during corporate restructuring?",
        "What factors determine whether a worker is an employee or independent contractor?",
        "How do severance provisions typically interact with termination clauses?",
    ],
    "real estate and leasing": [
        "What are standard commercial lease terms and escalation mechanisms?",
        "What due diligence steps are required for commercial property transactions?",
        "How do assignment and subletting provisions typically work in commercial leases?",
        "What environmental assessment requirements apply to real estate transactions?",
    ],
    "corporate governance": [
        "What fiduciary duties do corporate directors owe to shareholders?",
        "How do corporate governance codes address board independence?",
        "What are the standard mechanisms for shareholder voting and proxy solicitation?",
        "How do audit committees function within corporate governance frameworks?",
    ],
    "regulatory compliance": [
        "What elements constitute an effective corporate compliance program?",
        "How do regulatory enforcement actions typically proceed?",
        "What risk assessment methodologies are used in compliance frameworks?",
        "What are the consequences of non-compliance with industry regulations?",
    ],
    "financial transactions": [
        "What covenants are standard in commercial lending agreements?",
        "How are default and cross-default provisions structured in loan agreements?",
        "What collateral requirements are typical for secured lending transactions?",
        "How do payment terms and late payment penalties typically operate?",
    ],
    "contract terms and conditions": [
        "What are standard limitation of liability provisions in commercial contracts?",
        "How do force majeure clauses allocate risk between contracting parties?",
        "What warranty and representation provisions are typical in commercial agreements?",
        "How do indemnification provisions interact with limitation of liability clauses?",
        "What are the key principles governing contract interpretation?",
    ],
    "dispute resolution": [
        "What factors should be considered when choosing between arbitration and litigation?",
        "How are arbitration clauses typically structured in commercial agreements?",
        "What are the advantages and disadvantages of mediation as a dispute resolution mechanism?",
        "How is the enforcement of arbitral awards governed internationally?",
    ],
    "insurance and indemnification": [
        "What types of insurance coverage are typically required in commercial agreements?",
        "How do indemnification provisions allocate risk between contracting parties?",
        "What are standard insurance coverage requirements for commercial transactions?",
    ],
    "licensing and royalties": [
        "How are royalty rates and payment structures determined in licensing agreements?",
        "What are the standard terms for technology licensing arrangements?",
        "How do exclusive and non-exclusive licensing rights differ in practice?",
    ],
    "change of control": [
        "How do change of control provisions protect parties in commercial agreements?",
        "What triggers are standard in change of control clauses?",
        "How do anti-assignment provisions interact with change of control events?",
    ],
    "non-compete and non-solicitation": [
        "What factors determine the enforceability of non-compete agreements?",
        "How are non-solicitation provisions typically structured and enforced?",
        "What constitutes reasonable geographic and temporal scope for restrictive covenants?",
    ],
    "termination and expiration": [
        "What are standard termination for cause and convenience provisions?",
        "How do notice and cure periods function in contract termination?",
        "What post-termination obligations typically survive contract expiration?",
    ],
    "confidentiality and trade secrets": [
        "What elements are required for an enforceable non-disclosure agreement?",
        "How are confidential information and trade secrets legally defined and protected?",
        "What remedies are available for breach of confidentiality obligations?",
    ],
    "data privacy and security": [
        "What are the key requirements of GDPR for data processing organizations?",
        "How do data breach notification requirements vary across jurisdictions?",
        "What technical and organizational measures are required for data protection compliance?",
    ],
    "tax and fiscal matters": [
        "How do transfer pricing rules affect cross-border transactions?",
        "What tax indemnification provisions are standard in commercial agreements?",
        "How do withholding tax obligations apply to international payments?",
    ],
}

# Generic fallback questions
_FALLBACK_QUESTIONS = [
    "What are the key legal considerations for this type of commercial arrangement?",
    "How do parties typically allocate risks and responsibilities in similar agreements?",
    "What regulatory requirements apply to transactions of this nature?",
    "What dispute resolution mechanisms are most appropriate for this type of agreement?",
    "What are the standard industry practices for managing obligations under similar contracts?",
    "How do limitation of liability provisions typically function in this context?",
    "What due diligence steps are recommended for this type of transaction?",
    "What representations and warranties are standard for agreements in this area?",
]


class TemplateQuestionGenerator(ProxyQuestionGenerator):
    """Generates proxy questions by looking up topic→question bank.
    No LLM calls. Target: <1ms.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, s0_text: str, n: int = 10) -> list[str]:
        # Extract topic hints from S0 text by keyword matching
        s0_lower = s0_text.lower()
        matched_questions: list[str] = []

        for topic_key, questions in QUESTION_BANK.items():
            # Check if topic keywords appear in the S0 text
            topic_words = topic_key.lower().split()
            if any(w in s0_lower for w in topic_words if len(w) > 3):
                matched_questions.extend(questions)

        # Deduplicate preserving order
        seen = set()
        unique = []
        for q in matched_questions:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        # If not enough, add fallback questions
        if len(unique) < n:
            for q in _FALLBACK_QUESTIONS:
                if q not in seen:
                    unique.append(q)
                    seen.add(q)

        # Shuffle and take n
        self._rng.shuffle(unique)
        return unique[:n]
