"""
CUAD dataset loader.

CUAD uses a SQuAD-format JSON (nested: data → paragraphs → qas). The HuggingFace
datasets library cannot load it with load_dataset('cuad') on datasets v4.8+ (loading
scripts are deprecated). We load the raw JSON directly and flatten it.

CUAD questions are all "Highlight the parts (if any) of this contract related to [X]
that should be reviewed by a lawyer." This is span-extraction, not reasoning. We rewrite
each question into a natural reasoning form using category-specific templates before
passing it to the pipeline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datasets import load_dataset

# Maps raw CUAD question prefixes to natural question templates.
# Key: substring that identifies the category in the raw question.
# Value: natural question template (used as the rewritten question).
QUESTION_TEMPLATES: dict[str, str] = {
    "Document Name": "What is the official name of this contract document?",
    "Parties": "Who are the parties entering into this contract?",
    "Agreement Date": "When was this contract signed or agreed upon?",
    "Effective Date": "When does this contract take effect?",
    "Expiration Date": "When does this contract expire or end?",
    "Renewal Term": "What are the renewal terms for this contract, if any?",
    "Notice Period To Terminate Renewal": "How much notice is required to terminate or prevent renewal of this contract?",
    "Governing Law": "Which jurisdiction's laws govern this contract?",
    "Most Favored Nation": "Does this contract include a most-favored-nation clause, and if so, what does it require?",
    "Non-Compete": "What non-compete restrictions does this contract impose on the parties?",
    "Exclusivity": "Does this contract grant any exclusivity rights, and to whom?",
    "No-Solicit Of Customers": "Does this contract prohibit soliciting the other party's customers?",
    "No-Solicit Of Employees": "Does this contract prohibit soliciting the other party's employees?",
    "Non-Disparagement": "Does this contract include non-disparagement obligations?",
    "Termination For Convenience": "Under what conditions can either party terminate this contract for convenience?",
    "Rofr/Rofo/Rofn": "Does this contract include any right of first refusal, first offer, or first negotiation?",
    "Change Of Control": "What happens to this contract in the event of a change of control of one of the parties?",
    "Anti-Assignment": "Does this contract restrict either party from assigning their rights or obligations?",
    "Revenue/Profit Sharing": "Does this contract include revenue or profit sharing provisions?",
    "Price Restrictions": "Does this contract impose any price restrictions or price controls?",
    "Minimum Commitment": "Does this contract require either party to meet a minimum purchase or service commitment?",
    "Volume Restriction": "Does this contract impose any volume restrictions on the parties?",
    "Ip Ownership Assignment": "Who owns intellectual property created under this contract?",
    "Joint Ip Ownership": "Does this contract provide for joint ownership of intellectual property?",
    "License Grant": "What licenses does this contract grant, and to whom?",
    "Non-Transferable License": "Is the license granted in this contract transferable or sublicensable?",
    "Affiliate License-Licensor": "Can the licensor extend the license to its affiliates under this contract?",
    "Affiliate License-Licensee": "Can the licensee extend the license to its affiliates under this contract?",
    "Unlimited/All-You-Can-Eat-License": "Does this contract include an unlimited or all-you-can-eat license?",
    "Irrevocable Or Perpetual License": "Is the license granted in this contract irrevocable or perpetual?",
    "Source Code Escrow": "Does this contract include a source code escrow arrangement?",
    "Post-Termination Services": "What services or obligations continue after this contract is terminated?",
    "Audit Rights": "Does this contract grant either party the right to audit the other?",
    "Uncapped Liability": "Are there any uncapped liability provisions in this contract?",
    "Cap On Liability": "What caps on liability does this contract impose?",
    "Liquidated Damages": "Does this contract include liquidated damages provisions?",
    "Warranty Duration": "What is the duration of the warranty provided in this contract?",
    "Insurance": "What insurance requirements does this contract impose?",
    "Covenant Not To Sue": "Does this contract include a covenant not to sue?",
    "Third Party Beneficiary": "Does this contract name any third party beneficiaries?",
    "Competitive Restriction Exception": "Are there any exceptions to the competitive restrictions in this contract?",
}


def _get_template(raw_question: str) -> str:
    """Match a raw CUAD question to its natural language template."""
    for key, template in QUESTION_TEMPLATES.items():
        if key.lower() in raw_question.lower():
            return template
    # Fallback: strip the boilerplate prefix/suffix
    q = raw_question.replace(
        "Highlight the parts (if any) of this contract related to \"", ""
    ).replace(
        "\" that should be reviewed by a lawyer. Details: ", " "
    ).strip()
    return f"What does this contract say about: {q}?"


def _extract_category(raw_question: str) -> str:
    """Extract the category name from a raw CUAD question."""
    for key in QUESTION_TEMPLATES:
        if key.lower() in raw_question.lower():
            return key
    return "Other"


@dataclass
class CUADSample:
    id: str
    question: str           # rewritten natural question (used in pipeline)
    raw_question: str       # original CUAD span-extraction question
    context: str            # contract text
    answers: list[str]      # gold answer spans (from CUAD)
    category: str           # CUAD clause category (e.g. "Non-Compete")
    task_type: str          # mapped task type (from question_taxonomy)
    doc_title: str
    is_impossible: bool


class CUADDataset:
    HF_PATH = "theatticusproject/cuad"
    JSON_FILE = "CUAD_v1/CUAD_v1.json"

    def __init__(self) -> None:
        self._samples: list[CUADSample] | None = None

    def load(self) -> list[CUADSample]:
        if self._samples is not None:
            return self._samples

        # Load the raw SQuAD-format JSON from HuggingFace
        ds = load_dataset(
            "json",
            data_files=f"hf://datasets/{self.HF_PATH}/{self.JSON_FILE}",
            split="train",
        )
        # ds has a single row with field "data" containing the SQuAD structure
        squad_data = ds[0]["data"]  # list of {title, paragraphs: [...]}

        from src.data.question_taxonomy import get_task_type

        samples: list[CUADSample] = []
        for doc in squad_data:
            title = doc["title"]
            for para in doc["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    raw_q = qa["question"]
                    category = _extract_category(raw_q)
                    answers = [a["text"] for a in qa.get("answers", [])]
                    samples.append(CUADSample(
                        id=qa["id"],
                        question=_get_template(raw_q),
                        raw_question=raw_q,
                        context=context,
                        answers=answers,
                        category=category,
                        task_type=get_task_type(category),
                        doc_title=title,
                        is_impossible=qa.get("is_impossible", len(answers) == 0),
                    ))

        self._samples = samples
        return samples

    def sample(
        self,
        n: int,
        seed: int = 42,
        answerable_only: bool = True,
        categories: list[str] | None = None,
    ) -> list[CUADSample]:
        samples = self.load()

        if answerable_only:
            samples = [s for s in samples if not s.is_impossible]
        if categories:
            samples = [s for s in samples if s.category in categories]
        if n <= 0 or n >= len(samples):
            return samples

        # Stratified by task_type for representative coverage
        rng = random.Random(seed)
        by_type: dict[str, list[CUADSample]] = {}
        for s in samples:
            by_type.setdefault(s.task_type, []).append(s)

        result: list[CUADSample] = []
        per_type = max(1, n // len(by_type))
        for group in by_type.values():
            rng.shuffle(group)
            result.extend(group[:per_type])

        rng.shuffle(result)
        return result[:n]

    def get_categories(self) -> list[str]:
        return list(QUESTION_TEMPLATES.keys())
