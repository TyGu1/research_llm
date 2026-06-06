"""
Step 6: Local LLM synthesizes the final answer.
Inputs: original S2 question (held locally) + S0 answers from cloud LLM.
The local LLM applies generic reasoning frameworks to the specific S2 context.
"""
from src.pipeline.base import Synthesizer, S0Answer

SYSTEM_PROMPT = """You are a legal analyst. You have access to general legal knowledge
and a specific confidential contract. Apply general legal frameworks to answer
specific questions about the contract. Be precise and grounded in the contract text."""

PROMPT_TEMPLATE = """You need to answer a specific question about a contract.

QUESTION (about this specific contract):
{question}

CONTRACT EXCERPT:
{context}

RELEVANT GENERAL LEGAL KNOWLEDGE (from external research):
{s0_answers}

Using the general legal knowledge above as a framework, answer the specific question
about this contract. Be concrete and reference the contract text where possible."""


class LLMSynthesizer(Synthesizer):
    def __init__(self, client, max_context_chars: int = 8000):
        self._client = client
        self._max_context_chars = max_context_chars

    def synthesize(
        self,
        original_question: str,
        context: str,
        s0_answers: list[S0Answer],
    ) -> str:
        truncated_ctx = context[:self._max_context_chars]

        s0_text = "\n\n".join(
            f"Q: {a.question}\nA: {a.answer}" for a in s0_answers
        )

        prompt = PROMPT_TEMPLATE.format(
            question=original_question,
            context=truncated_ctx,
            s0_answers=s0_text,
        )

        return self._client.generate(prompt, system=SYSTEM_PROMPT)
