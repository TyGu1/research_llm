"""
Shared pytest fixtures for all tests.

Unit tests use only mock clients — no Ollama required.
Integration tests use real Ollama and are marked with @pytest.mark.integration.
"""
import pytest

from src.config import PipelineConfig
from src.data.cuad_loader import CUADSample
from src.llm.mock_client import MockOllamaClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama() -> MockOllamaClient:
    return MockOllamaClient()


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        cloud_provider="mock",
        local_model="phi3",
        embed_model="nomic-embed-text",
        num_proxy_questions=8,
        num_selected_questions=3,
        temperature=0.0,
        use_vocab_exclusion=False,
    )


@pytest.fixture
def sample_cuad_data() -> list[CUADSample]:
    """Three hardcoded CUAD samples across different task types. No HF download needed."""
    return [
        CUADSample(
            id="test-001",
            question="What non-compete restrictions does this contract impose on the parties?",
            raw_question="Highlight the parts (if any) of this contract related to Non-Compete.",
            context=(
                "This Agreement is entered into between Acme Corp (Delaware) and BetaCo Inc. "
                "Neither party shall, during the Term and for two (2) years following termination, "
                "directly or indirectly engage in any business that competes with the other party's "
                "core business operations in North America. Acme Corp's primary business includes "
                "enterprise software solutions for Fortune 500 companies. BetaCo Inc specializes in "
                "cloud infrastructure services with annual revenues exceeding $50M."
            ),
            answers=["two (2) years", "North America"],
            category="Non-Compete",
            task_type="compliance_regulatory",
            doc_title="Acme-BetaCo Partnership Agreement 2023",
            is_impossible=False,
        ),
        CUADSample(
            id="test-002",
            question="Who owns intellectual property created under this contract?",
            raw_question="Highlight the parts (if any) of this contract related to Ip Ownership Assignment.",
            context=(
                "Section 7. Intellectual Property. All inventions, developments, works of authorship, "
                "and other intellectual property (collectively 'Work Product') created by Contractor "
                "in the performance of services under this Agreement shall be considered works made "
                "for hire and shall vest exclusively in Client upon creation. Contractor hereby assigns "
                "to Client all right, title, and interest in and to any Work Product, including all "
                "patent rights, copyright, trade secrets, and moral rights. Contractor retains no "
                "rights in any Work Product. Client may file patent applications on any Work Product "
                "in its sole discretion."
            ),
            answers=["Client", "works made for hire"],
            category="Ip Ownership Assignment",
            task_type="entity_specific",
            doc_title="Software Development Services Agreement",
            is_impossible=False,
        ),
        CUADSample(
            id="test-003",
            question="Under what conditions can either party terminate this contract for convenience?",
            raw_question="Highlight the parts (if any) of this contract related to Termination For Convenience.",
            context=(
                "Section 12. Termination. 12.1 Termination for Convenience. Either party may terminate "
                "this Agreement for any reason or no reason upon sixty (60) days' prior written notice "
                "to the other party. Upon receipt of such notice, the parties shall cooperate in good "
                "faith to wind down any outstanding obligations. 12.2 Termination for Cause. Either "
                "party may terminate immediately upon written notice if the other party materially "
                "breaches this Agreement and fails to cure such breach within thirty (30) days of "
                "receiving written notice of the breach."
            ),
            answers=["sixty (60) days' prior written notice", "thirty (30) days"],
            category="Termination For Convenience",
            task_type="procedural",
            doc_title="Master Services Agreement v2.1",
            is_impossible=False,
        ),
    ]
