"""
Unit tests for baseline implementations — oracle, local-only, entity substitution.
"""
from src.baselines.oracle import OracleBaseline, BaselineResult
from src.baselines.local_only import LocalOnlyBaseline
from src.baselines.entity_substitution import EntitySubstitutionBaseline, EntitySubstitutionResult


# ── Oracle ───────────────────────────────────────────────────────────────────

def test_oracle_returns_baseline_result(mock_ollama, sample_cuad_data):
    oracle = OracleBaseline(cloud_client=mock_ollama)
    result = oracle.run(sample_cuad_data[0])
    assert isinstance(result, BaselineResult)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.timing > 0


def test_oracle_truncates_context(mock_ollama, sample_cuad_data):
    oracle = OracleBaseline(cloud_client=mock_ollama, max_context_chars=50)
    result = oracle.run(sample_cuad_data[0])
    assert isinstance(result.answer, str)


# ── Local-only ───────────────────────────────────────────────────────────────

def test_local_only_returns_baseline_result(mock_ollama, sample_cuad_data):
    local = LocalOnlyBaseline(local_client=mock_ollama)
    result = local.run(sample_cuad_data[0])
    assert isinstance(result, BaselineResult)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert result.timing > 0


# ── Entity substitution ─────────────────────────────────────────────────────

def test_entity_sub_returns_result(mock_ollama, sample_cuad_data):
    entity_sub = EntitySubstitutionBaseline(cloud_client=mock_ollama)
    result = entity_sub.run(sample_cuad_data[0])
    assert isinstance(result, EntitySubstitutionResult)
    assert isinstance(result.answer, str)
    assert result.timing > 0


def test_entity_sub_substitutes_entities(mock_ollama, sample_cuad_data):
    entity_sub = EntitySubstitutionBaseline(cloud_client=mock_ollama)
    result = entity_sub.run(sample_cuad_data[0])
    # The sample context has "$50M" and "2023" and company names — some should be substituted
    if result.substitution_map:
        # Substituted text should contain placeholders
        has_placeholder = any(
            label in result.substituted_context
            for label in result.substitution_map.values()
        )
        assert has_placeholder, "Expected at least one placeholder in substituted context"


def test_entity_sub_restore_reverses_substitution():
    entity_sub = EntitySubstitutionBaseline(cloud_client=None)
    entity_map = {"Acme Corp": "ENTITY_1", "$50M": "AMOUNT_1"}
    text = "ENTITY_1 has revenues of AMOUNT_1"
    restored = entity_sub._restore(text, entity_map)
    assert "Acme Corp" in restored
    assert "$50M" in restored
    assert "ENTITY_1" not in restored
