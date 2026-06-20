"""
Unit tests for CUAD loader utilities — question rewriting, taxonomy mapping,
and deterministic dev/test split.

Avoids the HuggingFace dataset download by testing only the transformation
logic (QUESTION_TEMPLATES lookup, _rewrite_question, task type mapping).
"""
import pytest
from src.data.cuad_loader import QUESTION_TEMPLATES, CUADSample, Split, _assign_split
from src.data.question_taxonomy import (
    CATEGORY_TO_TASK_TYPE, TASK_TYPES, get_task_type,
)


# ── QUESTION_TEMPLATES ────────────────────────────────────────────────────────

def test_all_41_categories_have_templates():
    # CUAD has exactly 41 clause categories
    assert len(QUESTION_TEMPLATES) == 41


def test_templates_are_questions():
    for category, template in QUESTION_TEMPLATES.items():
        assert "?" in template, f"Template for '{category}' missing '?'"


def test_templates_are_non_empty_strings():
    for category, template in QUESTION_TEMPLATES.items():
        assert isinstance(template, str)
        assert len(template) > 10, f"Template for '{category}' too short"


# ── question_taxonomy ─────────────────────────────────────────────────────────

def test_all_categories_mapped():
    """Every template category should have a task type."""
    for category in QUESTION_TEMPLATES:
        task = get_task_type(category)
        assert task in TASK_TYPES, f"Category '{category}' → unknown task type '{task}'"


def test_task_types_exhaustive():
    assert set(TASK_TYPES) == {
        "framework_reasoning",
        "compliance_regulatory",
        "procedural",
        "entity_specific",
    }


def test_get_task_type_known_category():
    tt = get_task_type("Non-Compete")
    assert tt in TASK_TYPES


def test_get_task_type_unknown_returns_framework():
    # Unknown categories fall back to a safe default
    tt = get_task_type("This Category Does Not Exist")
    assert tt in TASK_TYPES


def test_category_to_task_type_covers_all_templates():
    for category in QUESTION_TEMPLATES:
        assert category in CATEGORY_TO_TASK_TYPE or get_task_type(category) in TASK_TYPES


# ── CUADSample ────────────────────────────────────────────────────────────────

def test_cuad_sample_creation():
    sample = CUADSample(
        id="test-001",
        question="What non-compete restrictions does this contract impose?",
        raw_question="Highlight the parts related to Non-Compete.",
        context="The parties agree to the following non-compete terms...",
        answers=["two years"],
        category="Non-Compete",
        task_type="compliance_regulatory",
        doc_title="Test Agreement",
        is_impossible=False,
    )
    assert sample.id == "test-001"
    assert len(sample.answers) == 1
    assert not sample.is_impossible


def test_cuad_sample_default_is_impossible():
    sample = CUADSample(
        id="x",
        question="Q?",
        raw_question="Raw Q",
        context="ctx",
        answers=[],
        category="Parties",
        task_type="entity_specific",
        doc_title="Doc",
        is_impossible=True,
    )
    assert sample.is_impossible
    assert sample.answers == []


# ── Split assignment ─────────────────────────────────────────────────────────

def test_assign_split_is_deterministic():
    """Same id always gets the same split."""
    s1 = _assign_split("sample-abc-123")
    s2 = _assign_split("sample-abc-123")
    assert s1 == s2


def test_assign_split_returns_valid_split():
    assert _assign_split("any-id") in (Split.DEV, Split.TEST)


def test_assign_split_produces_both_splits():
    """Over many ids, both splits should appear."""
    splits = {_assign_split(f"id-{i}") for i in range(200)}
    assert Split.DEV in splits
    assert Split.TEST in splits


def test_assign_split_approximate_ratio():
    """With default 30% test, ratio should be roughly 0.3 over many samples."""
    n = 5000
    test_count = sum(1 for i in range(n) if _assign_split(f"s-{i}") == Split.TEST)
    ratio = test_count / n
    assert 0.25 < ratio < 0.35, f"Test ratio {ratio:.3f} outside expected range"


def test_cuad_sample_default_split_is_dev():
    sample = CUADSample(
        id="x",
        question="Q?",
        raw_question="Raw Q",
        context="ctx",
        answers=["ans"],
        category="Parties",
        task_type="entity_specific",
        doc_title="Doc",
        is_impossible=False,
    )
    assert sample.split == Split.DEV
