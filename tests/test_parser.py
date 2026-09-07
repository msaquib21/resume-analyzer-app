"""Tests for the self-healing parser in backend/agent.py and layout-aware parsing in backend/rag.py."""

import json
import pytest

from backend.agent import _normalize_list_item, _clean_json_str
from backend.rag import _parse_multicolumn_layout


def test_normalize_list_item_string():
    assert _normalize_list_item("Plain string input") == "Plain string input"


def test_normalize_list_item_dict_gap():
    assert _normalize_list_item({"gap": "Missing SQL experience"}) == "Missing SQL experience"


def test_normalize_list_item_dict_multi():
    """Multi-key gap dict should produce 'SkillName: evidence. JD requires: demand.'"""
    item = {"Skill Name": "Python", "Missing Evidence": "No projects", "JD Demands": "Requires Python"}
    result = _normalize_list_item(item)
    assert "Python" in result
    assert "No projects" in result
    assert "Requires Python" in result


def test_normalize_list_item_dict_action():
    """Improvement dict should produce 'Target Area: X | Action Required: Y'"""
    item = {"Target Area": "Education", "Action Required": "Add GPA"}
    result = _normalize_list_item(item)
    assert "Education" in result
    assert "Add GPA" in result


def test_normalize_list_item_none():
    assert _normalize_list_item(None) == ""


def test_normalize_list_item_json_string():
    assert _normalize_list_item('{"gap": "JSON gap"}') == "JSON gap"


def test_clean_json_str_clean():
    assert _clean_json_str('{"key": "value"}') == '{"key": "value"}'


def test_clean_json_str_markdown():
    result = _clean_json_str('```json\n{"key": "value"}\n```')
    assert '"key"' in result
    assert '"value"' in result


def test_parse_multicolumn_layout_disentangles_columns():
    """Multi-column resumes must be read vertically, not merged horizontally."""
    multicolumn_raw = (
        "Skills: Redis (Vector DB), OCI                     Experience: Senior ML Engineer at TechCorp\n"
        "Certifications: AWS Solution Architect             Led distributed RAG and fine-tuning pipelines\n"
        "Education: B.S. in Computer Science                Deployed FastAPI microservices with Docker\n"
        "Languages: Python, Go, TypeScript                  Managed high-throughput vector databases\n"
    )
    parsed = _parse_multicolumn_layout(multicolumn_raw)
    
    # Skills must appear together before Experience block
    assert "Redis (Vector DB)" in parsed
    assert "OCI" in parsed
    assert "Experience: Senior ML Engineer" in parsed
    
    # Verify that Column 1 skills are not merged horizontally on the same line with Column 2 experience
    for line in parsed.splitlines():
        if "Redis (Vector DB)" in line:
            assert "Senior ML Engineer" not in line, "Redis and Senior ML Engineer should NOT be horizontally merged on the same line"
        if "Certifications: AWS" in line:
            assert "Led distributed RAG" not in line, "Certifications and Experience should NOT be horizontally merged on the same line"


def test_parse_multicolumn_layout_single_column():
    """Single column text should be preserved cleanly without breaking."""
    single_column = (
        "Summary: Experienced AI systems engineer.\n"
        "Work Experience:\n"
        "- Built RAG applications with ChromaDB and LangGraph.\n"
    )
    parsed = _parse_multicolumn_layout(single_column)
    assert "Summary: Experienced AI systems engineer." in parsed
    assert "Work Experience:" in parsed
