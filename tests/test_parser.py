"""Tests for the self-healing parser in backend/agent.py."""

import json
import pytest

from backend.agent import _normalize_list_item, _clean_json_str


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
