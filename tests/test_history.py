"""Tests for the SQLite history store in backend/history.py."""

import os
import tempfile

import pytest

from backend.history import (
    AnalysisRecord,
    delete_analysis,
    get_all_analyses,
    get_score_trend,
    save_analysis,
)
from backend.config import settings


@pytest.fixture(autouse=True)
def temp_db_path(monkeypatch, tmp_path):
    """Redirect history DB to a temp file for each test."""
    db_path = str(tmp_path / "test_history.db")
    monkeypatch.setattr(settings, "history_db_path", db_path)
    yield db_path


def test_save_and_retrieve_analysis():
    record = AnalysisRecord(
        resume_filename="resume.pdf",
        jd_snippet="Looking for a Python developer",
        score=8,
        gaps=["Missing Docker experience"],
    )
    record_id = save_analysis(record)
    assert record_id is not None

    records = get_all_analyses()
    assert len(records) > 0
    assert any(r["score"] == 8 and r["resume_filename"] == "resume.pdf" for r in records)


def test_delete_analysis():
    record = AnalysisRecord(
        resume_filename="resume2.pdf",
        jd_snippet="Senior ML Engineer",
        score=5,
        gaps=["No cloud experience"],
    )
    record_id = save_analysis(record)

    deleted = delete_analysis(record_id)
    assert deleted is True

    records = get_all_analyses()
    assert not any(r.get("id") == record_id for r in records)


def test_score_trend():
    r1 = AnalysisRecord(resume_filename="r1.pdf", score=7)
    r2 = AnalysisRecord(resume_filename="r2.pdf", score=9)
    save_analysis(r1)
    save_analysis(r2)

    trend = get_score_trend()
    assert isinstance(trend, list)
    assert len(trend) >= 2
    scores = [t["score"] for t in trend]
    assert 7 in scores
    assert 9 in scores
