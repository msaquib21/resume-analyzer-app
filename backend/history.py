"""
SQLite-backed analysis history store.

Persists every completed analysis so users can track score progression,
compare across JDs, and revisit past results.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import settings


@dataclass
class AnalysisRecord:
    """Single persisted analysis result."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resume_filename: str = ""
    jd_snippet: str = ""
    score: int = 0
    gaps: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    preparation: list[str] = field(default_factory=list)
    keywords: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    feedback: dict = field(default_factory=dict)  # {"gaps": {0: "up"}, ...}


# ── Database helpers ──────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS analyses (
    id               TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,
    resume_filename  TEXT,
    jd_snippet       TEXT,
    score            INTEGER DEFAULT 0,
    gaps             TEXT DEFAULT '[]',
    improvements     TEXT DEFAULT '[]',
    preparation      TEXT DEFAULT '[]',
    keywords         TEXT DEFAULT '[]',
    elapsed_seconds  REAL DEFAULT 0.0,
    feedback         TEXT DEFAULT '{}'
)
"""


def _get_db() -> sqlite3.Connection:
    """Return a connection to the history database, creating it if needed."""
    db_path = settings.history_db_path
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def save_analysis(record: AnalysisRecord) -> str:
    """Insert a new analysis record. Returns the record ID."""
    conn = _get_db()
    try:
        conn.execute(
            """INSERT INTO analyses
               (id, timestamp, resume_filename, jd_snippet, score,
                gaps, improvements, preparation, keywords, elapsed_seconds, feedback)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.timestamp,
                record.resume_filename,
                record.jd_snippet[:500],  # Truncate for storage
                record.score,
                json.dumps(record.gaps),
                json.dumps(record.improvements),
                json.dumps(record.preparation),
                json.dumps(record.keywords),
                record.elapsed_seconds,
                json.dumps(record.feedback),
            ),
        )
        conn.commit()
        return record.id
    finally:
        conn.close()


def get_all_analyses() -> list[dict]:
    """Return all analyses, newest first."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY timestamp DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_analysis(analysis_id: str) -> Optional[dict]:
    """Return a single analysis by ID, or None."""
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_analysis(analysis_id: str) -> bool:
    """Delete an analysis. Returns True if a row was actually deleted."""
    conn = _get_db()
    try:
        cursor = conn.execute(
            "DELETE FROM analyses WHERE id = ?", (analysis_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_score_trend() -> list[dict]:
    """Return score history for charting: [{timestamp, score, resume_filename}]."""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT timestamp, score, resume_filename FROM analyses ORDER BY timestamp ASC"
        ).fetchall()
        return [
            {
                "timestamp": r["timestamp"],
                "score": r["score"],
                "resume_filename": r["resume_filename"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def save_feedback(analysis_id: str, feedback: dict) -> bool:
    """Update feedback for an existing analysis."""
    conn = _get_db()
    try:
        cursor = conn.execute(
            "UPDATE analyses SET feedback = ? WHERE id = ?",
            (json.dumps(feedback), analysis_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict with parsed JSON fields."""
    d = dict(row)
    for key in ("gaps", "improvements", "preparation", "keywords", "feedback"):
        if isinstance(d.get(key), str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key] = [] if key != "feedback" else {}
    return d
