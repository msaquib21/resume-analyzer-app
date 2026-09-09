"""
Pydantic models for request/response validation and LLM output parsing.

Using explicit BaseModel schemas for the API surface demonstrates the correct
use of FastAPI + Pydantic — typed contracts instead of raw dicts.
"""

from __future__ import annotations

import json
import re
from typing import Any
from pydantic import BaseModel, Field, field_validator


def normalize_list_item(item: Any) -> str:
    """Extract a clean plain string from whatever the LLM puts in a list slot.

    Handles three cases:
    1. Plain string  → return as-is (most common after fixing prompts).
    2. Simple dict   → pick known keys or format structured sub-keys into crisp pipe-separated paragraphs.
    3. Multi-key dict → concatenate all non-empty string values into one paragraph.
    """
    if isinstance(item, dict):
        # 1. Check for structured gaps dictionary
        skill_val = item.get("Skill Name") or item.get("skill_name") or item.get("Skill") or item.get("name")
        missing_val = item.get("Missing Evidence") or item.get("missing_evidence") or item.get("Missing")
        jd_val = item.get("JD Demands") or item.get("jd_demands") or item.get("JD") or item.get("demands")
        if skill_val and isinstance(skill_val, str) and skill_val.strip() and (missing_val or jd_val):
            parts = []
            if isinstance(missing_val, str) and missing_val.strip():
                parts.append(missing_val.strip().rstrip(".") + ".")
            if isinstance(jd_val, str) and jd_val.strip():
                parts.append(f"JD requires: {jd_val.strip().rstrip('.')}.")
            body = " ".join(parts) if parts else ""
            if body:
                return f"{skill_val.strip()}: {body}"

        # 2. Check for structured improvements dictionary
        action_val = item.get("Action Required") or item.get("action_required") or item.get("Action") or item.get("action")
        area_val = item.get("Target Area") or item.get("target_area") or item.get("Area") or item.get("area") or item.get("Target")
        align_val = item.get("JD Alignment") or item.get("jd_alignment") or item.get("Alignment") or item.get("why")
        if action_val and isinstance(action_val, str) and action_val.strip():
            area_str = f"Target Area: {area_val.strip()} | " if isinstance(area_val, str) and area_val.strip() else ""
            align_str = f" | JD Alignment: {align_val.strip()}" if isinstance(align_val, str) and align_val.strip() else ""
            return f"{area_str}Action Required: {action_val.strip()}{align_str}"

        # 3. Check for structured preparation dictionary
        gap_val = item.get("Target Gap") or item.get("target_gap") or item.get("Gap") or item.get("gap")
        study_val = item.get("Study") or item.get("study") or item.get("resource")
        prac_val = item.get("Practice") or item.get("practice") or item.get("project")
        angle_val = item.get("Interview Angle") or item.get("interview_angle") or item.get("angle")
        study_str = study_val.strip() if isinstance(study_val, str) else ""
        prac_str = prac_val.strip() if isinstance(prac_val, str) else ""
        angle_str = angle_val.strip() if isinstance(angle_val, str) else ""
        if study_str or prac_str or angle_str:
            gap_str = f"Target Gap: {gap_val.strip()} | " if isinstance(gap_val, str) and gap_val.strip() else ""
            s_str = f"Study: {study_str} | " if study_str else ""
            p_str = f"Practice: {prac_str} | " if prac_str else ""
            a_str = f"Interview Angle: {angle_str}" if angle_str else ""
            return f"{gap_str}{s_str}{p_str}{a_str}".strip(" |")

        # Try well-known single-key patterns next
        single_key_priority = (
            "type", "gap", "improvement", "preparation",
            "text", "description", "value", "content",
        )
        for k in single_key_priority:
            if isinstance(item.get(k), str) and item[k].strip():
                return item[k].strip()

        # Multi-key dict: reconstruct as "Key: value. Key2: value2."
        str_values = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
        if str_values:
            if len(str_values) == 1:
                return str_values[0]
            head = str_values[0].rstrip(".")
            tail = " ".join(v.rstrip(".") + "." for v in str_values[1:])
            return f"{head}: {tail}"
        return ""

    if isinstance(item, str):
        s = item.strip()
        # Attempt to decode if it looks like a stringified dict
        if s.startswith("{"):
            try:
                decoded = json.loads(s)
                if isinstance(decoded, dict):
                    return normalize_list_item(decoded)
            except Exception:
                pass
        if s in ('""', "''", "[]", "{}", "none", "n/a", "null"):
            return ""
        return s

    if item is None:
        return ""
    s = str(item).strip()
    return "" if s in ('""', "''", "[]", "{}", "none", "n/a", "null") else s


def _sanitize_string_list(v: Any) -> list[str]:
    """Filter out empty strings, whitespace-only strings, and empty dictionaries/placeholders."""
    if not v:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []

    cleaned: list[str] = []
    for raw_item in v:
        if raw_item is None:
            continue
        normalized = normalize_list_item(raw_item).strip()
        if normalized and normalized not in ('""', "''", "[]", "{}", "none", "n/a", "null"):
            cleaned.append(normalized)
    return cleaned


# ── LLM Output Schemas (used by PydanticOutputParser in agent nodes) ─────────


class GapAnalysisOutput(BaseModel):
    """Structured output expected from the gap-analysis LLM node."""

    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of specific, actionable skill or experience gaps. "
            "If the candidate's resume explicitly satisfies a job description requirement, do not flag it as a gap. "
            "You are a strict text-matcher. Base gaps ONLY on the provided job description text. "
            "Do not hallucinate industry standards (e.g., AWS, Pinecone) if they are not explicitly written. "
            "If there are no items to report, you MUST return a perfectly empty list []. Do NOT return lists containing empty strings or placeholder text."
        ),
    )

    parse_failed: bool = Field(
        default=False,
        description="Flag indicating if the LLM output could not be parsed into the schema.",
    )

    @field_validator("gaps", mode="before")
    @classmethod
    def clean_gaps(cls, v: Any) -> list[str]:
        return _sanitize_string_list(v)


class ScoreCoachOutput(BaseModel):
    """Structured output expected from the score + coaching LLM node."""

    score: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Integer match score from 0 (no alignment) to 10 (perfect match).",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of skill/experience gaps tied strictly to the JD. "
            "If the candidate's resume explicitly satisfies a job description requirement, do not flag it as a gap. "
            "You are a strict text-matcher. Base gaps ONLY on the provided job description text. "
            "Do not hallucinate industry standards (e.g., AWS, Pinecone) if they are not explicitly written. "
            "If there are no items to report, you MUST return a perfectly empty list []. Do NOT return lists containing empty strings or placeholder text."
        ),
    )
    improvements: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of concrete resume bullet rewrites or additions addressing each identified gap. "
            "Never recommend a resume improvement or bullet point that is already visibly present in the candidate's uploaded resume text. "
            "If there are no items to report, you MUST return a perfectly empty list []. Do NOT return lists containing empty strings or placeholder text."
        ),
    )
    preparation: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of specific interview study topics aligned strictly to the identified job description gaps and weaknesses. "
            "If there are no items to report, you MUST return a perfectly empty list []. Do NOT return lists containing empty strings or placeholder text."
        ),
    )
    parse_failed: bool = Field(
        default=False,
        description="Flag indicating if the LLM output could not be parsed into the schema.",
    )

    @field_validator("gaps", "improvements", "preparation", mode="before")
    @classmethod
    def clean_lists(cls, v: Any) -> list[str]:
        return _sanitize_string_list(v)


# ── Keyword Matching ──────────────────────────────────────────────────────────


class KeywordMatch(BaseModel):
    """Single JD keyword and its presence status in the resume."""

    keyword: str = Field(description="Skill, tool, or technology from the JD.")
    found_in_resume: bool = Field(
        default=False,
        description="Whether this keyword was found in the resume text.",
    )
    category: str = Field(
        default="skill",
        description="Category: 'skill', 'tool', 'certification', 'soft_skill'.",
    )


# ── API Response Schemas ──────────────────────────────────────────────────────


class AnalysisResponse(BaseModel):
    """Typed response body returned by ``POST /analyze``."""

    analysis_id: str = Field(default="", description="Unique analysis ID for history tracking.")
    score: int = Field(ge=0, le=10, description="Resume-to-JD match score (0–10).")
    gaps: list[str] = Field(description="Identified skill/experience gaps.")
    improvements: list[str] = Field(description="Actionable resume improvements.")
    preparation: list[str] = Field(description="Interview preparation steps.")
    keywords: list[KeywordMatch] = Field(
        default_factory=list,
        description="JD keywords with resume match status.",
    )
    elapsed_seconds: float = Field(default=0.0, description="Analysis duration in seconds.")

    @field_validator("gaps", "improvements", "preparation", mode="before")
    @classmethod
    def clean_response_lists(cls, v: Any) -> list[str]:
        return _sanitize_string_list(v)


class AnalysisHistoryItem(BaseModel):
    """Summary of a past analysis for the history list."""

    id: str = Field(description="Analysis ID.")
    timestamp: str = Field(description="ISO 8601 timestamp.")
    resume_filename: str = Field(default="", description="Original resume filename.")
    jd_snippet: str = Field(default="", description="First ~200 chars of the JD.")
    score: int = Field(default=0, ge=0, le=10, description="Match score.")
    elapsed_seconds: float = Field(default=0.0, description="Analysis duration.")
    gap_count: int = Field(default=0, description="Number of gaps identified.")
    keyword_match_pct: float = Field(
        default=0.0,
        description="Percentage of JD keywords found in resume.",
    )


class AnalysisDetail(BaseModel):
    """Full detail of a past analysis."""

    id: str
    timestamp: str
    resume_filename: str = ""
    jd_snippet: str = ""
    score: int = 0
    gaps: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    preparation: list[str] = Field(default_factory=list)
    keywords: list[KeywordMatch] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    feedback: dict = Field(default_factory=dict)


class ScoreTrendPoint(BaseModel):
    """Single data point for the score trend chart."""

    timestamp: str
    score: int
    resume_filename: str = ""


class FeedbackRequest(BaseModel):
    """Request body for submitting feedback on analysis cards."""

    analysis_id: str = Field(description="Analysis to attach feedback to.")
    feedback: dict = Field(
        description="Feedback map, e.g. {'gaps': {'0': 'up', '1': 'down'}}."
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str = Field(description="'ok' if all services are reachable.")
    llm_provider: str = Field(default="ollama", description="Active LLM provider ('ollama' or 'groq').")
    ollama_reachable: bool = Field(description="Whether the Ollama API responded.")
    groq_configured: bool = Field(default=False, description="Whether Groq API is configured with an API key.")
    model: str = Field(description="Configured LLM model name.")
    embeddings_model: str = Field(description="Configured embeddings model name.")
    available_models: list[str] = Field(
        default_factory=list,
        description="Supported local Ollama models that can be selected.",
    )


class ErrorDetail(BaseModel):
    """Structured error payload for non-2xx responses."""

    detail: str = Field(description="Human-readable error message.")
    stage: str | None = Field(
        default=None,
        description="Pipeline stage where the error occurred, if applicable.",
    )
