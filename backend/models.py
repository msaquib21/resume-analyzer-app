"""
Pydantic models for request/response validation and LLM output parsing.

Using explicit BaseModel schemas for the API surface demonstrates the correct
use of FastAPI + Pydantic — typed contracts instead of raw dicts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── LLM Output Schemas (used by PydanticOutputParser in agent nodes) ─────────


class GapAnalysisOutput(BaseModel):
    """Structured output expected from the gap-analysis LLM node."""

    gaps: list[str] = Field(
        default_factory=list,
        description="List of specific, actionable skill or experience gaps.",
    )


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
        description="Refined list of skill/experience gaps tied to the JD.",
    )
    improvements: list[str] = Field(
        default_factory=list,
        description="Concrete resume bullet rewrites or additions addressing each gap.",
    )
    preparation: list[str] = Field(
        default_factory=list,
        description="Specific interview study topics aligned to gaps and weaknesses.",
    )


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
    ollama_reachable: bool = Field(description="Whether the Ollama API responded.")
    model: str = Field(description="Configured LLM model name.")
    embeddings_model: str = Field(description="Configured embeddings model name.")


class ErrorDetail(BaseModel):
    """Structured error payload for non-2xx responses."""

    detail: str = Field(description="Human-readable error message.")
    stage: str | None = Field(
        default=None,
        description="Pipeline stage where the error occurred, if applicable.",
    )
