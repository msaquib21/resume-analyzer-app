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


# ── API Response Schemas ──────────────────────────────────────────────────────


class AnalysisResponse(BaseModel):
    """Typed response body returned by ``POST /analyze``."""

    score: int = Field(ge=0, le=10, description="Resume-to-JD match score (0–10).")
    gaps: list[str] = Field(description="Identified skill/experience gaps.")
    improvements: list[str] = Field(description="Actionable resume improvements.")
    preparation: list[str] = Field(description="Interview preparation steps.")


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
