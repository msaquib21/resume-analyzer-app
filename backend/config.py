"""
Centralised application configuration via Pydantic Settings.

All values can be overridden with environment variables, e.g.::

    OLLAMA_MODEL=llama3.2:3b uvicorn backend.server:app --reload
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Ollama ────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_temperature: float = 0.0

    # ── Embeddings ────────────────────────────────────────────────────────
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG / Chunking (Optimized: 20% overlap, increased Top-K) ──────────
    chunk_size: int = 600
    chunk_overlap: int = 120        # 20% overlap to preserve semantic context across splits
    retrieval_k: int = 10           # Increased Top-K chunks per query for comprehensive context
    retrieval_queries: int = 4      # Multi-query distinct RAG angles
    vector_store_cache_max: int = 5

    # ── Server ────────────────────────────────────────────────────────────
    graph_timeout_seconds: int = 300
    cors_origins: list[str] = ["*"]
    max_concurrent_analyses: int = 3

    # ── History ───────────────────────────────────────────────────────────
    history_db_path: str = "data/history.db"


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
