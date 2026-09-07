"""
Centralised application configuration via Pydantic Settings.

All values can be overridden with environment variables, e.g.::

    OLLAMA_MODEL=qwen2.5:7b uvicorn backend.server:app --reload
    LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=lsv2_... uvicorn backend.server:app
"""

from __future__ import annotations

import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Dual LLM Provider (Local Ollama vs. Groq Cloud API) ───────────────
    # Set LLM_PROVIDER=groq and GROQ_API_KEY=gsk_... for free cloud deployment on Render/Vercel
    llm_provider: str = "ollama"       # "ollama" or "groq"
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"

    # ── Ollama Model Selection & Targets ──────────────────────────────────
    # Easily swap model via environment variable OLLAMA_MODEL (e.g. qwen3.5:9b or qwen2.5:7b)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    available_models: List[str] = [
        "qwen2.5:3b",      # Ultra-fast local execution (default)
        "qwen3.5:9b",      # Next-gen reasoning with enhanced tool adherence
        "qwen2.5:7b",      # Enhanced reasoning and complex schema adherence
        "qwen3:8b",        # Structured instruction-following
        "llama3.1:8b",     # General open-source reasoning
    ]
    ollama_temperature: float = 0.0

    # ── Embeddings ────────────────────────────────────────────────────────
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG / Hybrid Retrieval (BM25 + ChromaDB) ──────────────────────────
    chunk_size: int = 600
    chunk_overlap: int = 120        # 20% overlap preserved across splits
    retrieval_k: int = 10           # Top-K chunks per query for comprehensive context
    retrieval_queries: int = 4      # Multi-query distinct RAG angles
    vector_store_cache_max: int = 5
    bm25_weight: float = 0.5        # Equal balance between sparse lexical & dense semantic

    # ── Server ────────────────────────────────────────────────────────────
    graph_timeout_seconds: int = 300
    cors_origins: list[str] = ["*"]
    max_concurrent_analyses: int = 3

    # ── History ───────────────────────────────────────────────────────────
    history_db_path: str = "data/history.db"

    # ── LangSmith Observability & Tracing ──────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "agentic-resume-analyzer"

    def setup_observability(self) -> None:
        """Propagate LangSmith tracing configuration into os.environ for LangGraph / LangChain."""
        if self.langchain_tracing_v2 or os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            if self.langchain_api_key:
                os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
settings.setup_observability()
