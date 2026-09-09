"""
Centralised application configuration via Pydantic Settings.

All values can be overridden with environment variables, e.g.::

OLLAMA_MODEL=qwen2.5:7b uvicorn backend.server:app --reload
LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=lsv2_... uvicorn backend.server:app
"""

from __future__ import annotations

import json
import os
from typing import List, Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_list(raw: str | list, fallback: List[str]) -> List[str]:
    """Parse a list setting from a bare, comma-separated, or JSON string.

    pydantic-settings JSON-decodes list-typed fields inside its env source, before
    validators run, so ``CORS_ORIGINS=*`` used to raise SettingsError and take the
    whole backend down at import. These settings are therefore declared as plain
    strings and parsed here, which accepts every spelling someone might type.
    """
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = (raw or "").strip()
    if not text:
        return list(fallback)
    if text.startswith("["):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(x).strip() for x in decoded if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


_DEFAULT_MODELS = [
    "qwen2.5:3b",      # Ultra-fast local execution (~3s) - recommended default
    "qwen3.5:9b",      # Next-gen reasoning with large context (requires 16GB+ RAM)
    "qwen2.5-coder:7b",# Code and technical specialist
    "qwen2.5:7b",      # Enhanced general reasoning
    "llama3.1:latest", # Llama 3.1 fallback
]


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
    # Swap model via environment variable OLLAMA_MODEL, or per-request via the
    # `model` form field on /analyze and /analyze/stream.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    # Declared as a string so bare and comma-separated values work; read the
    # parsed list via the `available_models` property below.
    available_models_raw: str = Field(
        default=",".join(_DEFAULT_MODELS),
        validation_alias=AliasChoices("AVAILABLE_MODELS", "available_models"),
    )
    ollama_temperature: float = 0.0

    # ── Sampling overrides ────────────────────────────────────────────────
    ollama_top_k: int = 20
    ollama_top_p: float = 0.95
    ollama_repeat_penalty: float = 1.0

    # ── Ollama Runtime Window ─────────────────────────────────────────────
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 1200

    # ── Embeddings ────────────────────────────────────────────────────────
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG / Hybrid Retrieval (BM25 + ChromaDB) ──────────────────────────
    chunk_size: int = 600
    chunk_overlap: int = 120        # 20% overlap preserved across splits
    retrieval_k: int = 10           # Top-K chunks per individual query
    retrieval_queries: int = 4      # Multi-query distinct RAG angles
    retrieval_top_n: int = 12       # Chunks kept after RRF fusion, re-sorted to document order
    single_page_max_words: int = 1000   # Below this, skip retrieval and use the full document
    vector_store_cache_max: int = 5
    bm25_weight: float = 0.5        # Equal balance between sparse lexical & dense semantic

    # ── Server ────────────────────────────────────────────────────────────
    graph_timeout_seconds: int = 300
    cors_origins_raw: str = Field(
        default="*",
        validation_alias=AliasChoices("CORS_ORIGINS", "cors_origins"),
    )
    max_concurrent_analyses: int = 3

    # ── History ───────────────────────────────────────────────────────────
    history_db_path: str = "data/history.db"

    # ── LangSmith Observability & Tracing ──────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "agentic-resume-analyzer"

    # ── Parsed list settings ──────────────────────────────────────────────
    # Callers use these names; the *_raw string fields above hold the input.

    @property
    def cors_origins(self) -> List[str]:
        return _parse_list(self.cors_origins_raw, ["*"])

    @property
    def available_models(self) -> List[str]:
        return _parse_list(self.available_models_raw, _DEFAULT_MODELS)

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