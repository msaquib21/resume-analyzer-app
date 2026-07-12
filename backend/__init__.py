"""
Agentic Resume Analyzer — Backend Package
==========================================

A local-only, privacy-first resume analysis pipeline powered by:
- **LangGraph** multi-agent workflow
- **RAG** with ChromaDB + sentence-transformers
- **Ollama** for local LLM inference

Public API
----------
- ``build_resume_analysis_graph`` — compile the LangGraph analysis pipeline
- ``Settings`` — centralised, env-configurable application settings
- ``AnalysisResponse`` — typed response schema for the ``/analyze`` endpoint
"""

from .agent import build_resume_analysis_graph
from .config import Settings
from .models import AnalysisResponse

__all__ = [
    "build_resume_analysis_graph",
    "Settings",
    "AnalysisResponse",
]
