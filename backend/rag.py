"""
RAG pipeline: PDF parsing, chunking, embedding, and multi-query retrieval.

Key design decisions
--------------------
- **Multi-query retrieval**: runs several distinct semantic queries derived from
  the JD (full description, keyword cluster, individual skill phrases) and merges
  results via Reciprocal Rank Fusion (RRF).  This significantly improves recall
  compared to a single-query similarity search.
- **Embedding + vector-store caching**: avoids re-embedding the same resume PDF
  on every request; keyed by SHA-256 hash of the raw PDF bytes.
- **Local-only**: no external API calls — everything runs via HuggingFace
  sentence-transformers and ChromaDB.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class RetrievedContext:
    """Result of a multi-query RAG retrieval pass."""

    retrieved_chunks: List[str] = field(default_factory=list)
    all_chunks_preview: List[str] = field(default_factory=list)
    query_count: int = 0  # number of distinct queries executed


# ── PDF loading & chunking ─────────────────────────────────────────────────────


def load_and_chunk_pdf(pdf_bytes: bytes) -> List[str]:
    """Extract text from PDF bytes and return a list of text chunks.

    Uses a temporary file because ``PyPDFLoader`` requires a filesystem path.
    The temp file is always cleaned up — even on error.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        split_docs = splitter.split_documents(docs)
        chunks = [d.page_content for d in split_docs if d.page_content.strip()]
        logger.info("PDF parsed: %d raw docs → %d chunks", len(docs), len(chunks))
        return chunks
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Embedding & vector store caching ─────────────────────────────────────────

_EMBEDDINGS_CACHE_LOCK = threading.Lock()
_EMBEDDINGS_CACHE: Dict[str, HuggingFaceEmbeddings] = {}

_VS_CACHE_LOCK = threading.Lock()
_VS_CACHE: Dict[str, Chroma] = {}


def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Return a cached ``HuggingFaceEmbeddings`` instance for *model_name*."""
    with _EMBEDDINGS_CACHE_LOCK:
        if model_name not in _EMBEDDINGS_CACHE:
            logger.info("Loading embeddings model: %s", model_name)
            _EMBEDDINGS_CACHE[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        return _EMBEDDINGS_CACHE[model_name]


def _resume_cache_key(pdf_bytes: bytes, model_name: str) -> str:
    """Stable cache key: SHA-256 of (model_name + pdf_bytes)."""
    h = hashlib.sha256()
    h.update(model_name.encode())
    h.update(b"::")
    h.update(pdf_bytes)
    return h.hexdigest()


def make_vector_store(
    chunks: List[str],
    *,
    embeddings_model_name: str,
    resume_pdf_bytes: Optional[bytes] = None,
) -> Chroma:
    """Build (or retrieve from cache) a Chroma vector store for *chunks*.

    The store is keyed by the SHA-256 hash of the raw PDF bytes so the same
    resume is never re-embedded within a server session.
    """
    embeddings = _get_embeddings(embeddings_model_name)
    cache_key = (
        _resume_cache_key(resume_pdf_bytes, embeddings_model_name)
        if resume_pdf_bytes is not None
        else None
    )

    if cache_key is not None:
        with _VS_CACHE_LOCK:
            cached = _VS_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("Vector store cache hit for key %s…", cache_key[:8])
            return cached

    persist_dir = (
        os.path.join(tempfile.gettempdir(), f"resume_chroma_{cache_key}")
        if cache_key
        else tempfile.mkdtemp(prefix="resume_chroma_")
    )
    os.makedirs(persist_dir, exist_ok=True)

    logger.info("Building vector store: %d chunks → %s", len(chunks), persist_dir)
    vs = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="resume_chunks",
    )

    if cache_key is not None:
        with _VS_CACHE_LOCK:
            if len(_VS_CACHE) >= settings.vector_store_cache_max:
                evicted = next(iter(_VS_CACHE))
                _VS_CACHE.pop(evicted)
                logger.debug("Vector store cache evicted key %s…", evicted[:8])
            _VS_CACHE[cache_key] = vs

    return vs


# ── JD query generation ───────────────────────────────────────────────────────


def _extract_skill_keywords(job_description: str, k: int = 12) -> List[str]:
    """Lightweight local keyword extraction — no LLM required.

    Prefers longer, technical tokens and ranks by frequency.
    """
    import re

    STOP_WORDS = {
        "the", "and", "to", "of", "in", "a", "for", "with", "on", "is",
        "are", "as", "an", "or", "will", "be", "you", "our", "at", "by",
        "from", "that", "this", "it", "they", "them", "their", "may",
        "must", "should", "role", "responsibilities", "requirements",
        "experience", "work", "using", "use", "team", "strong", "ability",
        "knowledge", "years", "proven", "background", "good",
    }

    words = re.findall(r"[a-z0-9][a-z0-9+.#_\-]{1,}", job_description.lower())
    freq: Dict[str, int] = {}
    for w in words:
        token = w.strip("_-#.")
        if not token or token in STOP_WORDS or len(token) < 3:
            continue
        freq[token] = freq.get(token, 0) + 1

    ranked = sorted(freq, key=lambda w: (freq[w], len(w)), reverse=True)
    seen: set[str] = set()
    out: List[str] = []
    for kw in ranked:
        if kw not in seen:
            seen.add(kw)
            out.append(kw)
        if len(out) >= k:
            break
    return out


def _build_retrieval_queries(job_description: str, n_queries: int = 3) -> List[str]:
    """Generate *n_queries* semantically diverse queries for multi-query RAG.

    Strategy:
    1. Full JD (dense semantic signal)
    2. Top-k extracted skill keywords (lexical / technical signal)
    3. First 512 chars of JD (intro / seniority signal)
    """
    queries: List[str] = []

    # Query 1: full job description (truncated for embedding model token limits)
    queries.append(job_description[:2000])

    # Query 2: top technical keywords joined as a skill phrase
    keywords = _extract_skill_keywords(job_description, k=15)
    if keywords:
        queries.append(" ".join(keywords))

    # Query 3: opening paragraph (often contains seniority / role summary)
    opening = job_description.strip()[:512]
    if opening and opening not in queries:
        queries.append(opening)

    return queries[:n_queries]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    ranked_lists: List[List[str]], k: int = 60
) -> List[str]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score = Σ  1 / (k + rank_i)   for each list that contains the document.

    Documents with the highest combined RRF scores appear first in the output.
    Duplicate texts are deduplicated while preserving fusion order.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)

    sorted_docs = sorted(scores, key=lambda d: scores[d], reverse=True)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for doc in sorted_docs:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)
    return unique


# ── Public retrieval API ──────────────────────────────────────────────────────


def retrieve_relevant_chunks(
    *,
    job_description: str,
    resume_chunks: List[str],
    embeddings_model_name: str,
    k: int = 6,
    resume_pdf_bytes: Optional[bytes] = None,
) -> RetrievedContext:
    """Multi-query semantic retrieval with Reciprocal Rank Fusion.

    Runs ``n_queries`` independent similarity searches against the resume's
    vector store, then merges the ranked result lists via RRF to produce a
    single, deduplicated, high-recall set of relevant chunks.

    Args:
        job_description: Raw JD text used to generate queries.
        resume_chunks: Pre-chunked resume text strings.
        embeddings_model_name: HuggingFace model identifier.
        k: Number of chunks to retrieve *per query*.
        resume_pdf_bytes: Raw PDF bytes for cache-keying (optional).

    Returns:
        :class:`RetrievedContext` with the fused, ranked chunks.
    """
    vector_store = make_vector_store(
        resume_chunks,
        embeddings_model_name=embeddings_model_name,
        resume_pdf_bytes=resume_pdf_bytes,
    )

    queries = _build_retrieval_queries(job_description, n_queries=settings.retrieval_queries)
    logger.info("Multi-query retrieval: %d queries × top-%d chunks", len(queries), k)

    per_query_results: List[List[str]] = []
    for i, query in enumerate(queries):
        hits = vector_store.similarity_search(query, k=k)
        texts = [h.page_content for h in hits if h.page_content.strip()]
        logger.debug("Query %d returned %d chunks", i + 1, len(texts))
        per_query_results.append(texts)

    fused = _reciprocal_rank_fusion(per_query_results)
    logger.info("RRF fusion complete: %d unique chunks from %d queries", len(fused), len(queries))

    return RetrievedContext(
        retrieved_chunks=fused,
        all_chunks_preview=resume_chunks[:3],
        query_count=len(queries),
    )
