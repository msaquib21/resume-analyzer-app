"""
RAG pipeline: Layout-aware PDF parsing, chunking, embedding, and multi-query retrieval.

Key architectural optimizations
-------------------------------
- **Layout-Aware PDF Parsing**: Prevents multi-column resumes from merging columns
  horizontally (e.g. skills like "Redis (Vector DB)", "OCI" being joined into experience text).
- **20% Chunk Overlap**: Preserves full contextual sentences and bullet points across boundaries.
- **High Recall Multi-Query Retrieval**: Executes diverse semantic queries fused via RRF,
  with an expanded Top-K window to guarantee full candidate coverage for the LLM.
- **Embedding & Vector-Store Caching**: In-memory LRU Chroma caching keyed by SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pypdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
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


# ── Layout-Aware Multi-Column PDF Parsing ──────────────────────────────────────


def _parse_multicolumn_layout(layout_text: str) -> str:
    """Disentangle multi-column resume layouts into logical reading order.

    In two-column resumes, naive text extractors read lines horizontally across
    the page, jumbling separate sections (e.g. left sidebar skills with right-hand
    work experience). This function detects vertical whitespace gutters between
    columns and reconstructs the text column-by-column in top-to-bottom order.
    """
    lines = layout_text.splitlines()
    if not lines:
        return ""

    # Filter lines that have substantial text and wide whitespace gaps
    long_lines = [l for l in lines if len(l.strip()) > 25 and "   " in l]
    if len(long_lines) < 3:
        # Single column or irregular: collapse excess spaces within lines
        return "\n".join(re.sub(r" {2,}", " ", l).strip() for l in lines if l.strip())

    # Find positions of vertical whitespace gutters (>= 3 consecutive spaces)
    gap_positions: List[int] = []
    for l in long_lines:
        for m in re.finditer(r" {3,}", l):
            gap_positions.append((m.start() + m.end()) // 2)

    if not gap_positions:
        return "\n".join(re.sub(r" {2,}", " ", l).strip() for l in lines if l.strip())

    # Bin gutter positions in 6-character intervals to find dominant column split
    binned = Counter(p // 6 * 6 for p in gap_positions)
    best_bin, count = binned.most_common(1)[0]

    # If at least 35% of multi-space lines share this column gutter:
    if count >= len(long_lines) * 0.35:
        split_x = best_bin + 3
        col1_lines: List[str] = []
        col2_lines: List[str] = []
        for l in lines:
            if len(l) > split_x:
                c1 = l[:split_x].strip()
                c2 = l[split_x:].strip()
                if c1:
                    col1_lines.append(c1)
                if c2:
                    col2_lines.append(c2)
            else:
                c1 = l.strip()
                if c1:
                    col1_lines.append(c1)

        # Output Column 1 top-to-bottom, followed by Column 2 top-to-bottom
        return "\n".join(col1_lines) + "\n\n" + "\n".join(col2_lines)

    return "\n".join(re.sub(r" {2,}", " ", l).strip() for l in lines if l.strip())


def extract_layout_aware_text(pdf_bytes: bytes) -> str:
    """Extract full text from PDF bytes with layout awareness and column separation."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    page_texts: List[str] = []

    for i, page in enumerate(reader.pages):
        raw_text = ""
        # 1. Attempt layout-aware extraction
        try:
            raw_text = page.extract_text(extraction_mode="layout") or ""
        except Exception as exc:
            logger.debug("Page %d layout extraction failed: %s; trying plain mode", i + 1, exc)

        # 2. Fallback to plain extraction if layout mode returned nothing
        if not raw_text.strip():
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning("Page %d plain extraction failed: %s", i + 1, exc)

        if raw_text.strip():
            clean_page = _parse_multicolumn_layout(raw_text)
            page_texts.append(clean_page)

    full_resume_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
    logger.info("PDF extraction complete: %d pages, %d characters", len(reader.pages), len(full_resume_text))
    return full_resume_text


# ── Section-Aware Resume Chunking ─────────────────────────────────────────────

SECTION_HEADER_RE = re.compile(
    r"(?m)^[ \t]*(?:[#*\-•\d\.\)]+[ \t]*)*("
    r"(?:TECHNICAL\s+|CORE\s+|KEY\s+)?SKILLS(?:\s*(?:&|AND|\/)\s*(?:EXPERTISE|TECHNOLOGIES|ABILITIES|TOOLS|COMPETENCIES))?|"
    r"CORE\s+COMPETENCIES|AREAS\s+OF\s+EXPERTISE|TECHNOLOGIES(?:\s*&|\s*AND|\s*\/)?(?:\s*TOOLS)?|"
    r"(?:WORK\s+|PROFESSIONAL\s+|RELEVANT\s+|EMPLOYMENT\s+)?EXPERIENCE|EMPLOYMENT\s+HISTORY|"
    r"(?:KEY\s+|FEATURED\s+|SELECTED\s+|PERSONAL\s+|ACADEMIC\s+|TECHNICAL\s+)?PROJECTS|"
    r"EDUCATION(?:\s*(?:&|AND|\/)\s*(?:TRAINING|ACADEMICS))?|ACADEMIC\s+BACKGROUND|"
    r"CERTIFICATIONS?|LICENSES?(?:\s*(?:&|AND|\/)\s*CERTIFICATIONS?)?|"
    r"(?:PROFESSIONAL\s+|CAREER\s+|EXECUTIVE\s+)?(?:SUMMARY|OBJECTIVE|PROFILE)|"
    r"PUBLICATIONS|AWARDS|ACHIEVEMENTS|HONORS|ACTIVITIES"
    r")\s*[:\-\u2013\u2014]?[ \t]*(?:\r?\n|$)",
    re.IGNORECASE,
)


def section_aware_chunk_resume(
    full_text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> List[str]:
    """Segment resume into logical sections and chunk intelligently without fracturing skills.

    Dense sections (e.g. Skills, Technologies, Certifications) are kept cohesive
    in single chunks up to 1800 characters so technical terms are never split mid-phrase.
    Longer narrative sections (Experience, Projects) are split at paragraph and bullet
    boundaries while prepending the section context header.
    """
    text = full_text.strip()
    if not text:
        return []

    matches = list(SECTION_HEADER_RE.finditer(text))
    # Fallback to paragraph splitter if unstructured or fewer than 2 headers detected
    if len(matches) < 2:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n--- PAGE BREAK ---\n\n", "\n\n", "\n", ". ", " ", ""],
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]

    sections: List[tuple[str, str]] = []
    first_start = matches[0].start()
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections.append(("Candidate Overview", preamble))

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((header, content))

    chunks: List[str] = []
    dense_keywords = {"skill", "technolog", "competenc", "tool", "certificat"}

    for header, content in sections:
        norm_header = header.title()
        is_dense = any(dk in header.lower() for dk in dense_keywords)

        # Dense keyword lists: keep intact to prevent breaking technical phrases
        if is_dense and len(content) <= 1800:
            chunks.append(f"[{norm_header}]\n{content}")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n• ", "\n- ", "\n* ", "\n", ". ", " "],
            )
            sub_chunks = splitter.split_text(content)
            for sc in sub_chunks:
                sc_clean = sc.strip()
                if sc_clean:
                    chunks.append(f"[{norm_header}]\n{sc_clean}")

    return chunks


def load_and_chunk_pdf(pdf_bytes: bytes) -> List[str]:
    """Extract text using layout-aware parsing and split using section-aware chunking."""
    full_text = extract_layout_aware_text(pdf_bytes)
    if not full_text.strip():
        logger.warning("PDF extraction returned empty text.")
        return []

    clean_chunks = section_aware_chunk_resume(
        full_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info(
        "PDF section-chunked: %d chunks (chunk_size=%d, overlap=%d)",
        len(clean_chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return clean_chunks


# ── Embedding & Vector Store Caching ─────────────────────────────────────────

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
    """Build (or retrieve from cache) a Chroma vector store for *chunks*."""
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


# ── JD Query Generation ───────────────────────────────────────────────────────


def _extract_skill_keywords(job_description: str, k: int = 16) -> List[str]:
    """Lightweight local keyword extraction — no LLM required."""
    STOP_WORDS = {
        "the", "and", "to", "of", "in", "a", "for", "with", "on", "is",
        "are", "as", "an", "or", "will", "be", "you", "our", "at", "by",
        "from", "that", "this", "it", "they", "them", "their", "may",
        "must", "should", "role", "responsibilities", "requirements",
        "experience", "work", "using", "use", "team", "strong", "ability",
        "knowledge", "years", "proven", "background", "good", "plus",
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


def _build_retrieval_queries(job_description: str, n_queries: int = 4) -> List[str]:
    """Generate *n_queries* semantically diverse queries for multi-query RAG."""
    queries: List[str] = []

    # Query 1: full job description (core requirements focus)
    queries.append(job_description[:2000])

    # Query 2: top technical keywords joined as a skills cluster
    keywords = _extract_skill_keywords(job_description, k=16)
    if keywords:
        queries.append(" ".join(keywords))

    # Query 3: opening paragraph (seniority / title / domain focus)
    opening = job_description.strip()[:600]
    if opening and opening not in queries:
        queries.append(opening)

    # Query 4: secondary technical keywords
    if len(keywords) > 8:
        queries.append(" ".join(keywords[8:24]))

    return queries[:n_queries]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[str]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Supports optional weighting per ranked list to calibrate dense vector
    and sparse lexical (BM25) search contributions.
    """
    scores: Dict[str, float] = {}
    for idx, ranked in enumerate(ranked_lists):
        w = weights[idx] if weights and idx < len(weights) else 1.0
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)

    sorted_docs = sorted(scores, key=lambda d: scores[d], reverse=True)
    seen: set[str] = set()
    unique: List[str] = []
    for doc in sorted_docs:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)
    return unique


# ── Public Retrieval API ──────────────────────────────────────────────────────


def retrieve_relevant_chunks(
    *,
    job_description: str,
    resume_chunks: List[str],
    embeddings_model_name: str,
    k: int = 10,
    resume_pdf_bytes: Optional[bytes] = None,
) -> RetrievedContext:
    """Multi-query Hybrid retrieval (Dense ChromaDB + Sparse BM25) with Reciprocal Rank Fusion."""
    if not resume_chunks:
        return RetrievedContext(retrieved_chunks=[], all_chunks_preview=[], query_count=0)

    # 1. Initialize Dense Vector Store (Chroma)
    vector_store = make_vector_store(
        resume_chunks,
        embeddings_model_name=embeddings_model_name,
        resume_pdf_bytes=resume_pdf_bytes,
    )

    # 2. Initialize Sparse BM25 Retriever
    bm25_retriever = None
    try:
        bm25_retriever = BM25Retriever.from_texts(resume_chunks, k=k)
    except Exception as exc:
        logger.warning("BM25Retriever initialization failed: %s; falling back to dense only", exc)

    queries = _build_retrieval_queries(job_description, n_queries=settings.retrieval_queries)
    logger.info(
        "Hybrid multi-query retrieval: %d queries × top-%d (BM25 weight=%.2f)",
        len(queries),
        k,
        settings.bm25_weight,
    )

    ranked_lists: List[List[str]] = []
    weights: List[float] = []

    for i, query in enumerate(queries):
        # Dense ChromaDB similarity search
        dense_hits = vector_store.similarity_search(query, k=k)
        dense_texts = [h.page_content for h in dense_hits if h.page_content.strip()]
        ranked_lists.append(dense_texts)
        weights.append(1.0)

        # Sparse BM25 keyword search
        if bm25_retriever is not None:
            try:
                bm25_hits = bm25_retriever.invoke(query)
                bm25_texts = [h.page_content for h in bm25_hits if h.page_content.strip()]
                ranked_lists.append(bm25_texts)
                weights.append(settings.bm25_weight)
            except Exception as exc:
                logger.debug("BM25 retrieval failed for query %d: %s", i + 1, exc)

    fused = _reciprocal_rank_fusion(ranked_lists, k=60, weights=weights)
    logger.info(
        "Hybrid RRF fusion complete: %d unique chunks retrieved from %d search passes",
        len(fused),
        len(ranked_lists),
    )

    return RetrievedContext(
        retrieved_chunks=fused,
        all_chunks_preview=resume_chunks[:3],
        query_count=len(queries),
    )
