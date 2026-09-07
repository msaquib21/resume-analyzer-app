"""
FastAPI server for the Agentic Resume Analyzer.

Endpoints
---------
GET  /health              Liveness + dependency check.
POST /analyze             Synchronous full analysis; returns ``AnalysisResponse``.
POST /analyze/stream      SSE streaming version with live progress events.
GET  /history             List all past analyses.
GET  /history/{id}        Single analysis detail.
DELETE /history/{id}      Delete an analysis.
GET  /history/trend       Score trend data for charting.
POST /feedback            Submit thumbs-up/down feedback on analysis cards.
POST /export/pdf          Generate and download a PDF report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .agent import build_resume_analysis_graph
from .config import settings
from .history import (
    AnalysisRecord,
    delete_analysis,
    get_all_analyses,
    get_analysis,
    get_score_trend,
    save_analysis,
    save_feedback,
)
from .models import (
    AnalysisDetail,
    AnalysisHistoryItem,
    AnalysisResponse,
    ErrorDetail,
    FeedbackRequest,
    HealthResponse,
    KeywordMatch,
    ScoreTrendPoint,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rate limiting semaphore ───────────────────────────────────────────────────
_analysis_semaphore = asyncio.Semaphore(settings.max_concurrent_analyses)


# ── App lifespan (startup / shutdown) ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic Resume Analyzer backend")
    logger.info("Ollama URL  : %s", settings.ollama_base_url)
    logger.info("LLM model   : %s", settings.ollama_model)
    logger.info("Embeddings  : %s", settings.embeddings_model)
    logger.info("Max concurrent: %d", settings.max_concurrent_analyses)
    yield
    logger.info("Backend shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Resume Analyzer",
    description=(
        "Local-only resume analysis powered by LangGraph + RAG + Ollama. "
        "No data leaves your machine."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compiled once at startup — shared across all requests.
graph = build_resume_analysis_graph(model=settings.ollama_model)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_graph_payload(job_description: str, pdf_bytes: bytes, callback=None) -> dict:
    """Assemble the initial LangGraph state dict."""
    return {
        "job_description": job_description,
        "resume_pdf_bytes": pdf_bytes,
        "embeddings_model_name": settings.embeddings_model,
        "ollama_model_name": settings.ollama_model,
        "progress_callback": callback,
        # Default outputs (overwritten by nodes)
        "resume_chunks": [],
        "retrieved_chunks": [],
        "gap_analysis_text": "",
        "gaps": [],
        "improvements": [],
        "preparation": [],
        "keywords": [],
        "score": 0,
    }


def _build_keyword_matches(raw_keywords: list) -> list[KeywordMatch]:
    """Convert raw keyword dicts to KeywordMatch models."""
    result = []
    for kw in (raw_keywords or []):
        if isinstance(kw, dict):
            result.append(KeywordMatch(
                keyword=kw.get("keyword", ""),
                found_in_resume=kw.get("found_in_resume", False),
                category=kw.get("category", "skill"),
            ))
    return result


# ── Endpoints: Meta ───────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and dependency check",
    tags=["Meta"],
)
async def health() -> HealthResponse:
    """Check that the server is running and Ollama is reachable."""
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_reachable=ollama_ok,
        model=settings.ollama_model,
        embeddings_model=settings.embeddings_model,
    )


# ── Endpoints: Analysis ──────────────────────────────────────────────────────


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Synchronous resume analysis",
    tags=["Analysis"],
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorDetail},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ErrorDetail},
        status.HTTP_504_GATEWAY_TIMEOUT: {"model": ErrorDetail},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorDetail},
    },
)
async def analyze(
    job_description: str = Form(..., description="Full job description text."),
    resume_pdf: UploadFile = File(..., description="Candidate's resume as a PDF."),
) -> AnalysisResponse:
    """Run the full LangGraph analysis with rate limiting."""
    logger.info("POST /analyze — file=%s size=~%s", resume_pdf.filename, resume_pdf.size)

    pdf_bytes = await resume_pdf.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded PDF is empty.",
        )

    # Rate limiting
    if _analysis_semaphore.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Server is busy — max {settings.max_concurrent_analyses} concurrent analyses allowed.",
        )

    t0 = time.perf_counter()
    async with _analysis_semaphore:
        payload = _build_graph_payload(job_description, pdf_bytes)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(graph.invoke, payload),
                timeout=settings.graph_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error("Graph timed out after %ds", settings.graph_timeout_seconds)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Analysis timed out after {settings.graph_timeout_seconds}s.",
            )
        except Exception as exc:
            logger.exception("Graph execution failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis pipeline failed: {exc}",
            )

    elapsed = time.perf_counter() - t0
    keywords = _build_keyword_matches(result.get("keywords", []))

    # Save to history
    record = AnalysisRecord(
        resume_filename=resume_pdf.filename or "unknown.pdf",
        jd_snippet=job_description[:500],
        score=result["score"],
        gaps=result["gaps"],
        improvements=result["improvements"],
        preparation=result["preparation"],
        keywords=[kw.model_dump() for kw in keywords],
        elapsed_seconds=round(elapsed, 2),
    )
    analysis_id = save_analysis(record)

    logger.info(
        "POST /analyze — done in %.1fs | score=%s gaps=%d keywords=%d",
        elapsed, result.get("score"), len(result.get("gaps", [])), len(keywords),
    )

    return AnalysisResponse(
        analysis_id=analysis_id,
        score=result["score"],
        gaps=result["gaps"],
        improvements=result["improvements"],
        preparation=result["preparation"],
        keywords=keywords,
        elapsed_seconds=round(elapsed, 2),
    )


@app.post(
    "/analyze/stream",
    summary="Streaming resume analysis (SSE)",
    tags=["Analysis"],
    response_class=StreamingResponse,
)
async def analyze_stream(
    job_description: str = Form(...),
    resume_pdf: UploadFile = File(...),
) -> StreamingResponse:
    logger.info(
        "POST /analyze/stream — file=%s size=~%s | JD length=%d (preview: %.80r)",
        resume_pdf.filename,
        resume_pdf.size,
        len(job_description),
        job_description[:80],
    )
    pdf_bytes = await resume_pdf.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded PDF is empty.",
        )

    if _analysis_semaphore.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Server is busy — max {settings.max_concurrent_analyses} concurrent analyses allowed.",
        )

    resume_filename = resume_pdf.filename or "unknown.pdf"
    event_queue: queue.Queue = queue.Queue()

    def _progress_callback(stage: str, message: str) -> None:
        event_queue.put({"event": stage, "message": message})

    payload = _build_graph_payload(job_description, pdf_bytes, callback=_progress_callback)

    async def _event_generator() -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        loop = asyncio.get_running_loop()

        async with _analysis_semaphore:
            graph_future = loop.run_in_executor(None, graph.invoke, payload)

            def _fmt(obj: dict) -> str:
                return f"data: {json.dumps(obj)}\n\n"

            yield _fmt({"event": "started", "message": "Analysis pipeline started…"})

            while not graph_future.done():
                await asyncio.sleep(0.1)
                while not event_queue.empty():
                    evt = event_queue.get_nowait()
                    yield _fmt(evt)

            # Drain any remaining events
            while not event_queue.empty():
                yield _fmt(event_queue.get_nowait())

            try:
                result = await graph_future
                elapsed = time.perf_counter() - t0
                keywords = _build_keyword_matches(result.get("keywords", []))

                # Save to history
                record = AnalysisRecord(
                    resume_filename=resume_filename,
                    jd_snippet=job_description[:500],
                    score=result["score"],
                    gaps=result["gaps"],
                    improvements=result["improvements"],
                    preparation=result["preparation"],
                    keywords=[kw.model_dump() for kw in keywords],
                    elapsed_seconds=round(elapsed, 2),
                )
                analysis_id = save_analysis(record)

                response = AnalysisResponse(
                    analysis_id=analysis_id,
                    score=result["score"],
                    gaps=result["gaps"],
                    improvements=result["improvements"],
                    preparation=result["preparation"],
                    keywords=keywords,
                    elapsed_seconds=round(elapsed, 2),
                )
                yield _fmt({"event": "result", "data": response.model_dump()})
                logger.info("SSE stream complete — score=%d id=%s", response.score, analysis_id)
            except Exception as exc:
                logger.exception("SSE stream failed: %s", exc)
                yield _fmt({"event": "error", "message": str(exc)})

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Endpoints: History ────────────────────────────────────────────────────────


@app.get(
    "/history",
    response_model=list[AnalysisHistoryItem],
    summary="List all past analyses",
    tags=["History"],
)
async def list_history() -> list[AnalysisHistoryItem]:
    """Return all analyses, newest first."""
    records = get_all_analyses()
    items = []
    for r in records:
        keywords = r.get("keywords", [])
        total = len(keywords)
        found = sum(1 for k in keywords if k.get("found_in_resume"))
        items.append(AnalysisHistoryItem(
            id=r["id"],
            timestamp=r["timestamp"],
            resume_filename=r.get("resume_filename", ""),
            jd_snippet=r.get("jd_snippet", "")[:200],
            score=r.get("score", 0),
            elapsed_seconds=r.get("elapsed_seconds", 0.0),
            gap_count=len(r.get("gaps", [])),
            keyword_match_pct=round((found / total * 100) if total > 0 else 0, 1),
        ))
    return items


@app.get(
    "/history/trend",
    response_model=list[ScoreTrendPoint],
    summary="Score trend data for charting",
    tags=["History"],
)
async def score_trend() -> list[ScoreTrendPoint]:
    """Return score history for trend visualization."""
    points = get_score_trend()
    return [ScoreTrendPoint(**p) for p in points]


@app.get(
    "/history/{analysis_id}",
    response_model=AnalysisDetail,
    summary="Get single analysis detail",
    tags=["History"],
)
async def get_history_detail(analysis_id: str) -> AnalysisDetail:
    """Return full detail of a past analysis."""
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    keywords = [
        KeywordMatch(**kw) if isinstance(kw, dict) else kw
        for kw in record.get("keywords", [])
    ]
    return AnalysisDetail(
        id=record["id"],
        timestamp=record["timestamp"],
        resume_filename=record.get("resume_filename", ""),
        jd_snippet=record.get("jd_snippet", ""),
        score=record.get("score", 0),
        gaps=record.get("gaps", []),
        improvements=record.get("improvements", []),
        preparation=record.get("preparation", []),
        keywords=keywords,
        elapsed_seconds=record.get("elapsed_seconds", 0.0),
        feedback=record.get("feedback", {}),
    )


@app.delete(
    "/history/{analysis_id}",
    summary="Delete an analysis",
    tags=["History"],
)
async def delete_history(analysis_id: str) -> dict:
    """Delete a specific analysis from history."""
    deleted = delete_analysis(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"deleted": True, "id": analysis_id}


# ── Endpoints: Feedback ───────────────────────────────────────────────────────


@app.post(
    "/feedback",
    summary="Submit feedback on analysis cards",
    tags=["Feedback"],
)
async def submit_feedback(req: FeedbackRequest) -> dict:
    """Store thumbs-up/down feedback for an analysis."""
    updated = save_feedback(req.analysis_id, req.feedback)
    if not updated:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"updated": True, "analysis_id": req.analysis_id}


# ── Endpoints: PDF Export ─────────────────────────────────────────────────────


@app.post(
    "/export/pdf",
    summary="Generate PDF report",
    tags=["Export"],
)
async def export_pdf(
    analysis_id: str = Form(..., description="Analysis ID to export."),
) -> Response:
    """Generate a branded PDF report for a past analysis."""
    record = get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    # Lazy import to avoid startup cost
    from .pdf_export import generate_report

    pdf_bytes = generate_report(
        score=record.get("score", 0),
        gaps=record.get("gaps", []),
        improvements=record.get("improvements", []),
        preparation=record.get("preparation", []),
        keywords=record.get("keywords", []),
        resume_filename=record.get("resume_filename", "resume.pdf"),
        jd_snippet=record.get("jd_snippet", ""),
        elapsed_seconds=record.get("elapsed_seconds", 0.0),
    )

    filename = f"resume_analysis_{analysis_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
