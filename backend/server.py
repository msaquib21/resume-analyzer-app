"""
FastAPI server for the Agentic Resume Analyzer.

Endpoints
---------
GET  /health          Liveness + dependency check (Ollama reachable, model configured).
POST /analyze         Synchronous full analysis; returns ``AnalysisResponse``.
POST /analyze/stream  SSE streaming version — emits progress events during the
                      LangGraph pipeline and a final ``result`` event.

Design notes
------------
- All configuration is sourced from ``Settings`` (env-var overridable).
- Logging uses the stdlib ``logging`` module with structured messages.
- The LangGraph graph is compiled once at startup and reused across requests.
- ``asyncio.to_thread`` keeps the sync graph execution off the event loop.
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
from fastapi.responses import StreamingResponse

from .agent import build_resume_analysis_graph
from .config import settings
from .models import AnalysisResponse, ErrorDetail, HealthResponse

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── App lifespan (startup / shutdown) ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Agentic Resume Analyzer backend")
    logger.info("Ollama URL  : %s", settings.ollama_base_url)
    logger.info("LLM model   : %s", settings.ollama_model)
    logger.info("Embeddings  : %s", settings.embeddings_model)
    yield
    logger.info("Backend shutting down")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic Resume Analyzer",
    description=(
        "Local-only resume analysis powered by LangGraph + RAG + Ollama. "
        "No data leaves your machine."
    ),
    version="1.0.0",
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
        "score": 0,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────


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


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Synchronous resume analysis",
    tags=["Analysis"],
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorDetail},
        status.HTTP_504_GATEWAY_TIMEOUT: {"model": ErrorDetail},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorDetail},
    },
)
async def analyze(
    job_description: str = Form(..., description="Full job description text."),
    resume_pdf: UploadFile = File(..., description="Candidate's resume as a PDF."),
) -> AnalysisResponse:
    """Run the full 3-node LangGraph analysis and return a structured result."""
    logger.info("POST /analyze — file=%s size=~%s", resume_pdf.filename, resume_pdf.size)
    t0 = time.perf_counter()

    pdf_bytes = await resume_pdf.read()
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded PDF is empty.",
        )

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
            detail=f"Analysis timed out after {settings.graph_timeout_seconds}s. Try a smaller PDF or a faster model.",
        )
    except Exception as exc:
        logger.exception("Graph execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline failed: {exc}",
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "POST /analyze — done in %.1fs | score=%s gaps=%d improvements=%d",
        elapsed,
        result.get("score"),
        len(result.get("gaps", [])),
        len(result.get("improvements", [])),
    )

    return AnalysisResponse(
        score=result["score"],
        gaps=result["gaps"],
        improvements=result["improvements"],
        preparation=result["preparation"],
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
    """Server-Sent Events endpoint.

    Emits ``data: <json>`` lines with the shape::

        {"event": "<stage>", "message": "<human text>"}

    Terminates with a ``result`` event containing the full ``AnalysisResponse``.
    """
    logger.info("POST /analyze/stream — file=%s", resume_pdf.filename)
    pdf_bytes = await resume_pdf.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded PDF is empty.",
        )

    event_queue: queue.Queue = queue.Queue()

    def _progress_callback(stage: str, message: str) -> None:
        event_queue.put({"event": stage, "message": message})

    payload = _build_graph_payload(job_description, pdf_bytes, callback=_progress_callback)

    async def _event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
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
            response = AnalysisResponse(
                score=result["score"],
                gaps=result["gaps"],
                improvements=result["improvements"],
                preparation=result["preparation"],
            )
            yield _fmt({"event": "result", "data": response.model_dump()})
            logger.info("SSE stream complete — score=%d", response.score)
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
