<div align="center">

# 🤖 Agentic Resume Analyzer

### AI-powered resume-to-job-description matching · 100% local · No API keys required

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Pipeline-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![CI](https://img.shields.io/github/actions/workflow/status/msaquib21/resume-analyzer-app/ci.yml?style=for-the-badge&label=CI&logo=githubactions&logoColor=white)](https://github.com/msaquib21/resume-analyzer-app/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **Upload your resume PDF + paste a Job Description → Get an instant AI-powered match score, keyword heatmap, identified skill gaps, actionable resume improvements, and a personalised interview preparation roadmap — all running 100% locally.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Core Features](#-core-features)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Docker Setup](#-docker-setup)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)

---

## 🎯 Overview

**Agentic Resume Analyzer** is a full-stack AI application that uses a **multi-step LangGraph agentic pipeline** to intelligently compare a candidate's resume against a target job description. Unlike simple keyword matchers, it uses **local LLM inference** (via [Ollama](https://ollama.com/)) combined with **Retrieval-Augmented Generation (RAG)** to provide deep, contextual analysis — all running entirely on your own machine with **zero cloud dependencies or API costs**.

The system produces five key deliverables displayed across a polished dark-themed Streamlit UI:
1. **Match Score (0–10)** — Calibrated against the JD's actual requirements with animated SVG ring
2. **JD Keyword Heatmap** — Visual grid showing which JD keywords exist in your resume vs. which are missing
3. **Identified Skill Gaps** — Specific skills missing from the resume with JD evidence and confidence levels
4. **Actionable Resume Improvements** — Precise directives telling you *what to add/change* in your resume
5. **Interview Preparation Roadmap** — Study resources, hands-on projects, and mock interview questions

Plus: **Analysis History & Trends** dashboard, **PDF Report Export**, **Feedback System**, and full **Docker Compose** containerization.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) `≥1.33` | Interactive multi-page UI with SSE streaming, sidebar navigation |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | Async REST API with SSE, rate limiting, history CRUD |
| **LLM Runtime** | [Ollama](https://ollama.com/) (`qwen2.5:3b`, `qwen2.5:7b`, `qwen3:8b`, `llama3.1:8b`) | 100% local LLM inference with dynamic model swapping |
| **Agentic Pipeline** | [LangGraph](https://langchain-ai.github.io/langgraph/) `≥0.1` | Stateful multi-node graph-based agent orchestration |
| **LLM Integration** | [LangChain](https://python.langchain.com/) / [langchain-community](https://github.com/langchain-ai/langchain) | Ollama LLM wrapper, output parsers, BM25 retriever |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` | Local semantic embedding (no GPU required) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | In-memory vector store with persistent LRU caching |
| **Hybrid Search** | [rank-bm25](https://github.com/dorianbrown/rank_bm25) + ChromaDB | Dense semantic + sparse lexical search fused via weighted RRF |
| **PDF Parsing** | [PyPDF](https://pypdf.readthedocs.io/) + Section-Aware Parser | Multi-column regex header detection preserving dense skill sections |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Production-grade tracing via `LANGCHAIN_TRACING_V2` |
| **PDF Export** | [fpdf2](https://py-pdf.github.io/fpdf2/) | Branded PDF report generation with color-coded sections |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) v2 + [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Strict I/O schemas and environment config |
| **Persistence** | SQLite (built-in) | Analysis history, feedback, score trends |
| **Streaming** | [sse-starlette](https://github.com/sysid/sse-starlette) + [httpx](https://www.python-httpx.org/) | Server-Sent Events for real-time progress |
| **Containerization** | [Docker](https://www.docker.com/) + [Docker Compose](https://docs.docker.com/compose/) | One-command deployment with Ollama |
| **CI/CD** | [GitHub Actions](https://github.com/features/actions) | Automated linting (ruff) and testing (pytest) |
| **Language** | Python 3.10+ | Core application language |

---

## ✨ Core Features

### 🔍 Skill Gap Analysis
- Identifies **3 critical gaps** between the resume and JD with structured evidence
- Each gap includes: **what's missing**, **what the JD demands**, and **the real-world consequence**
- Color-coded severity levels (High / Medium / Low) based on overall match score

### 🔑 JD Keyword Heatmap (ATS Simulation)
- Extracts 20+ skill/tool/certification keywords from the Job Description
- Cross-references each keyword against the resume text
- Visual chip grid: ✓ green for found, ✗ red for missing
- Shows match percentage for quick ATS compatibility assessment

### ⚡ Actionable Resume Improvements
- **Direct action items**: *"Add a bullet to your Experience section demonstrating X..."*
- Each card includes: **Target Area**, **Action Required**, **JD Alignment** explanation

### 🎯 Interview Preparation Roadmap
- Structured per identified gap: **Study** resources, **Practice** projects, **Interview Angle** questions

### 📊 Analysis History & Trends
- Every analysis automatically saved to local SQLite database
- **Score trend chart** showing improvement over time
- Browse, review, and delete past analyses
- Compare performance across different JDs

### 📄 PDF Report Export
- One-click branded PDF download with score, gaps, improvements, and preparation
- Color-coded sections, clean typography, professional formatting

### 👍 Feedback System
- Thumbs up/down buttons on each gap, improvement, and preparation card
- Feedback persisted per analysis for future reference

### 🚀 Performance & Reliability
- **Rate limiting**: Max 3 concurrent analyses via `asyncio.Semaphore`
- **Self-healing JSON parser**: 4-stage fallback chain handles malformed LLM output
- **Multi-query RAG with RRF**: 3 query angles fused via Reciprocal Rank Fusion
- **Single consolidated LLM call**: All analysis in one inference step
- **Vector store caching**: ChromaDB instance cached per resume hash
- **SSE streaming**: Real-time pipeline progress in the UI

### 🎨 Modern Dark UI
- Deep navy-black base (`#060914`) with animated aurora gradient background
- Neon Mint (`#00FFA3`), Hot Rose (`#FF4060`), Electric Violet (`#B44DFF`) accents
- Glassmorphism cards with micro-animations on hover
- Animated SVG score ring with color-coded glow
- Sidebar navigation with system health indicator

---

## ⚙ System Architecture

```
┌──────────────────────────────────────────────────────┐
│                     USER BROWSER                      │
│  Streamlit UI (port 8501)                            │
│  ├── 🔬 New Analysis (upload + analyze)              │
│  └── 📊 History & Trends (past analyses)             │
└───────────────────┬──────────────────────────────────┘
                    │ HTTP (SSE stream)
                    ▼
┌──────────────────────────────────────────────────────┐
│           FastAPI Backend (port 8000)                 │
│                                                      │
│  Endpoints:                                          │
│  ├── POST /analyze/stream   (SSE analysis)           │
│  ├── GET  /history          (list analyses)          │
│  ├── GET  /history/trend    (score chart data)       │
│  ├── POST /export/pdf       (PDF report)             │
│  ├── POST /feedback         (thumbs up/down)         │
│  └── GET  /health           (system status)          │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │         LangGraph StateGraph               │      │
│  │                                            │      │
│  │  Node 1: PDF → Chunks → Embeddings → RAG   │      │
│  │  Node 2: LLM Analysis (score/gaps/actions) │      │
│  │  + Keyword Extraction (regex + NER-style)  │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐       │
│  │ ChromaDB │  │ SQLite   │  │ fpdf2        │       │
│  │ (vectors)│  │ (history)│  │ (PDF export) │       │
│  └──────────┘  └──────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────┘
                    │
                    ▼ HTTP
┌──────────────────────────────────────────────────────┐
│         Ollama (port 11434)                          │
│         qwen2.5:3b local LLM                         │
└──────────────────────────────────────────────────────┘
```

---

## 📦 Installation & Setup

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.12 recommended |
| [Ollama](https://ollama.com/download) | Latest | Must be running locally |
| Git | Any | For cloning |

### 1. Clone the Repository

```bash
git clone https://github.com/msaquib21/resume-analyzer-app.git
cd resume-analyzer-app
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the LLM Model via Ollama

```bash
ollama pull qwen2.5:3b
```

### 5. (Optional) Create a `.env` File

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TEMPERATURE=0.0
```

---

## 🚀 Running the Application

Open **two terminals** in the project root:

### Terminal 1 — Backend API

```bash
uvicorn backend.server:app --reload --port 8000
```

### Terminal 2 — Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker Setup

Run the entire stack with one command:

```bash
docker-compose up --build
```

This starts:
- **Ollama** (port 11434) with persistent model storage
- **App** (ports 8000 + 8501) with both backend and frontend

After startup, pull the model inside the Ollama container:
```bash
docker exec -it resume-analyzer-app-ollama-1 ollama pull qwen2.5:3b
```

Then open **http://localhost:8501**.

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest ruff

# Run all tests
pytest tests/ -v

# Run linter
ruff check backend/ tests/
```

### Test Coverage

| Test File | Tests | What's Covered |
|---|---|---|
| `test_parser.py` | 8 tests | `_normalize_list_item` (6 input types), `_clean_json_str` (2 formats) |
| `test_history.py` | 3 tests | Save/retrieve, delete, score trend operations |

### CI/CD

Every push to `main` and every PR triggers the GitHub Actions pipeline:
1. **Lint** with `ruff`
2. **Test** with `pytest`

---

## 🔧 Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | LLM model tag |
| `OLLAMA_TEMPERATURE` | `0.0` | Generation temperature |
| `EMBEDDINGS_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` | `500` | PDF text chunk size |
| `RETRIEVAL_K` | `6` | Chunks per query |
| `RETRIEVAL_QUERIES` | `3` | Multi-query RAG queries |
| `MAX_CONCURRENT_ANALYSES` | `3` | Rate limit |
| `HISTORY_DB_PATH` | `data/history.db` | SQLite DB path |
| `GRAPH_TIMEOUT_SECONDS` | `300` | Pipeline timeout |

---

## 📁 Project Structure

```
resume-analyzer-app/
│
├── backend/                        # FastAPI + LangGraph backend
│   ├── __init__.py
│   ├── agent.py                    # LangGraph pipeline, prompt engineering, keyword extraction
│   ├── config.py                   # Pydantic Settings (env-configurable)
│   ├── history.py                  # SQLite analysis history store
│   ├── models.py                   # Pydantic I/O models (15+ schemas)
│   ├── pdf_export.py               # fpdf2 branded PDF report generator
│   ├── rag.py                      # PDF parsing, chunking, ChromaDB, multi-query RAG
│   └── server.py                   # FastAPI app, 8 endpoints, SSE streaming, rate limiting
│
├── frontend/
│   └── app.py                      # Streamlit UI: multi-page, CSS design system, SSE consumer
│
├── tests/
│   ├── __init__.py
│   ├── test_parser.py              # Unit tests for self-healing JSON parser
│   └── test_history.py             # Unit tests for SQLite history store
│
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions CI pipeline (ruff + pytest)
│
├── data/                           # Auto-created: SQLite history database
├── Dockerfile                      # Python 3.12-slim container
├── docker-compose.yml              # Ollama + App orchestration
├── .dockerignore
├── .gitignore
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health + Ollama status |
| `POST` | `/analyze` | Synchronous analysis |
| `POST` | `/analyze/stream` | SSE streaming analysis (recommended) |
| `GET` | `/history` | List all past analyses |
| `GET` | `/history/trend` | Score trend data for charting |
| `GET` | `/history/{id}` | Single analysis detail |
| `DELETE` | `/history/{id}` | Delete an analysis |
| `POST` | `/feedback` | Submit thumbs up/down feedback |
| `POST` | `/export/pdf` | Generate PDF report |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [msaquib21](https://github.com/msaquib21) · Powered by Ollama · LangGraph · Streamlit

⭐ **Star this repo** if you found it useful for your job search!

</div>
