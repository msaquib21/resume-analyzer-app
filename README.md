<div align="center">

# 🤖 Agentic Resume Analyzer (v2.0)

### Production-grade resume-to-job-description matching engine · 100% Local or Cloud Groq · Multi-query Hybrid RAG · LangGraph Orchestration

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Pipeline-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![Groq](https://img.shields.io/badge/Groq-Cloud_Inference-F05A28?style=for-the-badge&logo=fastapi&logoColor=white)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![CI](https://img.shields.io/github/actions/workflow/status/msaquib21/resume-analyzer-app/ci.yml?style=for-the-badge&label=CI&logo=githubactions&logoColor=white)](https://github.com/msaquib21/resume-analyzer-app/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **Upload your resume PDF + paste a Job Description → Receive an evidence-based match score, an ATS keyword heatmap, verified technical skill gaps, concrete resume bullet rewrites, and an interview preparation roadmap — powered by local LLMs or ultra-fast Groq cloud inference.**

</div>

---

## 📌 Table of Contents

1. [Overview & What's New in v2.0](#-overview--whats-new-in-v20)
2. [Key Architecture & Engineering Breakthroughs](#-key-architecture--engineering-breakthroughs)
3. [Core Application Features](#-core-application-features)
4. [Tech Stack](#-tech-stack)
5. [System Architecture Diagram](#-system-architecture-diagram)
6. [Installation & Local Setup](#-installation--local-setup)
7. [Dual-LLM Provider Modes (Ollama vs. Groq)](#-dual-llm-provider-modes-ollama-vs-groq)
8. [Running the Application](#-running-the-application)
9. [Docker Deployment](#-docker-deployment)
10. [Comprehensive Configuration Reference](#-comprehensive-configuration-reference)
11. [Testing & Quality Assurance](#-testing--quality-assurance)
12. [API Reference](#-api-reference)
13. [Project Directory Layout](#-project-directory-layout)
14. [License & Acknowledgements](#-license--acknowledgements)

---

## 🎯 Overview & What's New in v2.0

**Agentic Resume Analyzer** is an end-to-end AI career copilot designed for software engineers, data professionals, and hiring managers. Unlike basic keyword counters or generic prompt wrappers, this application uses a stateful **LangGraph pipeline**, **Hybrid RAG retrieval (BM25 lexical + ChromaDB dense vectors)**, and **evidence-grounded mathematical scoring** to objectively audit a candidate's profile against any target Job Description (JD).

### 🚀 Key Improvements in v2.0

- **Dual-LLM Provider Engine**: Seamlessly toggle between local private inference via [Ollama](https://ollama.com/) (e.g., `qwen2.5:3b`, `qwen3.5:9b`, `llama3.1`) and free, high-throughput cloud inference via [Groq Cloud](https://groq.com/) (`llama-3.1-8b-instant`) for deployment on Vercel or Render.
- **Single-Page Resume RAG Bypass**: Automatic word-count evaluation. Resumes under 1,000 words (~1 page) bypass chunking and vector search to be evaluated as a single cohesive unit, preventing fragmentation of dense skill sections.
- **Section-Aware Multi-Column PDF Parsing**: Layout-aware parsing disentangles multi-column PDF layouts vertically, preventing skills (e.g., "Redis", "Docker") and experience from horizontally coalescing into unreadable lines.
- **Symbol-Aware ATS Keyword Matcher**: Tokenizer with lookaround regex boundaries accurately captures technical symbols such as `C++`, `C#`, `.NET`, `Node.js`, and `React.js` without boundary degradation.
- **Negative Constraint Recalibration & Mathematical Rubric**: Replaced timid negative constraints with an objective technical recruiter persona and a strict mathematical scoring rubric (starting at 10/10 with standard deductions for missing core languages, frameworks, seniority, and cloud tools).
- **Empty String Schema Failure Defense**: Pydantic v2 `@field_validator(mode="before")` hooks and frontend defensive sanitizers clean dirty list objects into pure empty arrays `[]`, rendering high-contrast celebratory success cards for perfect-match categories.

---

## 🧠 Key Architecture & Engineering Breakthroughs

### 1. Mathematical Scoring Rubric & Anti-Hallucination Grounding
Large Language Models are prone to either hallucinating missing requirements or timidly defaulting to high scores ("negative constraint overcorrection"). To solve this:
- **Base Score = 10/10**: Deduct **2 to 3 points** for missing core programming languages or primary frameworks.
- **Experience Deductions**: Deduct **2 points** if the candidate lacks the required years of experience or seniority.
- **Secondary Tool Deductions**: Deduct **1 to 2 points** if secondary tools or cloud platforms are missing.
- **Threshold Rule**: Resumes missing the majority of JD requirements **must score below 4/10**. 10/10 is reserved strictly for flawless matches.

### 2. Multi-Query Hybrid RAG Pipeline
For multi-page resumes:
1. **Section-Aware Chunking**: Regex-based header detection groups sections (`Skills`, `Experience`, `Projects`, `Education`) with a 600-character chunk size and 20% overlap (`120 characters`).
2. **Multi-Query Decomposition**: Generates distinct query perspectives from the JD (core requirements, programming languages, cloud/DevOps, architecture).
3. **Weighted Reciprocal Rank Fusion (RRF)**: Fuses lexical sparse search (`rank-bm25`) and dense semantic vectors (`all-MiniLM-L6-v2` in ChromaDB) with configurable weights (`50/50`).
4. **Authoritative Keyword Reconciliation**: Reconciled keyword scan results are injected directly into the LLM prompt to anchor evaluation to verified facts.

### 3. Self-Healing Schema Fallbacks
LLMs occasionally wrap JSON in markdown blocks, add trailing commas, or echo raw schema properties. The pipeline implements a 4-tier fallback:
1. **PydanticOutputParser** for strict schema verification.
2. **Direct JSON loading & unwrapping** of metadata keys (`output`, `result`, `properties`).
3. **Balanced regex boundary extraction** to salvage valid JSON chunks from truncated streams.
4. **Safe model construction** (`parse_failed=True`) with friendly user-facing SSE notifications, preventing pipeline crashes.

---

## ✨ Core Application Features

### 1. 📊 Calibrated Match Score Gauge (0–10)
- Animated circular SVG gauge color-coded to candidate alignment:
  - 🟢 **8–10**: Strong Alignment (Green `#00FFA3`)
  - 🟡 **5–7**: Moderate Match (Amber `#FFB800`)
  - 🔴 **0–4**: Critical Gaps Present (Rose `#FF4060`)
- Displays execution duration and in-memory ChromaDB pipeline timings.

### 2. 🔍 Identified Skill Gaps Tab
- Deep cross-referencing between the JD and candidate resume.
- Cites the exact benchmark requirement quoted from the Job Description.
- Color-coded severity indicators based on overall alignment.
- **Success State**: If the resume satisfies all technical requirements, renders:
  > *🎉 Perfect Match! No critical technical gaps were identified for this role.*

### 3. ⚡ Actionable Resume Improvements Tab
- Concrete, copy-paste ready bullet rewrites using high-impact action verbs.
- Structured into:
  - **📌 Target Area**: Specific resume section to update (e.g., `Experience`, `Projects`).
  - **⚡ Action Required**: Clear directive on what project or impact metrics to highlight.
  - **🎯 JD Alignment**: Direct explanation of which JD requirement is satisfied.
- **Success State**:
  > *🌟 Outstanding Alignment! Your resume already explicitly demonstrates the key requirements for this position.*

### 4. 🎯 3-Tier Interview Preparation Roadmap
- Tailored study guides addressing each identified gap:
  - 📖 **Conceptual Study**: Official documentation and architecture guides to review.
  - 🛠️ **Hands-On Project**: Practical reference demo or mini-project to build.
  - 💬 **Interview Question**: Anticipated architectural questions and sample talking points.
- **Success State**:
  > *🎯 Interview Ready! Candidate meets all core technical requirements. Focus on standard system design and behavioral alignment.*

### 5. 🔑 ATS Keyword Heatmap
- Automatic extraction of 20+ technologies, frameworks, and certifications from the JD.
- Interactive chip grid:
  - ✅ **Found in Resume** (Green chips with checkmarks)
  - ❌ **Missing from Resume** (Red chips with crossmarks)
- Displays overall ATS keyword match percentage.

### 6. 📈 Analysis History & Score Trends
- Every analysis is automatically saved to an embedded SQLite database (`data/history.db`).
- Interactive score trend line chart tracking candidate improvement over multiple iterations.
- Browse past reports with full JD snippets, file metadata, and one-click deletion.

### 7. 📄 One-Click PDF Report Export
- Download a branded, formatted PDF executive summary generated via `fpdf2`.
- Includes match score gauge, ATS keyword summary table, gap analysis breakdown, improvement directives, and interview roadmaps.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | [Streamlit](https://streamlit.io/) `≥1.33` | Glassmorphic dark UI, sidebar navigation, real-time SSE progress streaming |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | High-performance asynchronous REST API, rate limiting, and SSE events |
| **Agentic Graph** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful pipeline orchestration with resilient progress callbacks |
| **Local LLM Engine** | [Ollama](https://ollama.com/) | 100% local, offline inference (`qwen2.5:3b`, `qwen3.5:9b`, `llama3.1`) |
| **Cloud LLM Engine** | [Groq](https://groq.com/) | Ultra-low latency cloud inference (`llama-3.1-8b-instant`) |
| **Dense Embeddings** | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Local dense semantic embedding generation (CPU friendly) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | In-memory vector database with persistent LRU caching |
| **Sparse Search** | [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Exact lexical keyword ranking fused with ChromaDB via RRF |
| **PDF Extraction** | [PyPDF](https://pypdf.readthedocs.io/) | Layout-aware multi-column PDF text extraction |
| **Report Generation** | [fpdf2](https://py-pdf.github.io/fpdf2/) | Professional multi-page PDF analysis report generation |
| **Validation** | [Pydantic](https://docs.pydantic.dev/) v2 + `pydantic-settings` | Strict typed data contracts and environment configuration |
| **Database** | SQLite3 | Embedded persistence for analysis history, keyword stats, and feedback |
| **Observability** | [LangSmith](https://smith.langchain.com/) | Optional production tracing via `LANGCHAIN_TRACING_V2` |

---

## ⚙ System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                              USER BROWSER                              │
│  Streamlit Frontend (http://localhost:8501)                            │
│  ├── 🔬 New Resume Analysis (PDF Upload, JD Input, Model Switcher)     │
│  └── 📊 History & Analytics (Score Trend Line Chart, Past Reports)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Multipart/Form-Data & SSE Stream
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FastAPI Server (http://127.0.0.1:8000)             │
│                                                                        │
│  API Endpoints:                                                        │
│  ├── POST /analyze/stream        (Real-time SSE analysis stream)       │
│  ├── POST /analyze               (Synchronous execution)               │
│  ├── GET  /history               (List past analysis summaries)        │
│  ├── GET  /history/trend         (Historical scores for trend chart)   │
│  ├── POST /export/pdf            (Download formatted PDF report)       │
│  └── GET  /health                (Service health & model discovery)    │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     LangGraph StateGraph                         │  │
│  │                                                                  │  │
│  │  [Node 1: extract_and_retrieve]                                  │  │
│  │  ├── PDF layout-aware parsing                                    │  │
│  │  ├── Word Count Evaluation:                                      │  │
│  │  │   ├── < 1,000 words: Single-Page RAG Bypass (Full text kept)  │  │
│  │  │   └── ≥ 1,000 words: Section Chunking + ChromaDB/BM25 Hybrid  │  │
│  │  │                                                               │  │
│  │  [Node 2: score_and_coach]                                       │  │
│  │  ├── Regex & NER Symbol-Aware ATS Keyword Extraction             │  │
│  │  ├── Recruiter Persona + Mathematical Scoring Rubric             │  │
│  │  └── Single consolidated LLM call (Gaps + Score + Coach)         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌────────────────────┐   ┌────────────────────┐   ┌────────────────┐  │
│  │ ChromaDB (Memory)  │   │ SQLite (History)   │   │ fpdf2 (PDF)    │  │
│  └────────────────────┘   └────────────────────┘   └────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ├─── [LLM_PROVIDER=ollama] ──► Local Ollama (port 11434)
                                    │                              (qwen2.5:3b / qwen3.5:9b)
                                    └─── [LLM_PROVIDER=groq]   ──► Groq Cloud API (HTTPS)
                                                                   (llama-3.1-8b-instant)
```

---

## 📦 Installation & Local Setup

### Prerequisites
- **Python**: `3.10`, `3.11`, `3.12`, or `3.14`
- **Ollama**: [Download & Install Ollama](https://ollama.com/download)
- **RAM Recommendation**:
  - `8 GB RAM`: Use `qwen2.5:3b` (Default — extremely fast, ~3s per analysis)
  - `16 GB+ RAM`: Use `qwen3.5:9b` or `qwen2.5-coder:7b`

### 1. Clone the Repository
```bash
git clone https://github.com/msaquib21/resume-analyzer-app.git
cd resume-analyzer-app
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the Default Ollama Model
Ensure Ollama is running, then pull the recommended default model:
```bash
ollama pull qwen2.5:3b
```
*(Optional: Pull additional models like `ollama pull qwen3.5:9b` or `ollama pull llama3.1:latest`)*

---

## 🔀 Dual-LLM Provider Modes (Ollama vs. Groq)

The application supports zero-code switching between Local Privacy and Cloud Hosting:

### Mode A: 100% Local Privacy (Default)
No API keys or internet connection required. Create or edit `.env`:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

### Mode B: Free Groq Cloud Deployment (Vercel / Render / Railway)
To deploy publicly without needing an expensive GPU instance:
1. Obtain a free API key from [console.groq.com](https://console.groq.com/).
2. Set your environment variables in `.env`:
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

---

## 🚀 Running the Application

Open two terminal windows:

### Terminal 1 — Start the FastAPI Backend
```bash
python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```
*Backend runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000) (Interactive Swagger docs at `/docs`)*

### Terminal 2 — Start the Streamlit Frontend
```bash
python -m streamlit run frontend/app.py --server.port 8501
```
*Frontend opens automatically at: [http://localhost:8501](http://localhost:8501)*

---

## 🐳 Docker Deployment

Run the complete multi-service stack with a single command:

```bash
docker-compose up --build
```

This spins up:
- **Ollama Service** (`port 11434`) with container-persisted model volumes.
- **Resume App Service** (`port 8000` for API and `port 8501` for UI).

Pull your model into the Docker container once:
```bash
docker exec -it resume-analyzer-app-ollama-1 ollama pull qwen2.5:3b
```
Then navigate to **http://localhost:8501**.

---

## ⚙ Comprehensive Configuration Reference

All settings can be configured via environment variables or a root `.env` file:

| Environment Variable | Default Value | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Active provider: `ollama` (local) or `groq` (cloud) |
| `GROQ_API_KEY` | `None` | API key for Groq Cloud API |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq cloud model identifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama daemon endpoint URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Default Ollama model tag |
| `AVAILABLE_MODELS` | `qwen2.5:3b,qwen3.5:9b,qwen2.5-coder:7b,...` | Comma-separated list of models for the UI dropdown |
| `OLLAMA_TEMPERATURE` | `0.0` | Sampling temperature (pinned to 0.0 for deterministic grounding) |
| `OLLAMA_NUM_CTX` | `4096` | Context window size for Ollama inference |
| `OLLAMA_NUM_PREDICT` | `1200` | Max tokens generated per response |
| `EMBEDDINGS_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face embedding model |
| `SINGLE_PAGE_MAX_WORDS` | `1000` | Word threshold below which RAG is bypassed to preserve full context |
| `CHUNK_SIZE` | `600` | Character chunk size for multi-page resumes |
| `CHUNK_OVERLAP` | `120` | 20% chunk overlap preserved across splits |
| `RETRIEVAL_K` | `10` | Chunks retrieved per query angle |
| `RETRIEVAL_QUERIES` | `4` | Number of distinct RAG decomposition queries |
| `RETRIEVAL_TOP_N` | `12` | Chunks retained after Reciprocal Rank Fusion |
| `BM25_WEIGHT` | `0.5` | Weight for BM25 lexical search in hybrid fusion |
| `MAX_CONCURRENT_ANALYSES` | `3` | Asynchronous rate-limiting semaphore limit |
| `GRAPH_TIMEOUT_SECONDS` | `300` | Timeout threshold for pipeline completion |
| `HISTORY_DB_PATH` | `data/history.db` | Local SQLite database file location |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith production observability |
| `LANGCHAIN_API_KEY` | `None` | LangSmith API key |

---

## 🧪 Testing & Quality Assurance

The codebase includes an automated unit and integration test suite covering PDF parsing, normalization, hybrid RAG weights, Pydantic schemas, and provider routing:

```bash
# Run the complete test suite
pytest tests/ -v

# Run linting and code formatting checks
ruff check backend/ tests/
```

### Verified Test Suite (21 Tests Passing)

| Test Module | Tests | Functionality Covered |
|---|---|---|
| `tests/test_parser.py` | 18 tests | Layout-aware multi-column parsing, dense skill preservation, BM25/ChromaDB RRF fusion, single-page RAG bypass, C++/C# technical symbol boundaries, flexible gap count, Dual-LLM provider routing, and empty string schema mitigation. |
| `tests/test_history.py` | 3 tests | SQLite CRUD operations: save analysis, retrieve records, cascade deletion, and chronological score trend data. |

---

## 📡 API Reference

Interactive OpenAPI documentation is available at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns backend liveness, Ollama reachability, active provider, and available model list. |
| `POST` | `/analyze/stream` | **Recommended**. Multipart/form-data upload streaming Server-Sent Events (progress + final JSON). |
| `POST` | `/analyze` | Synchronous analysis execution returning typed `AnalysisResponse`. |
| `GET` | `/history` | Returns all past analysis records stored in SQLite. |
| `GET` | `/history/{id}` | Retrieves a single analysis record with full gap, improvement, and keyword metadata. |
| `DELETE` | `/history/{id}` | Deletes a specified analysis record from SQLite. |
| `GET` | `/history/trend` | Returns timestamp and score pairs for historical trend visualization. |
| `POST` | `/feedback` | Saves user thumbs-up/thumbs-down feedback on specific analysis cards. |
| `POST` | `/export/pdf` | Generates and streams a downloadable branded PDF analysis report. |

---

## 📁 Project Directory Layout

```
resume-analyzer-app/
├── .github/workflows/
│   └── ci.yml                      # GitHub Actions automated lint & test pipeline
├── backend/
│   ├── __init__.py                 # Package initialization and graph exports
│   ├── agent.py                    # LangGraph 2-node pipeline, prompts, rubric, self-healing parser
│   ├── config.py                   # Pydantic Settings management with environment parsing
│   ├── history.py                  # SQLite storage engine for analyses, feedback, and trends
│   ├── models.py                   # Pydantic v2 data models, serializers, and list validators
│   ├── pdf_export.py               # fpdf2 branded PDF report layout generator
│   ├── rag.py                      # Multi-column layout PDF parser, hybrid search, RRF fusion
│   └── server.py                   # FastAPI application, SSE streaming endpoints, rate limiting
├── frontend/
│   └── app.py                      # Streamlit dashboard, custom CSS design system, SSE client
├── tests/
│   ├── __init__.py
│   ├── test_history.py             # SQLite persistence unit tests
│   └── test_parser.py              # Parsing, RAG, and schema normalization tests
├── data/                           # Auto-created directory for SQLite database storage
├── Dockerfile                      # Production multi-stage Docker build configuration
├── docker-compose.yml              # Multi-container orchestration (App + Ollama)
├── requirements.txt                # Pinned production Python dependencies
└── README.md                       # Comprehensive system documentation
```

---

## 📄 License & Acknowledgements

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

- **Author**: [Mohammad Saquib](https://github.com/msaquib21)
- Built with [LangGraph](https://github.com/langchain-ai/langgraph), [FastAPI](https://fastapi.tiangolo.com/), [Ollama](https://ollama.com/), and [Streamlit](https://streamlit.io/).

⭐ **Star this repository** if you found this project helpful for your technical job preparation!
