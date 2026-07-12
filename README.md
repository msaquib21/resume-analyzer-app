<div align="center">

# 🤖 Agentic Resume Analyzer

### AI-powered resume-to-job-description matching · 100% local · No API keys required

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Pipeline-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **Upload your resume PDF + paste a Job Description → Get an instant AI-powered match score, identified skill gaps, actionable resume improvements, and a personalised interview preparation roadmap.**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo Architecture](#-live-demo-architecture)
- [Tech Stack](#-tech-stack)
- [Core Features](#-core-features)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

**Agentic Resume Analyzer** is a full-stack AI application that uses a **multi-step LangGraph agentic pipeline** to intelligently compare a candidate's resume against a target job description. Unlike simple keyword matchers, it uses **local LLM inference** (via [Ollama](https://ollama.com/)) combined with **Retrieval-Augmented Generation (RAG)** to provide deep, contextual analysis — all running entirely on your own machine with **zero cloud dependencies or API costs**.

The system produces three key deliverables displayed across a polished Streamlit UI:
1. **Match Score (0–10)** — Calibrated against the JD's actual requirements
2. **Identified Skill Gaps** — Specific skills missing from the resume with JD evidence
3. **Actionable Resume Improvements** — Precise directives telling you *what to add/change* in your resume to better target this specific role
4. **Interview Preparation Roadmap** — Study resources, hands-on projects, and mock interview questions

---

## 🏗 Live Demo Architecture

```
┌─────────────────────────────────────────────┐
│                   USER                       │
│  Uploads Resume PDF + Pastes Job Description │
└──────────────────┬──────────────────────────┘
                   │ HTTP (SSE stream)
                   ▼
┌─────────────────────────────────────────────┐
│         Streamlit Frontend (port 8501)       │
│  Real-time progress updates via SSE          │
│  Animated score ring, tabbed result cards    │
└──────────────────┬──────────────────────────┘
                   │ POST /analyze/stream
                   ▼
┌─────────────────────────────────────────────┐
│         FastAPI Backend (port 8000)          │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │       LangGraph StateGraph          │    │
│  │                                     │    │
│  │  Node 1: PDF Extraction + RAG       │    │
│  │    └── PyPDF → Chunking             │    │
│  │    └── all-MiniLM-L6-v2 embeddings  │    │
│  │    └── ChromaDB vector store        │    │
│  │    └── Multi-query RRF retrieval    │    │
│  │                                     │    │
│  │  Node 2: Consolidated LLM Analysis  │    │
│  │    └── Ollama (qwen2.5:3b)          │    │
│  │    └── Score + Gaps + Improvements  │    │
│  │    └── Preparation roadmap          │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) `≥1.33` | Interactive UI with real-time SSE streaming |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | High-performance async REST API with SSE support |
| **LLM Runtime** | [Ollama](https://ollama.com/) + `qwen2.5:3b` | 100% local LLM inference — no API keys needed |
| **Agentic Pipeline** | [LangGraph](https://langchain-ai.github.io/langgraph/) `≥0.1` | Stateful multi-node graph-based agent orchestration |
| **LLM Integration** | [LangChain](https://python.langchain.com/) / [langchain-community](https://github.com/langchain-ai/langchain) | Ollama LLM wrapper, document loaders, text splitters |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` | Local semantic embedding (no GPU required) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | In-memory vector store with persistent caching |
| **RAG Strategy** | Multi-query + Reciprocal Rank Fusion (RRF) | High-recall retrieval across 3 distinct query angles |
| **PDF Parsing** | [PyPDF](https://pypdf.readthedocs.io/) | Text extraction from uploaded resume PDFs |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) v2 + [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | Strict I/O schema validation and environment config |
| **Streaming** | [sse-starlette](https://github.com/sysid/sse-starlette) + [httpx](https://www.python-httpx.org/) | Server-Sent Events for real-time progress updates |
| **Language** | Python 3.10+ | Core application language |

---

## ✨ Core Features

### 🔍 Skill Gap Analysis
- Identifies **3 critical gaps** between the resume and JD with structured evidence
- Each gap includes: **what's missing from the resume**, **what the JD explicitly demands**, and **the real-world consequence** of the gap
- Color-coded severity levels (High / Medium / Low) based on overall match score

### ⚡ Actionable Resume Improvements
- **Not just observations — direct action items**: *"Add a bullet point to your Experience section demonstrating X..."*
- Each improvement card includes:
  - **Target Area** — Which resume section to update (Experience, Skills, Summary, etc.)
  - **Action Required** — Exact imperative instruction for what to add or rephrase
  - **JD Alignment** — Why this specific change closes the gap for this role

### 🎯 Interview Preparation Roadmap
- Pipe-separated structured roadmap per identified gap:
  - **Study** — Specific books, docs, or courses
  - **Practice** — Concrete mini-projects to build
  - **Interview Angle** — Exact question a recruiter will ask

### 🚀 Performance Engineering
- **Multi-query RAG with RRF**: 3 distinct query angles for high-recall retrieval, fused via Reciprocal Rank Fusion
- **Single consolidated LLM call**: Gap analysis + scoring + coaching merged into one inference step (eliminates redundant ~12s round-trip)
- **Vector store caching**: ChromaDB instance cached per resume hash — no re-embedding on repeated runs
- **SSE streaming**: Real-time progress updates pushed to the UI so users always see what the pipeline is doing
- **Self-healing JSON parser**: 4-stage fallback chain that handles malformed LLM output gracefully — zero crashes

### 🎨 Modern Dark UI
- Custom dark theme (`#060914` background) with Electric Mint (`#00FFA3`), Hot Rose (`#FF4060`), and Electric Violet (`#B44DFF`) accents
- Animated SVG score ring with color-coded ring (green / amber / red) based on match level
- Compact card layout for gaps, improvements, and preparation — optimised for scanning

---

## ⚙ System Architecture

### LangGraph Pipeline Nodes

```python
# Node 1: PDF Extraction + Multi-Query RAG Retrieval
node_extract_and_retrieve(state)
  ├── PyPDF text extraction
  ├── RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
  ├── HuggingFace all-MiniLM-L6-v2 embeddings
  ├── ChromaDB vector store (LRU cached by resume hash)
  └── Multi-query retrieval (3 queries × top-6) → RRF fusion

# Node 2: Consolidated LLM Analysis
node_score_coach(state)
  ├── Builds few-shot prompt with concrete string examples
  ├── qwen2.5:3b via Ollama (temperature=0, num_ctx=2048)
  ├── Outputs: score, gaps[], improvements[], preparation[]
  └── 4-stage self-healing JSON parser (Pydantic → json.loads → regex → defaults)
```

### Self-Healing Parser Chain
1. **Pydantic `OutputParser`** — Standard parse attempt
2. **`json.loads` + pre-normalization** — Direct construction with list item coercion
3. **Regex extraction** — Pulls `{...}` from any wrapper text
4. **Safe defaults** — Returns zero-state to prevent pipeline crashes

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

Ensure Ollama is running, then pull the model:

```bash
ollama pull qwen2.5:3b
```

> **Note:** The first analysis run also downloads the `all-MiniLM-L6-v2` embedding model (~90MB) from Hugging Face automatically. Subsequent runs use the cached model.

### 5. (Optional) Create a `.env` File

You can override any default settings via environment variables:

```env
# .env — optional overrides
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TEMPERATURE=0.0
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_K=6
RETRIEVAL_QUERIES=3
```

---

## 🚀 Running the Application

Both servers must be running simultaneously. Open **two separate terminal windows** in the project root:

### Terminal 1 — Start the Backend API

```bash
uvicorn backend.server:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process using WatchFiles
INFO:     LangGraph pipeline compiled: 3 nodes
INFO:     Application startup complete.
```

### Terminal 2 — Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Expected output:
```
  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser.

### How to Use

1. **Upload Resume** — Drag & drop or browse for your resume PDF in the left panel
2. **Paste Job Description** — Paste the full JD text into the text area
3. **Select Model** — Keep the default `qwen2.5:3b` or change in the sidebar
4. **Click Analyze** — Watch real-time progress as the pipeline runs
5. **Review Results** — Navigate the three tabs: **Identified Gaps**, **Improvements**, **Preparation**

---

## 🔧 Configuration

All settings are managed via `backend/config.py` using Pydantic Settings. Every value can be overridden via environment variable or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen2.5:3b` | LLM model tag |
| `OLLAMA_TEMPERATURE` | `0.0` | Generation temperature (0 = deterministic) |
| `EMBEDDINGS_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | PDF text chunk size (tokens) |
| `CHUNK_OVERLAP` | `50` | Chunk overlap (tokens) |
| `RETRIEVAL_K` | `6` | Chunks retrieved per query |
| `RETRIEVAL_QUERIES` | `3` | Number of multi-query RAG queries |
| `GRAPH_TIMEOUT_SECONDS` | `300` | Max LangGraph pipeline timeout |

---

## 📁 Project Structure

```
resume-analyzer-app/
│
├── backend/                    # FastAPI + LangGraph backend
│   ├── __init__.py
│   ├── agent.py                # LangGraph pipeline: nodes, parser, prompt engineering
│   ├── config.py               # Pydantic Settings — all env-configurable settings
│   ├── models.py               # Pydantic I/O models (ResumeAnalysisState, etc.)
│   ├── rag.py                  # PDF parsing, chunking, ChromaDB, multi-query RAG
│   └── server.py               # FastAPI app, SSE streaming endpoint, CORS
│
├── frontend/
│   └── app.py                  # Streamlit UI: CSS design system, SSE consumer, card rendering
│
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [msaquib21](https://github.com/msaquib21) · Powered by Ollama · LangGraph · Streamlit

⭐ **Star this repo** if you found it useful for your job search!

</div>
