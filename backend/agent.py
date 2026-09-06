"""
LangGraph multi-agent pipeline for resume analysis.

Pipeline (3 nodes)
------------------
1. ``node_extract_and_retrieve``
   Parses the resume PDF, chunks it, builds a vector store, and runs
   multi-query RAG retrieval against the job description.

2. ``node_gap_analysis``
   Uses a local Ollama LLM to identify technical skill gaps between the
   retrieved resume context and the job description. Output is validated
   against ``GapAnalysisOutput`` via Pydantic.

3. ``node_score_coach``
   Produces the final scored analysis: a 0-10 match score, refined gap
   list, actionable improvements, and interview preparation steps.
   Output is validated against ``ScoreCoachOutput`` via Pydantic.

All nodes emit progress events via an optional callback so the server can
stream real-time status updates to the client.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Callable, List, Optional, TypedDict

from langchain_community.llms import Ollama
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph

from .config import settings
from .models import GapAnalysisOutput, ScoreCoachOutput
from .rag import load_and_chunk_pdf, retrieve_relevant_chunks

logger = logging.getLogger(__name__)

# ── Progress callback type ────────────────────────────────────────────────────
# Called by each node so the server can emit SSE events.  Signature:
#   callback(stage: str, message: str) -> None
ProgressCallback = Optional[Callable[[str, str], None]]


# ── LangGraph state schema ────────────────────────────────────────────────────


class ResumeAnalysisState(TypedDict):
    """Immutable-like typed state that flows through the LangGraph pipeline."""

    # ── Inputs (set by the server before graph.invoke) ──────────────────
    job_description: str
    resume_pdf_bytes: bytes
    embeddings_model_name: str
    ollama_model_name: str
    progress_callback: ProgressCallback  # optional; ignored by LangGraph routing

    # ── Node 1 outputs ───────────────────────────────────────────────────
    resume_chunks: List[str]
    retrieved_chunks: List[str]

    # ── Node 2 outputs ───────────────────────────────────────────────────
    gap_analysis_text: str
    gaps: List[str]

    # ── Node 3 / final outputs ───────────────────────────────────────────
    improvements: List[str]
    preparation: List[str]
    score: int
    keywords: list[dict]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a meticulous Generative AI engineer and technical recruiter hybrid.

Operate exclusively on the provided resume context and job description.
Your output must be a single, valid JSON object matching the schema requested.

Scoring rubric (integer 0–10):
  9-10  Highly aligned: most required tools/skills present with strong evidence.
  6-8   Mostly aligned: a few key tools or insufficient depth.
  3-5   Partial: multiple significant gaps or weak/unclear evidence.
  0-2   Largely misaligned or missing relevant technical evidence.

Gaps       → specific skills/tools/experience categories tied to the JD.
Improvements → concrete resume bullet rewrites or additions that close each gap.
Preparation  → interview study topics, each aligned to a specific gap.
"""


# ── LLM factory ──────────────────────────────────────────────────────────────


def _build_ollama(model: str, temperature: float = 0.0) -> Ollama:
    """Instantiate a local Ollama-backed LLM with JSON output mode and performance optimizations."""
    return Ollama(
        model=model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
        format="json",
        num_ctx=2048,
        num_predict=700,
        keep_alive="15m",
    )


# ── JSON fence stripper ───────────────────────────────────────────────────────


def _strip_json_fences(raw: str) -> str:
    """Remove markdown code fences that some models add around JSON output."""
    s = (raw or "").strip()
    if s.startswith("```json"):
        s = s[len("```json"):].lstrip()
    elif s.startswith("```"):
        s = s[len("```"):].lstrip()
    if s.endswith("```"):
        s = s[: -len("```")].rstrip()
    return s.strip()


def _clean_json_str(s: str) -> str:
    """Strip markdown, extract {...} or [...], and fix trailing commas."""
    s = _strip_json_fences(s)
    # Fix trailing commas inside JSON objects/arrays: `[ "foo", ]` -> `[ "foo" ]`
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def _normalize_list_item(item: Any) -> str:
    """Extract a clean plain string from whatever the LLM puts in a list slot.

    Handles three cases:
    1. Plain string  → return as-is (most common after fixing prompts).
    2. Simple dict   → pick known keys or format structured sub-keys into crisp pipe-separated paragraphs.
    3. Multi-key dict → concatenate all non-empty string values into one paragraph.
    """
    if isinstance(item, dict):
        # 1. Check for structured gaps dictionary
        skill_val = item.get("Skill Name") or item.get("skill_name") or item.get("Skill") or item.get("name")
        missing_val = item.get("Missing Evidence") or item.get("missing_evidence") or item.get("Missing")
        jd_val = item.get("JD Demands") or item.get("jd_demands") or item.get("JD") or item.get("demands")
        if skill_val and isinstance(skill_val, str) and (missing_val or jd_val):
            parts = []
            if isinstance(missing_val, str) and missing_val.strip(): parts.append(missing_val.strip().rstrip(".") + ".")
            if isinstance(jd_val, str) and jd_val.strip(): parts.append(f"JD requires: {jd_val.strip().rstrip('.')}.")
            body = " ".join(parts) if parts else "Candidate resume needs stronger alignment with job requirements."
            return f"{skill_val.strip()}: {body}"

        # 2. Check for structured improvements dictionary
        action_val = item.get("Action Required") or item.get("action_required") or item.get("Action") or item.get("action")
        area_val = item.get("Target Area") or item.get("target_area") or item.get("Area") or item.get("area") or item.get("Target")
        align_val = item.get("JD Alignment") or item.get("jd_alignment") or item.get("Alignment") or item.get("why")
        if action_val and isinstance(action_val, str):
            area_str = f"Target Area: {area_val.strip()} | " if isinstance(area_val, str) and area_val.strip() else ""
            align_str = f" | JD Alignment: {align_val.strip()}" if isinstance(align_val, str) and align_val.strip() else ""
            return f"{area_str}Action Required: {action_val.strip()}{align_str}"

        # 3. Check for structured preparation dictionary
        gap_val = item.get("Target Gap") or item.get("target_gap") or item.get("Gap") or item.get("gap")
        study_val = item.get("Study") or item.get("study") or item.get("resource")
        prac_val = item.get("Practice") or item.get("practice") or item.get("project")
        angle_val = item.get("Interview Angle") or item.get("interview_angle") or item.get("angle")
        if study_val or prac_val or angle_val:
            gap_str = f"Target Gap: {gap_val.strip()} | " if isinstance(gap_val, str) and gap_val.strip() else ""
            s_str = f"Study: {study_val.strip()} | " if isinstance(study_val, str) and study_val.strip() else ""
            p_str = f"Practice: {prac_val.strip()} | " if isinstance(prac_val, str) and prac_val.strip() else ""
            a_str = f"Interview Angle: {angle_val.strip()}" if isinstance(angle_val, str) and angle_val.strip() else ""
            return f"{gap_str}{s_str}{p_str}{a_str}".strip(" |")

        # Try well-known single-key patterns next
        single_key_priority = (
            "type", "gap", "improvement", "preparation",
            "text", "description", "value", "content",
        )
        for k in single_key_priority:
            if isinstance(item.get(k), str) and item[k].strip():
                return item[k].strip()

        # Multi-key dict: reconstruct as "Key: value. Key2: value2."
        str_values = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
        if str_values:
            if len(str_values) == 1:
                return str_values[0]
            head = str_values[0].rstrip(".")
            tail = " ".join(v.rstrip(".") + "." for v in str_values[1:])
            return f"{head}: {tail}"
        return str(item)

    if isinstance(item, str):
        s = item.strip()
        # Attempt to decode if it looks like a stringified dict
        if s.startswith("{"):
            try:
                decoded = json.loads(s)
                if isinstance(decoded, dict):
                    return _normalize_list_item(decoded)
            except Exception:
                pass
        return s

    return str(item) if item is not None else ""



def _clean_gap_text(text: str) -> str:
    s = text.strip()
    # Remove duplicate skill headers like 'Python: Python: ' or 'RAG: RAG: '
    parts = s.split(":", 2)
    if len(parts) >= 3 and parts[0].strip().lower() == parts[1].strip().lower():
        s = f"{parts[0].strip()}: {parts[2].strip()}"
    return s


def _clean_improvement_text(text: str) -> str:
    s = text.strip()
    # Strip prompt artifacts
    s = re.sub(r"^Add bullet:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*[—-]\s*addresses gap:.*$", "", s, flags=re.IGNORECASE)
    # Fix present tense 3B defaults to past tense professional resume language
    replacements = [
        (r"^Design and implement(ed)?\b", "Designed and implemented"),
        (r"^Participate(d)? in\b", "Spearheaded and executed"),
        (r"^Build and deploy(ed)?\b", "Built and deployed"),
        (r"^Develop(ed)?\b", "Developed"),
        (r"^Create(d)?\b", "Created"),
        (r"^Implement(ed)?\b", "Implemented"),
        (r"^Conduct(ed)?\b", "Conducted"),
        (r"^Optimize(d)?\b", "Optimized"),
        (r"^Architect(ed)?\b", "Architected"),
        (r"^Lead\b", "Led"),
    ]
    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    return s


def _normalize_model_lists(obj: Any) -> Any:
    """Ensure string list attributes on Pydantic models are stripped of dict wrappers and polished."""
    if hasattr(obj, "gaps") and isinstance(obj.gaps, list):
        obj.gaps = [_clean_gap_text(_normalize_list_item(x)) for x in obj.gaps if _normalize_list_item(x)]
    if hasattr(obj, "improvements") and isinstance(obj.improvements, list):
        obj.improvements = [_clean_improvement_text(_normalize_list_item(x)) for x in obj.improvements if _normalize_list_item(x)]
    if hasattr(obj, "preparation") and isinstance(obj.preparation, list):
        obj.preparation = [_normalize_list_item(x) for x in obj.preparation if _normalize_list_item(x)]
    return obj


def _parse_llm_output(raw: str, parser: PydanticOutputParser, model_cls):
    """Parse *raw* LLM text → Pydantic model, with a robust self-healing fallback chain."""
    cleaned = _clean_json_str(raw)
    # Attempt 1: PydanticOutputParser
    try:
        res = parser.parse(cleaned)
        return _normalize_model_lists(res)
    except Exception:
        pass

    # Attempt 2: direct json.loads + model construction
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            for lk in ("gaps", "improvements", "preparation"):
                if isinstance(data.get(lk), list):
                    data[lk] = [_normalize_list_item(x) for x in data[lk] if _normalize_list_item(x)]
            # Ensure score is an int, not a placeholder string
            if "score" in data and not isinstance(data["score"], (int, float)):
                try:
                    data["score"] = int(re.search(r"\d+", str(data["score"])).group())
                except Exception:
                    data["score"] = 0
        res = model_cls(**data)
        return _normalize_model_lists(res)
    except Exception:
        pass

    # Attempt 2.5: regex extraction of {...} or [...] inside any wrapper text
    data = None
    try:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            extracted = re.sub(r",\s*([\]}])", r"\1", match.group(1))
            data = json.loads(extracted)
            if isinstance(data, dict):
                for lk in ("gaps", "improvements", "preparation"):
                    if isinstance(data.get(lk), list):
                        data[lk] = [_normalize_list_item(x) for x in data[lk] if _normalize_list_item(x)]
                res = model_cls(**data)
                return _normalize_model_lists(res)
            elif isinstance(data, list) and model_cls == GapAnalysisOutput:
                res = GapAnalysisOutput(gaps=[_normalize_list_item(x) for x in data if _normalize_list_item(x)])
                return _normalize_model_lists(res)
    except Exception:
        pass

    # Attempt 3: Intelligent Schema Normalization & Self-Healing
    logger.warning("Standard parse failed for %s. Attempting self-healing normalization on raw text: %s", model_cls.__name__, raw[:600])
    try:
        if data is None and isinstance(raw, str):
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(re.sub(r",\s*([\]}])", r"\1", match.group(1)))
                except Exception:
                    pass

        if isinstance(data, dict):
            # Unwrap if model nested properties under 'properties', 'output', 'result', etc.
            for wrapper_key in ("properties", "output", "result", "response", "ScoreCoachOutput", "GapAnalysisOutput"):
                if isinstance(data.get(wrapper_key), dict):
                    data = data[wrapper_key]
                    break

            if model_cls == GapAnalysisOutput:
                gaps = data.get("gaps") or data.get("technical_gaps") or data.get("skill_gaps") or data.get("gaps_identified")
                if not gaps:
                    for v in data.values():
                        if isinstance(v, list):
                            gaps = v
                            break
                if isinstance(gaps, list):
                    res = GapAnalysisOutput(gaps=[_normalize_list_item(g) for g in gaps if _normalize_list_item(g)])
                    return _normalize_model_lists(res)
                elif isinstance(gaps, str):
                    res = GapAnalysisOutput(gaps=[_normalize_list_item(g) for g in re.split(r"\n|;", gaps) if _normalize_list_item(g)])
                    return _normalize_model_lists(res)

            elif model_cls == ScoreCoachOutput:
                # Extract score safely
                raw_score = data.get("score")
                if raw_score is None:
                    for sk in ("match_score", "final_score", "rating", "matchScore"):
                        if data.get(sk) is not None:
                            raw_score = data[sk]
                            break
                score_val = 0
                if isinstance(raw_score, (int, float)):
                    score_val = int(raw_score)
                elif isinstance(raw_score, str):
                    sm = re.search(r"\d+", raw_score)
                    if sm:
                        score_val = int(sm.group(0))

                # Extract list fields safely
                def _get_list(keys: list[str]) -> list[str]:
                    for k in keys:
                        val = data.get(k)
                        if isinstance(val, list):
                            return [_normalize_list_item(item) for item in val if _normalize_list_item(item)]
                        elif isinstance(val, str) and val.strip():
                            return [_normalize_list_item(s) for s in re.split(r"\n|;", val) if _normalize_list_item(s)]
                    return []

                gaps = _get_list(["gaps", "technical_gaps", "skill_gaps", "identified_gaps"])
                improvements = _get_list(["improvements", "recommendations", "resume_improvements", "suggestions", "bullet_points"])
                preparation = _get_list(["preparation", "interview_prep", "prep_plan", "study_topics", "preparation_steps"])

                res = ScoreCoachOutput(
                    score=max(0, min(10, score_val)),
                    gaps=gaps,
                    improvements=improvements,
                    preparation=preparation
                )
                return _normalize_model_lists(res)
    except Exception as e:
        logger.error("Self-healing normalization failed: %s", e)

    # Attempt 4: Return cleanly initialized defaults so the pipeline never crashes
    logger.error("All parse & self-healing attempts failed for %s. Returning safe defaults.", model_cls.__name__)
    try:
        return model_cls()
    except Exception:
        return model_cls.model_construct(score=0, gaps=[], improvements=[], preparation=[])


def extract_jd_keywords(jd_text: str, resume_text: str) -> list[dict]:
    """Extract skills/tools/technologies from JD and check if present in resume."""
    keywords_found = []
    seen = set()
    
    categories = {
        'skill': [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Go', 'Rust', 'Ruby', 'Kotlin', 'Swift', 'Scala', 'R', 'MATLAB', 'SQL',
            'TensorFlow', 'PyTorch', 'scikit-learn', 'Pandas', 'NumPy', 'LangChain', 'LangGraph', 'Ollama', 'OpenAI', 'Hugging Face', 'FAISS', 'Pinecone', 'ChromaDB', 'Weaviate', 'RAG', 'LLM', 'NLP'
        ],
        'tool': [
            'React', 'Angular', 'Vue', 'Django', 'Flask', 'FastAPI', 'Spring', 'Express', 'Node.js', 'Next.js',
            'AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes', 'Terraform',
            'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'Elasticsearch', 'DynamoDB',
            'Git', 'Jenkins', 'GitHub Actions', 'CI/CD', 'Jira'
        ],
        'certification': [
            'AWS Certified', 'PMP', 'Scrum'
        ]
    }
    
    jd_lower = jd_text.lower()
    resume_lower = resume_text.lower()
    
    for category, terms in categories.items():
        for term in terms:
            term_lower = term.lower()
            pattern = r'(?<!\w)' + re.escape(term_lower) + r'(?!\w)'
            if re.search(pattern, jd_lower):
                if term_lower not in seen:
                    found = bool(re.search(pattern, resume_lower))
                    keywords_found.append({
                        'keyword': term,
                        'found_in_resume': found,
                        'category': category
                    })
                    seen.add(term_lower)
    
    ner_patterns = [
        r'experience with\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)',
        r'proficiency in\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)',
        r'knowledge of\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)',
        r'familiarity with\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)',
        r'expertise in\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)'
    ]
    
    for pattern in ner_patterns:
        for match in re.finditer(pattern, jd_text):
            phrase = match.group(1).strip()
            if phrase:
                term_lower = phrase.lower()
                if term_lower not in seen and len(phrase) > 1:
                    pattern_search = r'(?<!\w)' + re.escape(term_lower) + r'(?!\w)'
                    found = bool(re.search(pattern_search, resume_lower))
                    keywords_found.append({
                        'keyword': phrase,
                        'found_in_resume': found,
                        'category': 'skill'
                    })
                    seen.add(term_lower)
                    
    return keywords_found[:20]


# ── Node 1: PDF extraction + multi-query RAG retrieval ───────────────────────


def node_extract_and_retrieve(state: ResumeAnalysisState) -> dict:
    """Parse resume PDF and retrieve relevant chunks via multi-query RAG."""
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("extracting", "Parsing PDF and building embeddings…")

    t0 = time.perf_counter()
    resume_chunks = load_and_chunk_pdf(state["resume_pdf_bytes"])
    logger.info("Node 1 — PDF chunked: %d chunks (%.2fs)", len(resume_chunks), time.perf_counter() - t0)

    t1 = time.perf_counter()
    context = retrieve_relevant_chunks(
        job_description=state["job_description"],
        resume_chunks=resume_chunks,
        embeddings_model_name=state["embeddings_model_name"],
        k=settings.retrieval_k,
        resume_pdf_bytes=state["resume_pdf_bytes"],
    )
    logger.info(
        "Node 1 — RAG complete: %d chunks retrieved via %d queries (%.2fs)",
        len(context.retrieved_chunks),
        context.query_count,
        time.perf_counter() - t1,
    )

    if callback:
        callback("retrieved", f"Retrieved {len(context.retrieved_chunks)} relevant chunks via {context.query_count}-query RAG")

    return {
        "resume_chunks": resume_chunks,
        "retrieved_chunks": context.retrieved_chunks,
    }


# ── Node 2: Gap analysis ──────────────────────────────────────────────────────


def node_gap_analysis(state: ResumeAnalysisState) -> dict:
    """Lightweight pre-pass: signals the pipeline to proceed to node_score_coach.

    The actual gap identification is performed inside node_score_coach in a
    single consolidated LLM call — this eliminates a redundant round-trip that
    was costing ~12 seconds per request.
    """
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("analyzing_gaps", "Scanning resume against JD…")
    logger.info("Node 2 — pre-pass complete (no LLM call, merged into Node 3)")
    return {
        "gap_analysis_text": "",
        "gaps": [],
    }


# ── Node 3: Full consolidated analysis ────────────────────────────────────────


def node_score_coach(state: ResumeAnalysisState) -> dict:
    """Single consolidated LLM call: gap identification + scoring + improvements + prep.

    Uses a compact inline-schema prompt (no get_format_instructions()) to reduce
    input token count by ~300 tokens, cutting generation time significantly.
    """
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("scoring", "Running deep analysis…")

    parser = PydanticOutputParser(pydantic_object=ScoreCoachOutput)
    llm = _build_ollama(state["ollama_model_name"])

    # Limit context to 5 chunks to keep input tokens bounded
    resume_context = "\n\n".join(state["retrieved_chunks"][:5])
    jd = state["job_description"]

    keywords = extract_jd_keywords(jd, resume_context)

    prompt = (
        "You are a senior technical recruiter and career coach.\n"
        "Analyse this resume against the job description. Return ONLY valid JSON — no markdown, no explanation.\n\n"
        "JOB DESCRIPTION:\n"
        f"{jd}\n\n"
        "RESUME (most relevant sections):\n"
        f"{resume_context}\n\n"
        'Output this exact JSON structure with string values (NOT nested objects):\n'
        '{\n'
        '  "score": 6,\n'
        '  "gaps": [\n'
        '    "Python Proficiency: The resume shows basic scripting but no production-grade Python projects with testing or CI/CD pipelines. The JD requires 2+ years of Python at a senior engineering level. Without this depth the candidate cannot own backend services independently.",\n'
        '    "Cloud Deployment: No evidence of deploying models to AWS, GCP, or Azure is present in the resume. The JD explicitly requires experience deploying ML workloads to a major cloud provider. This gap means the candidate cannot own the MLOps lifecycle end-to-end.",\n'
        '    "Vector Databases: The resume does not mention Pinecone, Weaviate, FAISS, or any vector store integration. The JD demands hands-on experience building retrieval pipelines with vector databases. Without this the candidate cannot build the RAG systems central to this role."\n'
        '  ],\n'
        '  "improvements": [\n'
        '    "Target Area: Experience Section (Python & Testing) | Action Required: Add an explicit bullet point to your most recent role demonstrating how you structured Python backend services with unit testing (pytest) and CI/CD automation. | JD Alignment: The JD explicitly demands senior-level Python engineering with rigorous testing practices rather than just scripting.",\n'
        '    "Target Area: Projects Section (Cloud MLOps) | Action Required: Create or highlight a project where you deployed a real-time ML model to AWS SageMaker or GCP, specifically mentioning containerization (Docker) and endpoint latency. | JD Alignment: Addresses the JD\'s strict requirement for hands-on experience deploying and monitoring models in production cloud environments.",\n'
        '    "Target Area: Skills & Summary (Vector Databases) | Action Required: Explicitly list vector database tools like Pinecone, ChromaDB, or FAISS in your Skills section, and add a brief technical summary line explaining your approach to semantic search pipelines. | JD Alignment: Bridges the gap where the resume omitted vector store integration, which is core to the RAG systems outlined in this job description."\n'
        '  ],\n'
        '  "preparation": [\n'
        '    "Target Gap: Vector Databases | Study: Pinecone documentation + LangChain retrieval guide | Practice: Build a semantic search engine over 10,000 Wikipedia articles using FAISS | Interview Angle: How would you choose between Pinecone, Weaviate, and FAISS for a production RAG system?",\n'
        '    "Target Gap: Cloud Deployment | Study: AWS SageMaker Developer Guide + MLOps on AWS course (Coursera) | Practice: Deploy a scikit-learn model as a SageMaker endpoint with auto-scaling | Interview Angle: Walk me through how you would set up a CI/CD pipeline for an ML model on AWS.",\n'
        '    "Target Gap: Python Proficiency | Study: Fluent Python by Luciano Ramalho (chapters 1-10) | Practice: Re-implement one of your existing projects with full pytest coverage and a GitHub Actions CI pipeline | Interview Angle: How do you structure a Python project for long-term maintainability?"\n'
        '  ]\n'
        '}\n\n'
        "Now produce the SAME JSON structure for the actual resume and JD above. Replace all example values with real analysis.\n"
        "RULES:\n"
        "- score: integer only. 0-2=largely misaligned, 3-5=partial, 6-8=mostly aligned, 9-10=strong match.\n"
        "- gaps: exactly 3 strings. Each must be a single plain string (NOT a nested object). Format: 'Skill Name: sentence about missing evidence. sentence about what JD demands. sentence about real-world consequence.'\n"
        "- improvements: exactly 3 strings. Each string MUST give clear, actionable advice instructing the candidate what exact modifications or additions they should make to their resume/portfolio (`Add...`, `Highlight...`, `Quantify...`, `Rephrase...`) to tailor it for this specific Job Description. Format: 'Target Area: NAME | Action Required: DIRECTIVE | JD Alignment: EXPLANATION'. Do NOT write past-tense resume bullets about what the candidate already did.\n"
        "- preparation: exactly 3 strings. Format: 'Target Gap: NAME | Study: RESOURCE | Practice: PROJECT | Interview Angle: QUESTION'.\n"
        "- Return ONLY valid JSON. No markdown fences. No nested objects inside lists."
    )

    t0 = time.perf_counter()
    raw = llm.invoke(prompt)
    raw_text = getattr(raw, "content", str(raw))
    logger.info("Node 3 — consolidated LLM call: %.2fs", time.perf_counter() - t0)

    result: ScoreCoachOutput = _parse_llm_output(raw_text, parser, ScoreCoachOutput)

    score = max(0, min(10, getattr(result, "score", 0) or 0))
    gaps = getattr(result, "gaps", []) or []
    improvements = getattr(result, "improvements", []) or []
    preparation = getattr(result, "preparation", []) or []

    logger.info(
        "Node 3 — Final: score=%d gaps=%d improvements=%d preparation=%d",
        score,
        len(gaps),
        len(improvements),
        len(preparation),
    )

    if callback:
        callback("complete", f"Analysis complete — score {score}/10")

    return {
        "score": score,
        "gaps": gaps,
        "improvements": improvements,
        "preparation": preparation,
        "keywords": keywords,
    }


# ── Graph compilation ─────────────────────────────────────────────────────────


def build_resume_analysis_graph(model: str = settings.ollama_model):
    """Compile and return the LangGraph analysis pipeline.

    Args:
        model: Ollama model tag.  Defaults to ``settings.ollama_model``.

    Returns:
        A compiled ``StateGraph`` ready for ``.invoke()``.
    """
    _ = model  # Retained for backwards-compatibility; nodes use state["ollama_model_name"]
    graph = StateGraph(ResumeAnalysisState)

    graph.add_node("extract_and_retrieve", node_extract_and_retrieve)
    graph.add_node("gap_analysis", node_gap_analysis)
    graph.add_node("score_and_coach", node_score_coach)

    graph.set_entry_point("extract_and_retrieve")
    graph.add_edge("extract_and_retrieve", "gap_analysis")
    graph.add_edge("gap_analysis", "score_and_coach")
    graph.add_edge("score_and_coach", END)

    logger.info("LangGraph pipeline compiled: 3 nodes")
    return graph.compile()
