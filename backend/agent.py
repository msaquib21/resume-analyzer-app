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
from .rag import (
    extract_layout_aware_text,
    is_single_page_resume,
    load_and_chunk_pdf,
    retrieve_relevant_chunks,
)

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

You are a strict text-matcher. Base gaps ONLY on the provided job description text. Do not hallucinate industry standards (e.g., AWS, Pinecone) if they are not explicitly written.
If the candidate's resume explicitly satisfies a job description requirement, do not flag it as a gap.
Never recommend a resume improvement or bullet point that is already visibly present in the candidate's uploaded resume text.

Operate exclusively on the provided resume context and job description. Forbid prior-knowledge bias.
Your output must be a single, valid JSON object matching the schema requested.

Scoring rubric (integer 0–10):
  9-10  Highly aligned: most required tools/skills present with strong evidence.
  6-8   Mostly aligned: a few key tools or insufficient depth.
  3-5   Partial: multiple significant gaps or weak/unclear evidence.
  0-2   Largely misaligned or missing relevant technical evidence.

Gaps         → specific skills/tools/experience categories tied strictly to the JD (empty if candidate meets all requirements).
Improvements → concrete resume bullet rewrites or additions that close each genuine gap.
Preparation  → interview study topics, each aligned to a specific genuine gap.
"""


# ── LLM factory (Dual Provider: Ollama vs. Groq) ──────────────────────────────


def _build_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Instantiate configured LLM provider: local Ollama or Groq Cloud API.

    - If settings.llm_provider == "groq": ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    - If settings.llm_provider == "ollama": Ollama(model=settings.ollama_model, temperature=0.0)
    """
    provider = getattr(settings, "llm_provider", "ollama").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        groq_model = getattr(settings, "groq_model", "llama-3.1-8b-instant")
        logger.info("Instantiating Groq LLM provider: model=%s", groq_model)
        return ChatGroq(
            model=groq_model,
            temperature=0.0,
            groq_api_key=settings.groq_api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    else:
        target_model = model or settings.ollama_model
        logger.info("Instantiating local Ollama LLM provider: model=%s", target_model)
        return Ollama(
            model=target_model,
            temperature=0.0,  # Hardcoded to 0.0 to eliminate creative drift
            base_url=settings.ollama_base_url,
            format="json",
            num_ctx=4096,     # Expanded context window for comprehensive RAG chunks + JD
            num_predict=800,
            keep_alive="15m",
        )


# Backwards compatibility alias
_build_ollama = _build_llm


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
        return ""

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
        if s in ('""', "''", "[]", "{}", "none", "n/a", "null"):
            return ""
        return s

    if item is None:
        return ""
    s = str(item).strip()
    return "" if s in ('""', "''", "[]", "{}", "none", "n/a", "null") else s


def _clean_gap_text(text: str) -> str:
    s = text.strip()
    # Remove duplicate skill headers like 'Python: Python: ' or 'RAG: RAG: '
    parts = s.split(":", 2)
    if len(parts) >= 3 and parts[0].strip().lower() == parts[1].strip().lower():
        s = f"{parts[0].strip()}: {parts[2].strip()}"
    return s.strip()


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
    return s.strip()


def _normalize_model_lists(obj: Any) -> Any:
    """Ensure string list attributes on Pydantic models are stripped of dict wrappers and polished."""
    def _is_valid(val: str) -> bool:
        return bool(val and val.strip() and val.strip() not in ('""', "''", "[]", "{}", "none", "n/a", "null"))

    if hasattr(obj, "gaps") and isinstance(obj.gaps, list):
        obj.gaps = [_clean_gap_text(cleaned) for x in obj.gaps for cleaned in [_normalize_list_item(x)] if _is_valid(cleaned)]
    if hasattr(obj, "improvements") and isinstance(obj.improvements, list):
        obj.improvements = [_clean_improvement_text(cleaned) for x in obj.improvements for cleaned in [_normalize_list_item(x)] if _is_valid(cleaned)]
    if hasattr(obj, "preparation") and isinstance(obj.preparation, list):
        obj.preparation = [cleaned for x in obj.preparation for cleaned in [_normalize_list_item(x)] if _is_valid(cleaned)]
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
            # Robust symbol-aware boundary matcher for C++, C#, .NET, Node.js, etc.
            pattern = r"(?i)(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])"
            if re.search(pattern, jd_text):
                term_lower = term.lower()
                if term_lower not in seen:
                    found = bool(re.search(pattern, resume_text))
                    keywords_found.append({
                        "keyword": term,
                        "found_in_resume": found,
                        "category": category,
                    })
                    seen.add(term_lower)

    ner_patterns = [
        r"experience with\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)",
        r"proficiency in\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)",
        r"knowledge of\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)",
        r"familiarity with\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)",
        r"expertise in\s+([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)",
    ]

    for pattern in ner_patterns:
        for match in re.finditer(pattern, jd_text):
            phrase = match.group(1).strip()
            if phrase:
                term_lower = phrase.lower()
                if term_lower not in seen and len(phrase) > 1:
                    pattern_search = r"(?i)(?<![A-Za-z])" + re.escape(phrase) + r"(?![A-Za-z])"
                    found = bool(re.search(pattern_search, resume_text))
                    keywords_found.append({
                        "keyword": phrase,
                        "found_in_resume": found,
                        "category": "skill",
                    })
                    seen.add(term_lower)

    return keywords_found[:20]


# ── Node 1: PDF extraction + multi-query RAG retrieval ───────────────────────


# ── Node 1: PDF extraction + multi-query RAG retrieval ───────────────────────


def node_extract_and_retrieve(state: ResumeAnalysisState) -> dict:
    """Parse resume PDF, evaluate length, and retrieve context (bypassing RAG for single-page documents)."""
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("extracting", "Parsing PDF layout and evaluating document length…")

    t0 = time.perf_counter()
    full_text = extract_layout_aware_text(state["resume_pdf_bytes"])
    word_count = len(full_text.split())

    # Module 2: RAG Bypass for Single-Page Resumes (< 1,000 words)
    if is_single_page_resume(full_text, max_words=1000):
        logger.info(
            "Node 1 — Single-page resume detected (%d words < 1000): Bypassing ChromaDB retrieval to preserve complete context.",
            word_count,
        )
        if callback:
            callback("retrieved", f"Single-page resume ({word_count} words): Full context preserved (RAG bypassed)")
        return {
            "resume_chunks": [full_text],
            "retrieved_chunks": [full_text],
        }

    # Multi-page resumes: section-aware chunking + hybrid retrieval
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
        "Node 1 — Hybrid RAG complete: %d chunks retrieved via %d queries (%.2fs)",
        len(context.retrieved_chunks),
        context.query_count,
        time.perf_counter() - t1,
    )

    if callback:
        callback("retrieved", f"Retrieved {len(context.retrieved_chunks)} relevant chunks via {context.query_count}-query Hybrid RAG")

    return {
        "resume_chunks": resume_chunks,
        "retrieved_chunks": context.retrieved_chunks,
    }


# ── Node 2: Gap analysis ──────────────────────────────────────────────────────

NODE_2_SYSTEM_PROMPT = """\
You are a strict, highly analytical technical recruiter. You must perform a rigorous cross-reference of the Job Description against the candidate's resume. If a core technology, framework, or responsibility listed in the Job Description is missing from the resume, you MUST document it as a gap. Do not be lenient. However, if the candidate explicitly possesses the exact skill, do not document it as a gap.

Evaluate strictly against the provided Job Description text and retrieved resume sections:
1. Forbid prior-knowledge bias: evaluate strictly against what is written in the JD, nothing else.
2. If a core technology, framework, or responsibility listed in the Job Description is missing from the resume, you MUST document it as a gap.
3. If the candidate explicitly possesses the exact skill in their uploaded resume text, do not document it as a gap.
4. Do not invent unlisted requirements, and do not overlook genuine missing qualifications.
"""


def node_gap_analysis(state: ResumeAnalysisState) -> dict:
    """Gap analysis pre-pass: evaluates candidate alignment and prepares state for scoring.

    The primary gap identification and scoring are consolidated in node_score_coach
    to avoid redundant inference latency while preserving strict text matching.
    """
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("analyzing_gaps", "Scanning resume against JD with strict text matching…")
    logger.info("Node 2 — gap analysis pre-pass complete (grounded text matching active)")
    return {
        "gap_analysis_text": "",
        "gaps": [],
    }


# ── Node 3: Full consolidated analysis ────────────────────────────────────────


def node_score_coach(state: ResumeAnalysisState) -> dict:
    """Single consolidated LLM call: gap identification + scoring + improvements + prep.

    Enforces strict grounding, forbids prior-knowledge bias, and eliminates the forced 3 gaps trap.
    """
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("scoring", "Running deep analysis…")

    parser = PydanticOutputParser(pydantic_object=ScoreCoachOutput)
    format_instructions = parser.get_format_instructions()
    # Support Dual Provider (local Ollama or Groq Cloud API)
    llm = _build_llm(state.get("ollama_model_name"), temperature=0.0)

    # Provide comprehensive context of up to 10 top-ranked retrieved chunks
    resume_context = "\n\n".join(state["retrieved_chunks"][:10])
    jd = state["job_description"]

    keywords = extract_jd_keywords(jd, resume_context)
    found_kw = [k["keyword"] for k in keywords if k.get("found_in_resume")]
    missing_kw = [k["keyword"] for k in keywords if not k.get("found_in_resume")]
    kw_reconciliation = (
        f"- Confirmed Present in Resume: {', '.join(found_kw) if found_kw else 'None'}\n"
        f"- Explicitly Missing from Resume: {', '.join(missing_kw) if missing_kw else 'None'}"
    )

    recruiter_directive = (
        "You are a strict, highly analytical technical recruiter. You must perform a rigorous cross-reference of the "
        "Job Description against the candidate's resume. If a core technology, framework, or responsibility listed in the "
        "Job Description is missing from the resume, you MUST document it as a gap. Do not be lenient. "
        "However, if the candidate explicitly possesses the exact skill, do not document it as a gap."
    )

    scoring_rubric = (
        "MATHEMATICAL SCORING RUBRIC:\n"
        "You must calculate the final score logically. Start at 10/10. Apply the following strict deductions:\n"
        "- Deduct 2 to 3 points if core programming languages or primary frameworks are missing.\n"
        "- Deduct 2 points if the candidate lacks the required years of experience or seniority.\n"
        "- Deduct 1 to 2 points if secondary tools or cloud platforms are missing.\n"
        "- A resume that lacks the majority of the JD requirements MUST score below a 4/10.\n"
        "- Never award a 10/10 unless the candidate is a flawless match."
    )

    prompt = (
        f"{recruiter_directive}\n\n"
        f"{scoring_rubric}\n\n"
        "STRICT GROUNDING INSTRUCTIONS:\n"
        "1. Forbid prior-knowledge bias: evaluate strictly against what is written in the JD, nothing else.\n"
        "2. Cross-reference every core requirement, technology, framework, and responsibility in the JD with the RESUME CONTEXT.\n"
        "3. If a core technology, framework, or responsibility listed in the Job Description is missing from the resume, you MUST document it as a gap. Do not be lenient.\n"
        "4. If a required skill or tool (e.g., 'Redis', 'OCI', 'Docker', 'FastAPI', 'Python', 'C++', 'C#') is explicitly present in the candidate's resume text, IT IS NOT A GAP. Do NOT claim the candidate lacks something they explicitly have.\n"
        "5. Review the ATS KEYWORD RECONCILIATION below. Any skills marked as 'Explicitly Missing from Resume' MUST be factored into your gap analysis and score deductions.\n"
        "6. ONLY penalize for skills, technologies, or qualifications that are EXPLICITLY WRITTEN in the JOB DESCRIPTION below.\n"
        "7. DO NOT invent or assume unlisted tools, cloud providers, or frameworks not mentioned in the JD.\n"
        "8. For every identified gap, provide a concrete resume bullet improvement and an interview preparation roadmap item.\n"
        "9. Calculate the final score using the MATHEMATICAL SCORING RUBRIC above. Do not default to high scores when requirements are missing.\n\n"
        "JOB DESCRIPTION:\n"
        f"{jd}\n\n"
        "RESUME CONTEXT (retrieved sections):\n"
        f"{resume_context}\n\n"
        "ATS KEYWORD RECONCILIATION:\n"
        f"{kw_reconciliation}\n\n"
        f"{format_instructions}\n\n"
        "OUTPUT SCHEMA EXAMPLE (Return ONLY valid JSON matching this structure):\n"
        "{\n"
        '  "score": 3,\n'
        '  "gaps": [\n'
        '    "Missing JD Requirement: The resume lacks evidence for <Explicit Skill from JD>. The JD explicitly states: <Quote exact requirement from JD>."\n'
        "  ],\n"
        '  "improvements": [\n'
        '    "Target Area: Experience Section | Action Required: Add an explicit bullet point demonstrating how you implemented <Missing Skill> with measurable production impact. | JD Alignment: Directly satisfies the explicit JD requirement for <Missing Skill>."\n'
        "  ],\n"
        '  "preparation": [\n'
        '    "Target Gap: <Missing Skill> | Study: Key official documentation and best practices for <Missing Skill> | Practice: Build a reference demo implementing <Missing Skill> | Interview Angle: How do you address common production challenges in <Missing Skill>?"\n'
        "  ]\n"
        "}\n\n"
        "RULES:\n"
        "- score: integer (0-10) calculated strictly using the Mathematical Scoring Rubric. Start at 10/10 and apply deductions for missing core languages, frameworks, experience, and secondary tools. Resumes missing the majority of requirements MUST score below 4. Never award a 10/10 unless the candidate is a flawless match.\n"
        "- gaps: list of strings. Document every genuine missing requirement from the JD. If a skill is explicitly present in the resume, do NOT flag it. Only return [] if the candidate is truly a flawless match.\n"
        "- improvements: list of actionable resume modification advice addressing each identified gap. Format: 'Target Area: NAME | Action Required: DIRECTIVE | JD Alignment: EXPLANATION'.\n"
        "- preparation: list of interview study topics addressing each identified gap. Format: 'Target Gap: NAME | Study: RESOURCE | Practice: PROJECT | Interview Angle: QUESTION'.\n"
        "- Return ONLY valid JSON. No markdown fences. Zero hallucinations."
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
