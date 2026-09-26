"""
LangGraph pipeline for resume analysis.

Pipeline (2 nodes)
------------------
1. ``node_extract_and_retrieve``
   Parses the resume PDF once with layout-aware extraction. Short resumes are
   passed through whole; longer ones are section-chunked and narrowed by hybrid
   (dense + BM25) multi-query retrieval. The FULL extracted text is always kept
   in state, separately from the retrieved context.

2. ``node_score_coach``
   One consolidated LLM call producing score, gaps, improvements and interview
   preparation, validated against ``ScoreCoachOutput``.

Grounding design notes
----------------------
* ``SYSTEM_PROMPT`` is prepended to every request. It used to be defined and never
  referenced, so the only directive reaching the model was the "do not be lenient"
  recruiter framing — pressure in the direction of inventing gaps, with nothing
  pushing back.
* ATS keyword matching runs against ``full_resume_text``, never against the
  retrieved subset. Matching against the subset marked skills as missing purely
  because their chunk lost the retrieval ranking, and the prompt then instructed
  the model to deduct points for them.
* The keyword table is presented to the model as a lint hint, explicitly
  subordinate to the resume text, rather than as an authoritative MUST-deduct list.
* Post-processing never changes the meaning of model output. Verb rewrites that
  upgraded "participated in" to "spearheaded and executed" fabricated stronger
  claims than the model made.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable, List, Optional, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph

from .config import settings
from .models import GapAnalysisOutput, ScoreCoachOutput, normalize_list_item
from .rag import (
    extract_layout_aware_text,
    is_single_page_resume,
    retrieve_relevant_chunks,
    section_aware_chunk_resume,
)

logger = logging.getLogger(__name__)

# Backwards-compatible alias: tests/test_parser.py imports this from agent.
# The canonical implementation now lives in models.py so the Pydantic
# mode="before" validators can call it without a circular import.
_normalize_list_item = normalize_list_item


# ── Progress callback type ────────────────────────────────────────────────────
# Called by each node so the server can emit SSE events.  Signature:
#   callback(stage: str, message: str) -> None
ProgressCallback = Optional[Callable[[str, str], None]]


# ── LangGraph state schema ────────────────────────────────────────────────────


class ResumeAnalysisState(TypedDict):
    """Typed state that flows through the LangGraph pipeline."""

    # ── Inputs (set by the server before graph.invoke) ──────────────────
    job_description: str
    resume_pdf_bytes: bytes
    embeddings_model_name: str
    ollama_model_name: str
    progress_callback: ProgressCallback  # optional; ignored by LangGraph routing

    # ── Node 1 outputs ───────────────────────────────────────────────────
    full_resume_text: str      # complete extracted text — authoritative for keyword matching
    resume_chunks: List[str]
    retrieved_chunks: List[str]

    # ── Node 2 / final outputs ───────────────────────────────────────────
    improvements: List[str]
    preparation: List[str]
    gaps: List[str]
    score: int
    keywords: list[dict]
    parse_failed: bool         # True when the LLM output could not be parsed at all


# ── System prompt ─────────────────────────────────────────────────────────────
# Prepended to the analysis prompt. langchain_community.llms.Ollama is a
# completion model with no system role, so this is concatenated rather than
# passed as a separate message.

SYSTEM_PROMPT = """\
You are a strict technical recruiter performing evidence-based resume screening.

YOUR ONLY JOB: Compare THIS specific candidate's resume against THIS specific job description.
You are NOT evaluating a generic candidate. You are evaluating the INDIVIDUAL whose resume appears below.

ABSOLUTE GROUNDING RULES (these override every other instruction):
1. The RESUME TEXT below is the sole source of truth about this candidate. Read it word-by-word.
   If it is not written in the resume, the candidate does not have it. Period.
2. The JOB DESCRIPTION is the sole source of truth about requirements. Do not add requirements
   the JD omits, however conventional they seem for the role.
3. If evidence for a requirement appears ANYWHERE in the resume text (even in a project or
   achievements section), that requirement IS satisfied. Do NOT flag it as a gap.
4. Every gap you report MUST include a direct verbatim quote from the job description proving
   the requirement exists. If you cannot find that quote, it is NOT a gap — omit it.
5. Every gap you report MUST also confirm: "I searched the resume text above and found NO
   mention of [X]." If you DID find a mention, you must NOT report it as a gap.
6. Report only the gaps the evidence actually supports. Zero gaps is a valid and common answer.
7. NEVER suggest an improvement for something the resume already shows.
8. DO NOT use your training knowledge about what skills are "typical" for a role. Only use
   what is explicitly stated in the JD and the resume.

CRITICAL: Two different candidates with different resumes MUST produce different analyses.
If your analysis looks identical to what you would write for a different candidate,
you have failed to read the resume and must start over.

Your output must be a single valid JSON object matching the requested schema.
"""


# ── LLM factory (Dual Provider: Ollama vs. Groq) ──────────────────────────────


def _build_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Instantiate the configured LLM provider: local Ollama or Groq Cloud API."""
    deployment_env = (os.getenv("DEPLOYMENT_ENV") or getattr(settings, "deployment_env", "local")).lower().strip()
    provider = getattr(settings, "llm_provider", "ollama").lower().strip()
    use_cloud = (deployment_env == "cloud") or (provider == "groq")

    if use_cloud:
        from langchain_groq import ChatGroq

        groq_model = getattr(settings, "groq_model", "openai/gpt-oss-120b")
        groq_api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required when DEPLOYMENT_ENV=cloud or LLM_PROVIDER=groq. "
                "Add it as a deployment secret; do not put it in source code."
            )
        logger.info("Instantiating Groq LLM provider: model=%s", groq_model)
        return ChatGroq(
            model=groq_model,
            temperature=0.0,
            groq_api_key=groq_api_key,
            reasoning_effort="low",
            model_kwargs={
                "response_format": {"type": "json_object"},
            },
        )

    target_model = model or settings.ollama_model
    num_ctx = getattr(settings, "ollama_num_ctx", 8192)
    num_predict = getattr(settings, "ollama_num_predict", 1500)
    logger.info(
        "Instantiating local Ollama LLM provider: model=%s num_ctx=%d num_predict=%d",
        target_model, num_ctx, num_predict,
    )

    base_kwargs = dict(
        model=target_model,
        temperature=0.0,          # Hardcoded: eliminates creative drift
        base_url=settings.ollama_base_url,
        format="json",
        num_ctx=num_ctx,          # Must cover prompt + generation, or Ollama silently truncates
        num_predict=num_predict,
        keep_alive="15m",
        # qwen3.5:9b is a "thinking" model: by default it generates a long
        # <think>...</think> internal chain-of-thought that exhausts num_predict
        # before writing a single JSON character. reasoning=False disables this.
        # This is the LangChain OllamaLLM equivalent of Ollama's think=false API option.
        reasoning=False,
    )

    # Sampling overrides applied opportunistically: the LangChain Ollama wrappers
    # reject unknown fields by version, so a rejection falls back to the base
    # configuration (which already has reasoning=False).
    # qwen3.5's Modelfile sets presence_penalty 1.5, which pushes the model away
    # from reusing repeated JSON keys; repeat_penalty=1.0 neutralises it.
    tuning_kwargs = dict(
        top_k=getattr(settings, "ollama_top_k", 20),
        top_p=getattr(settings, "ollama_top_p", 0.95),
        repeat_penalty=getattr(settings, "ollama_repeat_penalty", 1.0),
    )

    # langchain_community.llms.Ollama is deprecated; prefer langchain_ollama.
    try:
        from langchain_ollama import OllamaLLM as _OllamaCls
    except ImportError:
        from langchain_community.llms import Ollama as _OllamaCls

        logger.debug("langchain_ollama unavailable; falling back to deprecated community Ollama")

    try:
        return _OllamaCls(**base_kwargs, **tuning_kwargs)
    except Exception as exc:
        logger.warning(
            "LLM wrapper rejected sampling overrides (%s); using base configuration "
            "(reasoning=False still active).",
            exc,
        )
        return _OllamaCls(**base_kwargs)


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
    """Strip markdown fences and remove trailing commas inside objects/arrays."""
    s = _strip_json_fences(s)
    # `[ "foo", ]` -> `[ "foo" ]`
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def _clean_gap_text(text: str) -> str:
    """Collapse duplicated skill headers, e.g. 'Python: Python: missing' -> 'Python: missing'."""
    s = text.strip()
    parts = s.split(":", 2)
    if len(parts) >= 3 and parts[0].strip().lower() == parts[1].strip().lower():
        s = f"{parts[0].strip()}: {parts[2].strip()}"
    return s.strip()


def _clean_improvement_text(text: str) -> str:
    """Remove prompt artifacts from improvement text.

    Deliberately does NOT rewrite verbs. The previous implementation mapped
    "participated in" to "spearheaded and executed" and similar, which invented
    stronger claims than the model produced — hallucination introduced after the
    LLM rather than by it. Tense and phrasing are the model's job now.
    """
    s = text.strip()
    s = re.sub(r"^Add bullet:\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*[—-]\s*addresses gap:.*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _normalize_model_lists(obj: Any) -> Any:
    """Apply cosmetic cleanup to already-validated list fields.

    Structural normalization happens in models.py via the mode="before" validators;
    this only handles presentation artifacts.
    """
    if hasattr(obj, "gaps") and isinstance(obj.gaps, list):
        obj.gaps = [t for t in (_clean_gap_text(x) for x in obj.gaps) if t]
    if hasattr(obj, "improvements") and isinstance(obj.improvements, list):
        obj.improvements = [t for t in (_clean_improvement_text(x) for x in obj.improvements) if t]
    if hasattr(obj, "preparation") and isinstance(obj.preparation, list):
        obj.preparation = [x.strip() for x in obj.preparation if x and x.strip()]
    return obj


def _extract_json_blob(cleaned: str) -> Any:
    """Best-effort extraction of a JSON object from text that may wrap or trail it.

    Prefers a balanced top-level object. A bare `[...]` match is NOT accepted here:
    when generation is truncated mid-object, the first complete inner array (e.g.
    the finished `gaps` list) would match and be mistaken for the whole payload,
    silently discarding the score and every other field.
    """
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = re.sub(r",\s*([\]}])", r"\1", cleaned[start:i + 1])
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    # Unbalanced: generation was cut off. Signal truncation rather than guessing.
    logger.warning("JSON object never closed — output was truncated (raise ollama_num_predict).")
    return None


def _parse_llm_output(raw: str, parser: PydanticOutputParser, model_cls):
    """Parse *raw* LLM text into a Pydantic model, with a self-healing fallback chain.

    On total failure returns an instance with ``parse_failed=True`` so the caller can
    surface an error. Previously this returned an all-defaults model, which rendered
    as a legitimate 0/100 with no gaps and was written to history as a real analysis.
    """
    cleaned = _clean_json_str(raw)

    # ── Attempt 1: PydanticOutputParser ───────────────────────────────────
    try:
        return _normalize_model_lists(parser.parse(cleaned))
    except Exception:
        pass

    # ── Attempt 2: direct json.loads + model construction ─────────────────
    data: Any = None
    try:
        data = json.loads(cleaned)
    except Exception:
        data = _extract_json_blob(cleaned)

    if isinstance(data, dict):
        # Unwrap schema echoes: small models often return the JSON Schema shape
        # ({"properties": {...}}) instead of an instance of it.
        for wrapper_key in ("properties", "output", "result", "response",
                            "ScoreCoachOutput", "GapAnalysisOutput"):
            inner = data.get(wrapper_key)
            if isinstance(inner, dict):
                data = inner
                break

        try:
            if model_cls is GapAnalysisOutput:
                gaps = _first_list(data, ["gaps", "technical_gaps", "skill_gaps", "gaps_identified"])
                return _normalize_model_lists(GapAnalysisOutput(gaps=gaps))

            if model_cls is ScoreCoachOutput:
                # A missing score used to be silently converted to 0 by the
                # Pydantic default, making an incomplete LLM response look like
                # a real "no match" verdict in the UI and history.
                if not any(data.get(key) is not None for key in (
                    "score", "match_score", "final_score", "rating", "matchScore",
                )):
                    raise ValueError("LLM response omitted the required score field")
                score = _coerce_score(data)
                result = ScoreCoachOutput(
                    score=score,
                    gaps=_first_list(data, ["gaps", "technical_gaps", "skill_gaps", "identified_gaps"]),
                    improvements=_first_list(data, [
                        "improvements", "recommendations", "resume_improvements",
                        "suggestions", "bullet_points",
                    ]),
                    preparation=_first_list(data, [
                        "preparation", "interview_prep", "prep_plan",
                        "study_topics", "preparation_steps",
                    ]),
                )
                return _normalize_model_lists(result)

            return _normalize_model_lists(model_cls(**data))
        except Exception as exc:
            logger.error("Schema normalization failed: %s", exc)

    # ── Attempt 3: give up loudly ─────────────────────────────────────────
    logger.error(
        "All parse attempts failed for %s. Raw output (first 600 chars): %s",
        model_cls.__name__, (raw or "")[:600],
    )
    return model_cls.model_construct(
        score=0, gaps=[], improvements=[], preparation=[], parse_failed=True
    )


def _first_list(data: dict, keys: list[str]) -> list:
    """Return the first present list-or-string field among *keys*, split if a string."""
    for k in keys:
        val = data.get(k)
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val.strip():
            return [s for s in re.split(r"\n|;", val) if s.strip()]
    return []


def _coerce_score(data: dict) -> int:
    """Pull an integer 0-100 score out of whatever key the model used."""
    raw_score = data.get("score")
    if raw_score is None:
        for sk in ("match_score", "final_score", "rating", "matchScore"):
            if data.get(sk) is not None:
                raw_score = data[sk]
                break

    value = 0
    if isinstance(raw_score, bool):
        value = 0
    elif isinstance(raw_score, (int, float)):
        value = int(raw_score)
    elif isinstance(raw_score, str):
        m = re.search(r"\d+", raw_score)
        if m:
            value = int(m.group(0))
    return max(0, min(100, value))


def calculate_evidence_score(keywords: list[dict], gaps: list[str]) -> int:
    """Calculate a stable, explainable score from verified evidence.

    Keyword matching is deterministic and runs over the complete parsed resume.
    The LLM is used to explain JD-specific gaps, but it must not be able to turn
    a partially matched resume into a 0 simply by emitting an arbitrary score.
    Each evidence-grounded gap applies a modest eight-point deduction, capped at
    24 points so the score remains proportional to measured keyword coverage.
    """
    total = len(keywords)
    if not total:
        # The JD contains no supported measurable keywords. In this uncommon
        # case there is no defensible numeric evidence score.
        logger.warning("calculate_evidence_score: 0 keywords extracted — returning 0")
        return 0

    found = sum(1 for keyword in keywords if keyword.get("found_in_resume"))
    keyword_coverage = round(found / total * 100)
    gap_deduction = min(len(gaps) * 8, 24)
    final_score = max(0, keyword_coverage - gap_deduction)
    logger.info(
        "calculate_evidence_score: %d/%d keywords matched (%d%%) — %d gaps (-%d pts) → score=%d",
        found, total, keyword_coverage, len(gaps), gap_deduction, final_score,
    )
    return final_score


# ── ATS keyword extraction ────────────────────────────────────────────────────

_CURATED_KEYWORDS = {
    "skill": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "Ruby", "Kotlin", "Swift", "Scala", "R", "MATLAB", "SQL",
        "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "LangChain",
        "LangGraph", "Ollama", "OpenAI", "Hugging Face", "FAISS", "Pinecone",
        "ChromaDB", "Weaviate", "RAG", "LLM", "NLP",
    ],
    "tool": [
        "React", "Angular", "Vue", "Django", "Flask", "FastAPI", "Spring",
        "Express", "Node.js", "Next.js", ".NET", "ASP.NET",
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "DynamoDB",
        "Git", "Jenkins", "GitHub Actions", "CI/CD", "Jira",
    ],
    "certification": [
        "AWS Certified", "PMP", "Scrum",
    ],
}

# Lead-in phrases that introduce a requirement in prose JDs.
_NER_LEAD_INS = (
    r"experience with", r"experience in", r"proficiency in", r"proficient in",
    r"knowledge of", r"familiarity with", r"expertise in", r"hands[- ]on with",
    r"skilled in", r"background in",
)

# Generic prose that is not a technology. A candidate phrase containing any of
# these is discarded, because checking a resume for the literal string
# "Strong Communication Skills" always fails and manufactures a false gap.
_NER_NOISE_TOKENS = {
    "strong", "excellent", "good", "solid", "proven", "working", "related",
    "similar", "plus", "bonus", "nice", "communication", "skills", "skill",
    "ability", "abilities", "experience", "team", "teams", "years", "year",
    "environment", "environments", "fast", "paced", "development", "developing",
    "methodologies", "methodology", "understanding", "knowledge", "principles",
    "practices", "best", "concepts", "fundamentals", "tools", "technologies",
    "systems", "software", "applications", "a", "an", "the", "and", "or",
    "large", "scale", "modern", "various", "multiple", "complex", "real",
    "world", "production", "enterprise", "cross", "functional", "problem",
    "solving", "written", "verbal", "attention", "detail", "degree",
    "bachelor", "master", "computer", "science", "field",
}


def _keyword_pattern(term: str) -> str:
    """Symbol-aware boundary matcher: handles C++, C#, .NET, Node.js, React.js."""
    return r"(?i)(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])"


def _phrase_present(phrase: str, text: str) -> bool:
    """True if *phrase* appears verbatim, or if every one of its tokens appears.

    Resumes rarely repeat a JD's exact multi-word phrasing. Requiring a verbatim
    match on phrases produced a steady stream of false "missing" keywords, so
    token-level coverage counts as present.
    """
    if re.search(_keyword_pattern(phrase), text):
        return True
    tokens = [t for t in phrase.split() if t]
    if len(tokens) < 2:
        return False
    return all(re.search(_keyword_pattern(t), text) for t in tokens)


def extract_jd_keywords(jd_text: str, resume_text: str) -> list[dict]:
    """Extract skills/tools from the JD and check presence in the FULL resume text.

    ``resume_text`` must be the complete extracted resume, not a retrieved subset.
    Passing a subset marks skills absent merely because their chunk lost the
    retrieval ranking.
    """
    keywords_found: list[dict] = []
    seen: set[str] = set()

    for category, terms in _CURATED_KEYWORDS.items():
        for term in terms:
            pattern = _keyword_pattern(term)
            if not re.search(pattern, jd_text):
                continue
            term_lower = term.lower()
            if term_lower in seen:
                continue
            keywords_found.append({
                "keyword": term,
                "found_in_resume": bool(re.search(pattern, resume_text)),
                "category": category,
            })
            seen.add(term_lower)

    # Prose requirements: capped at 3 capitalized tokens and confined to a single
    # line. `\s+` previously matched newlines, so "Experience with Docker\nStrong
    # Communication Skills" captured the whole run as one keyword.
    token = r"[A-Z][A-Za-z0-9+#.\-]*"
    for lead in _NER_LEAD_INS:
        pattern = (
            r"(?i)\b" + lead + r"[^\S\n]+"
            r"((?:" + token + r")(?:[^\S\n]+(?:" + token + r")){0,2})"
        )
        for match in re.finditer(pattern, jd_text):
            phrase = match.group(1).strip(" ,.;:-")
            if not phrase or len(phrase) < 2 or len(phrase) > 40:
                continue
            tokens = phrase.split()
            if any(t.strip(".,;:-").lower() in _NER_NOISE_TOKENS for t in tokens):
                continue
            phrase_lower = phrase.lower()
            if phrase_lower in seen:
                continue
            keywords_found.append({
                "keyword": phrase,
                "found_in_resume": _phrase_present(phrase, resume_text),
                "category": "skill",
            })
            seen.add(phrase_lower)

    # Surface unmatched keywords first so the 20-item cap does not hide gaps.
    keywords_found.sort(key=lambda k: k["found_in_resume"])
    return keywords_found[:20]


# ── Node 1: PDF extraction + retrieval ───────────────────────────────────────


def node_extract_and_retrieve(state: ResumeAnalysisState) -> dict:
    """Parse the resume PDF once, then choose full-document or retrieved context."""
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("extracting", "Parsing PDF layout and evaluating document length…")

    t0 = time.perf_counter()
    full_text = extract_layout_aware_text(state["resume_pdf_bytes"])

    if not full_text.strip():
        raise ValueError(
            "No text could be extracted from the uploaded PDF. It may be a scanned "
            "image — re-export it as a text-based PDF, or run OCR first."
        )

    word_count = len(full_text.split())
    max_words = getattr(settings, "single_page_max_words", 1000)

    # Short resumes: skip retrieval entirely, the whole document fits the context.
    if is_single_page_resume(full_text, max_words=max_words):
        logger.info(
            "Node 1 — short resume (%d words < %d): using full document, retrieval bypassed (%.2fs)",
            word_count, max_words, time.perf_counter() - t0,
        )
        if callback:
            callback("retrieved", f"Short resume ({word_count} words): full context preserved")
        return {
            "full_resume_text": full_text,
            "resume_chunks": [full_text],
            "retrieved_chunks": [full_text],
        }

    # Longer resumes: section-aware chunking + hybrid retrieval.
    # Chunk the text already extracted above rather than re-parsing the PDF.
    resume_chunks = section_aware_chunk_resume(
        full_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info(
        "Node 1 — PDF chunked: %d chunks from %d words (%.2fs)",
        len(resume_chunks), word_count, time.perf_counter() - t0,
    )

    t1 = time.perf_counter()
    context = retrieve_relevant_chunks(
        job_description=state["job_description"],
        resume_chunks=resume_chunks,
        embeddings_model_name=state["embeddings_model_name"],
        k=settings.retrieval_k,
        top_n=getattr(settings, "retrieval_top_n", 12),
        resume_pdf_bytes=state["resume_pdf_bytes"],
    )
    logger.info(
        "Node 1 — hybrid RAG complete: %d chunks kept via %d queries (%.2fs)",
        len(context.retrieved_chunks), context.query_count, time.perf_counter() - t1,
    )

    if callback:
        callback(
            "retrieved",
            f"Retrieved {len(context.retrieved_chunks)} sections via "
            f"{context.query_count}-query hybrid RAG",
        )

    return {
        "full_resume_text": full_text,
        "resume_chunks": resume_chunks,
        "retrieved_chunks": context.retrieved_chunks,
    }


# ── Node 2: Consolidated analysis ─────────────────────────────────────────────

SCORING_RUBRIC = """\
SCORING (integer 0-100, based only on evidence in the resume text):
  90-100  Every core requirement in the job description is evidenced.
  60-89   Core requirements evidenced; one or two secondary items unevidenced.
  30-59   Several core requirements unevidenced, or evidence is indirect.
  0-29    Most core requirements unevidenced.

Score the evidence you actually found. Do not reserve the top of the scale, and do
not deduct for anything the job description does not ask for. A strong resume that
matches the job description should score high; a weak one should score low.
"""


def _build_analysis_prompt(
    jd: str,
    resume_context: str,
    keywords: list[dict],
    format_instructions: str,
) -> str:
    """Assemble a compact, direct analysis prompt compatible with Ollama format=json.

    Grounding is enforced structurally: each gap entry must include an 'evidence'
    field quoting the exact JD requirement, which forces the model to read the JD
    rather than rely on prior knowledge. The schema example shows this structure.

    The multi-phase scratchpad approach (Phase 1/2/3) was removed because it caused
    qwen3.5:9b to generate thousands of reasoning tokens before the JSON, exhausting
    num_predict and producing an empty response.
    """
    missing_kw = [k["keyword"] for k in keywords if not k.get("found_in_resume")]
    found_kw = [k["keyword"] for k in keywords if k.get("found_in_resume")]

    keyword_hint = (
        "KEYWORD SCAN (regex, advisory only — not authoritative):\n"
        f"  Found in resume: {', '.join(found_kw) if found_kw else 'none'}\n"
        f"  Not found by scan: {', '.join(missing_kw) if missing_kw else 'none'}\n"
        "If the full resume text above contains a 'not found' keyword, the scan is wrong — "
        "do NOT report it as a gap."
    )

    schema_example = (
        "Output ONLY this JSON structure (no markdown, no commentary):\n"
        "{\n"
        '  "score": <integer 0-100>,\n'
        '  "gaps": [\n'
        '    "<skill/requirement>: not evidenced in resume. JD requires: \\"<exact JD quote>\\""\n'
        "  ],\n"
        '  "improvements": [\n'
        '    "Target Area: <section> | Action Required: <edit> | JD Alignment: <requirement>"\n'
        "  ],\n"
        '  "preparation": [\n'
        '    "Target Gap: <skill> | Study: <resource> | Practice: <task> | Interview Angle: <question>"\n'
        "  ]\n"
        "}\n"
        "gaps, improvements, and preparation must be empty lists [] if nothing is unevidenced."
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{SCORING_RUBRIC}\n\n"
        "--- JOB DESCRIPTION ---\n"
        f"{jd}\n\n"
        "--- CANDIDATE RESUME ---\n"
        f"{resume_context}\n"
        "--- END OF RESUME ---\n\n"
        f"{keyword_hint}\n\n"
        "TASK: Compare the candidate resume above against the job description above.\n"
        "For each JD requirement, check if evidence exists in the resume text.\n"
        "Only report a gap when you find ZERO evidence in the resume — quote the JD requirement.\n"
        "Pair each gap with one improvement and one preparation item.\n"
        "The score must reflect what you found in THIS specific resume, not a generic assessment.\n\n"
        f"{schema_example}"
    )

def node_score_coach(state: ResumeAnalysisState) -> dict:
    """Single consolidated LLM call: gaps + score + improvements + preparation."""
    callback: ProgressCallback = state.get("progress_callback")
    if callback:
        callback("scoring", "Running grounded analysis…")

    parser = PydanticOutputParser(pydantic_object=ScoreCoachOutput)
    llm = _build_llm(state.get("ollama_model_name"), temperature=0.0)

    resume_context = "\n\n".join(state["retrieved_chunks"])
    jd = state["job_description"]

    # Keyword matching runs against the FULL resume, never the retrieved subset.
    full_text = state.get("full_resume_text") or resume_context
    keywords = extract_jd_keywords(jd, full_text)

    prompt = _build_analysis_prompt(
        jd=jd,
        resume_context=resume_context,
        keywords=keywords,
        format_instructions=parser.get_format_instructions(),
    )

    t0 = time.perf_counter()
    raw = llm.invoke(prompt)
    raw_text = getattr(raw, "content", str(raw))
    elapsed_llm = time.perf_counter() - t0
    logger.info("Node 2 — consolidated LLM call: %.2fs | output length: %d chars", elapsed_llm, len(raw_text))
    if not raw_text.strip():
        logger.error("Node 2 — LLM returned EMPTY output. Model may be in thinking mode or out of tokens.")
    else:
        logger.debug("Node 2 — raw output (first 300 chars): %s", raw_text[:300])

    result: ScoreCoachOutput = _parse_llm_output(raw_text, parser, ScoreCoachOutput)
    parse_failed = bool(getattr(result, "parse_failed", False))

    if parse_failed:
        logger.error("Node 2 — output unparseable; flagging failure instead of returning 0/100")
        # No progress callback here: the server emits a single terminal `error`
        # event for this case, so emitting one now would send two.
        return {
            "score": 0,
            "gaps": [],
            "improvements": [],
            "preparation": [],
            "keywords": keywords,
            "parse_failed": True,
        }

    gaps = getattr(result, "gaps", []) or []
    improvements = getattr(result, "improvements", []) or []
    preparation = getattr(result, "preparation", []) or []
    score = calculate_evidence_score(keywords, gaps)

    logger.info(
        "Node 2 — final: score=%d gaps=%d improvements=%d preparation=%d",
        score, len(gaps), len(improvements), len(preparation),
    )

    if callback:
        callback("complete", f"Analysis complete — score {score}/100")

    return {
        "score": score,
        "gaps": gaps,
        "improvements": improvements,
        "preparation": preparation,
        "keywords": keywords,
        "parse_failed": False,
    }


# ── Graph compilation ─────────────────────────────────────────────────────────


def build_resume_analysis_graph(model: str = settings.ollama_model):
    """Compile and return the LangGraph analysis pipeline.

    The former ``gap_analysis`` node was a no-op that returned empty values and made
    no LLM call, so it has been removed. Gap identification happens in
    ``score_and_coach``.

    Args:
        model: Retained for backwards compatibility; nodes read
            ``state["ollama_model_name"]`` so the per-request override works.
    """
    _ = model
    graph = StateGraph(ResumeAnalysisState)

    graph.add_node("extract_and_retrieve", node_extract_and_retrieve)
    graph.add_node("score_and_coach", node_score_coach)

    graph.set_entry_point("extract_and_retrieve")
    graph.add_edge("extract_and_retrieve", "score_and_coach")
    graph.add_edge("score_and_coach", END)

    logger.info("LangGraph pipeline compiled: 2 nodes")
    return graph.compile()
