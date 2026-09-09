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
You are a technical recruiter performing evidence-based resume screening.

GROUNDING RULES (these override any other instruction):
1. The RESUME TEXT is the sole source of truth about the candidate. The JOB
   DESCRIPTION is the sole source of truth about what is required.
2. Never introduce a technology, tool, cloud platform, framework, certification
   or qualification that does not appear verbatim in the job description. Do not
   supply requirements the job description omits, however conventional they seem
   for the role.
3. If evidence for a requirement appears anywhere in the resume text, that
   requirement is satisfied. Do not flag it, and do not suggest adding it.
4. Every gap you report must quote the exact requirement wording from the job
   description. If you cannot quote it, it is not a gap — omit it.
5. Report the number of gaps the evidence actually supports. Zero is a valid
   answer. Do not pad the list to reach a target count.
6. Never suggest an improvement describing something the resume already contains.

Your output must be a single valid JSON object matching the requested schema.
"""


# ── LLM factory (Dual Provider: Ollama vs. Groq) ──────────────────────────────


def _build_llm(model: Optional[str] = None, temperature: float = 0.0):
    """Instantiate the configured LLM provider: local Ollama or Groq Cloud API."""
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
    )

    # Sampling overrides. qwen3.5's Modelfile sets presence_penalty 1.5, which pushes
    # the model away from reusing the repeated JSON keys and field labels this schema
    # requires. repeat_penalty is neutralised for the same reason.
    # Applied opportunistically: the LangChain Ollama wrappers reject unknown fields
    # and expose different options by version, so a rejection falls back to the base
    # configuration instead of failing the run.
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
            "LLM wrapper rejected sampling overrides (%s); using base configuration. "
            "If output drifts from the schema mid-response, set the parameters in a "
            "Modelfile instead — see the README.",
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
    as a legitimate 0/10 with no gaps and was written to history as a real analysis.
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
    """Pull an integer 0-10 score out of whatever key the model used."""
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
    return max(0, min(10, value))


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
SCORING (integer 0-10, based only on evidence in the resume text):
  9-10  Every core requirement in the job description is evidenced.
  6-8   Core requirements evidenced; one or two secondary items unevidenced.
  3-5   Several core requirements unevidenced, or evidence is indirect.
  0-2   Most core requirements unevidenced.

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
    """Assemble the single consolidated analysis prompt."""
    missing_kw = [k["keyword"] for k in keywords if not k.get("found_in_resume")]
    found_kw = [k["keyword"] for k in keywords if k.get("found_in_resume")]

    keyword_hint = (
        "AUTOMATED KEYWORD SCAN (advisory only)\n"
        "A regex scan of the resume produced the lists below. It is literal string\n"
        "matching with no understanding of synonyms, abbreviations or context, so it\n"
        "reports false negatives. The RESUME TEXT above is authoritative: if a term\n"
        "listed as unmatched is in fact evidenced in the resume, disregard this scan\n"
        "and do not flag it.\n"
        f"- Matched: {', '.join(found_kw) if found_kw else 'none'}\n"
        f"- Not matched by the scan: {', '.join(missing_kw) if missing_kw else 'none'}"
    )

    schema_example = (
        "OUTPUT EXAMPLE (structure only — do not reuse this content):\n"
        "{\n"
        '  "score": 6,\n'
        '  "gaps": [\n'
        '    "<Requirement>: no evidence in the resume. The job description states: '
        '\\"<exact quoted wording from the JD>\\"."\n'
        "  ],\n"
        '  "improvements": [\n'
        '    "Target Area: <resume section> | Action Required: <specific edit> | '
        'JD Alignment: <which quoted requirement this satisfies>"\n'
        "  ],\n"
        '  "preparation": [\n'
        '    "Target Gap: <requirement> | Study: <topic> | Practice: <exercise> | '
        'Interview Angle: <likely question>"\n'
        "  ]\n"
        "}"
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{SCORING_RUBRIC}\n\n"
        "TASK\n"
        "Compare the resume against the job description and produce the JSON object.\n"
        "For each requirement written in the job description, look for supporting\n"
        "evidence in the resume text. Report a gap only where you find none, and quote\n"
        "the requirement wording when you do. Pair every gap with one improvement and\n"
        "one preparation item. If nothing is unevidenced, return empty lists.\n\n"
        "JOB DESCRIPTION:\n"
        f"{jd}\n\n"
        "RESUME TEXT:\n"
        f"{resume_context}\n\n"
        f"{keyword_hint}\n\n"
        f"{format_instructions}\n\n"
        f"{schema_example}\n\n"
        "Return only the JSON object. No markdown fences, no commentary."
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
    logger.info("Node 2 — consolidated LLM call: %.2fs", time.perf_counter() - t0)

    result: ScoreCoachOutput = _parse_llm_output(raw_text, parser, ScoreCoachOutput)
    parse_failed = bool(getattr(result, "parse_failed", False))

    if parse_failed:
        logger.error("Node 2 — output unparseable; flagging failure instead of returning 0/10")
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

    score = max(0, min(10, getattr(result, "score", 0) or 0))
    gaps = getattr(result, "gaps", []) or []
    improvements = getattr(result, "improvements", []) or []
    preparation = getattr(result, "preparation", []) or []

    logger.info(
        "Node 2 — final: score=%d gaps=%d improvements=%d preparation=%d",
        score, len(gaps), len(improvements), len(preparation),
    )

    if callback:
        callback("complete", f"Analysis complete — score {score}/10")

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