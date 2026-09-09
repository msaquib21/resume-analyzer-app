"""Tests for the self-healing parser in backend/agent.py and layout-aware parsing in backend/rag.py."""

import json
import pytest

from backend.agent import _normalize_list_item, _clean_json_str
from backend.rag import _parse_multicolumn_layout


def test_normalize_list_item_string():
    assert _normalize_list_item("Plain string input") == "Plain string input"


def test_normalize_list_item_dict_gap():
    assert _normalize_list_item({"gap": "Missing SQL experience"}) == "Missing SQL experience"


def test_normalize_list_item_dict_multi():
    """Multi-key gap dict should produce 'SkillName: evidence. JD requires: demand.'"""
    item = {"Skill Name": "Python", "Missing Evidence": "No projects", "JD Demands": "Requires Python"}
    result = _normalize_list_item(item)
    assert "Python" in result
    assert "No projects" in result
    assert "Requires Python" in result


def test_normalize_list_item_dict_action():
    """Improvement dict should produce 'Target Area: X | Action Required: Y'"""
    item = {"Target Area": "Education", "Action Required": "Add GPA"}
    result = _normalize_list_item(item)
    assert "Education" in result
    assert "Add GPA" in result


def test_normalize_list_item_none():
    assert _normalize_list_item(None) == ""


def test_normalize_list_item_json_string():
    assert _normalize_list_item('{"gap": "JSON gap"}') == "JSON gap"


def test_clean_json_str_clean():
    assert _clean_json_str('{"key": "value"}') == '{"key": "value"}'


def test_clean_json_str_markdown():
    result = _clean_json_str('```json\n{"key": "value"}\n```')
    assert '"key"' in result
    assert '"value"' in result


def test_parse_multicolumn_layout_disentangles_columns():
    """Multi-column resumes must be read vertically, not merged horizontally."""
    multicolumn_raw = (
        "Skills: Redis (Vector DB), OCI                     Experience: Senior ML Engineer at TechCorp\n"
        "Certifications: AWS Solution Architect             Led distributed RAG and fine-tuning pipelines\n"
        "Education: B.S. in Computer Science                Deployed FastAPI microservices with Docker\n"
        "Languages: Python, Go, TypeScript                  Managed high-throughput vector databases\n"
    )
    parsed = _parse_multicolumn_layout(multicolumn_raw)
    
    # Skills must appear together before Experience block
    assert "Redis (Vector DB)" in parsed
    assert "OCI" in parsed
    assert "Experience: Senior ML Engineer" in parsed
    
    # Verify that Column 1 skills are not merged horizontally on the same line with Column 2 experience
    for line in parsed.splitlines():
        if "Redis (Vector DB)" in line:
            assert "Senior ML Engineer" not in line, "Redis and Senior ML Engineer should NOT be horizontally merged on the same line"
        if "Certifications: AWS" in line:
            assert "Led distributed RAG" not in line, "Certifications and Experience should NOT be horizontally merged on the same line"


def test_parse_multicolumn_layout_single_column():
    """Single column text should be preserved cleanly without breaking."""
    single_column = (
        "Summary: Experienced AI systems engineer.\n"
        "Work Experience:\n"
        "- Built RAG applications with ChromaDB and LangGraph.\n"
    )
    parsed = _parse_multicolumn_layout(single_column)
    assert "Summary: Experienced AI systems engineer." in parsed
    assert "Work Experience:" in parsed


def test_section_aware_chunk_resume_preserves_dense_skills():
    """Dense skills sections should be preserved in a single chunk, not fractured mid-phrase."""
    from backend.rag import section_aware_chunk_resume

    resume_text = (
        "Alex Mercer\nAI Systems Architect\nalex@example.com\n\n"
        "PROFESSIONAL SUMMARY\n"
        "Staff AI Architect with 8+ years experience building production LLM and RAG systems.\n\n"
        "TECHNICAL SKILLS\n"
        "Languages: Python, Go, TypeScript, C++, SQL\n"
        "Vector Databases: ChromaDB, Pinecone, Redis (Vector DB), Weaviate, Qdrant\n"
        "Cloud & DevOps: OCI, AWS, Docker, Kubernetes, Terraform, Helm\n"
        "ML/AI Frameworks: LangGraph, LangChain, PyTorch, Hugging Face, vLLM, Ollama\n\n"
        "WORK EXPERIENCE\n"
        "Principal AI Engineer at ScaleTech (2021 - Present)\n"
        "- Architected high-throughput RAG search using hybrid BM25 and dense vector index.\n"
        "- Decreased latency by 45% using quantized inference and caching.\n\n"
        "PROJECTS\n"
        "Agentic Resume Matcher\n"
        "- Built multi-agent evaluation pipeline with LangGraph and local LLM runtime.\n\n"
        "EDUCATION\n"
        "B.S. in Computer Science, University of California, Berkeley\n"
    )

    chunks = section_aware_chunk_resume(resume_text)
    assert len(chunks) >= 4

    # Find the skills chunk
    skills_chunk = next((c for c in chunks if "[Technical Skills]" in c), None)
    assert skills_chunk is not None
    # Verify all technical terms are intact in the same chunk
    assert "Redis (Vector DB)" in skills_chunk
    assert "OCI" in skills_chunk
    assert "LangGraph" in skills_chunk
    assert "ChromaDB" in skills_chunk
    assert "Python" in skills_chunk


def test_reciprocal_rank_fusion_hybrid_weights():
    """Hybrid RRF must properly blend dense vector and BM25 sparse lists with weights."""
    from backend.rag import _reciprocal_rank_fusion

    dense_list = ["doc_dense_1", "doc_shared", "doc_dense_2"]
    bm25_list = ["doc_shared", "doc_bm25_1", "doc_bm25_2"]

    # Fusing with dense weight 1.0 and BM25 weight 0.5
    fused = _reciprocal_rank_fusion([dense_list, bm25_list], k=60, weights=[1.0, 0.5])

    # The doc appearing in both lists should be the top-ranked doc
    assert fused[0] == "doc_shared"
    assert "doc_dense_1" in fused
    assert "doc_bm25_1" in fused


def test_pydantic_output_parser_strict_grounding_constraint():
    """Pydantic parser format instructions must contain the strict text-matcher instruction."""
    from langchain_core.output_parsers import PydanticOutputParser
    from backend.models import GapAnalysisOutput, ScoreCoachOutput

    expected_constraint = (
        "You are a strict text-matcher. Base gaps ONLY on the provided job description text. "
        "Do not hallucinate industry standards (e.g., AWS, Pinecone) if they are not explicitly written."
    )

    gap_parser = PydanticOutputParser(pydantic_object=GapAnalysisOutput)
    gap_instructions = gap_parser.get_format_instructions()
    assert expected_constraint in gap_instructions

    score_parser = PydanticOutputParser(pydantic_object=ScoreCoachOutput)
    score_instructions = score_parser.get_format_instructions()
    assert expected_constraint in score_instructions


def test_is_single_page_resume_bypass():
    """Documents under 1,000 words must trigger the single-page RAG bypass."""
    from backend.rag import is_single_page_resume

    short_text = "Experienced software engineer with Python and Docker skills. " * 30  # ~240 words
    long_text = "Experienced software engineer with Python and Docker skills. " * 200  # ~1600 words

    assert is_single_page_resume(short_text, max_words=1000) is True
    assert is_single_page_resume(long_text, max_words=1000) is False


def test_extract_keywords_technical_symbols():
    """ATS keyword extraction must accurately match C++, C#, .NET, Node.js without regex boundary breakage."""
    from backend.rag import _extract_skill_keywords
    from backend.agent import extract_jd_keywords

    jd_sample = "Looking for a Senior Software Engineer proficient in C++, C#, .NET, and Node.js with React.js experience."
    resume_sample = "Software Engineer with 4 years hands-on experience in C++ and .NET microservices. Also built Node.js backends."

    # 1. RAG query keywords
    rag_keywords = _extract_skill_keywords(jd_sample)
    assert any("C++" in kw.upper() for kw in rag_keywords)
    assert any("C#" in kw.upper() for kw in rag_keywords)
    assert any(".NET" in kw.upper() for kw in rag_keywords)

    # 2. Agent keyword matcher
    agent_keywords = extract_jd_keywords(jd_sample, resume_sample)
    kw_names = {k["keyword"] for k in agent_keywords}
    assert "C++" in kw_names
    assert "C#" in kw_names

    cpp_entry = next(k for k in agent_keywords if k["keyword"] == "C++")
    assert cpp_entry["found_in_resume"] is True

    csharp_entry = next(k for k in agent_keywords if k["keyword"] == "C#")
    assert csharp_entry["found_in_resume"] is False


def test_models_flexible_gap_count():
    """Schemas must not force a static count of 3 items — empty lists must be valid for strong fits."""
    from backend.models import GapAnalysisOutput, ScoreCoachOutput

    # Candidate satisfies everything: 0 gaps, 0 improvements, 0 prep
    gap_output = GapAnalysisOutput(gaps=[])
    assert gap_output.gaps == []

    perfect_score = ScoreCoachOutput(
        score=10,
        gaps=[],
        improvements=[],
        preparation=[],
    )
    assert perfect_score.score == 10
    assert perfect_score.gaps == []
    assert perfect_score.improvements == []
    assert perfect_score.preparation == []


def test_build_llm_dual_provider_selection(monkeypatch):
    """LLM factory must return ChatGroq when provider is 'groq' and Ollama when 'ollama'."""
    from backend.config import settings
    from backend.agent import _build_llm
    from langchain_groq import ChatGroq
    from langchain_community.llms import Ollama

    # Test Groq provider
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "gsk_test_mock_key")
    groq_llm = _build_llm()
    assert isinstance(groq_llm, ChatGroq)
    assert groq_llm.model_name == "llama-3.1-8b-instant"

    # Test Ollama provider
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    ollama_llm = _build_llm(model="qwen3.5:9b")
    try:
        from langchain_ollama import OllamaLLM
        expected_types = (Ollama, OllamaLLM)
    except ImportError:
        expected_types = (Ollama,)
    assert isinstance(ollama_llm, expected_types)
    assert getattr(ollama_llm, "model", None) == "qwen3.5:9b"


def test_empty_string_schema_failure_mitigation():
    """Verify that empty strings, whitespace, and empty dict placeholders are sanitized to empty lists []."""
    from backend.models import GapAnalysisOutput, ScoreCoachOutput, AnalysisResponse
    from backend.agent import _normalize_model_lists

    # 1. GapAnalysisOutput validator
    gap_out = GapAnalysisOutput(gaps=["", "   ", '""', "none", "n/a", "null", "[]", "{}"])
    assert gap_out.gaps == []

    # 2. ScoreCoachOutput validator
    score_out = ScoreCoachOutput(
        score=10,
        gaps=["", "  ", "null"],
        improvements=[{}, {"Action Required": "   "}, "  ", "''"],
        preparation=["\n\t", "n/a", "{}"],
    )
    assert score_out.score == 10
    assert score_out.gaps == []
    assert score_out.improvements == []
    assert score_out.preparation == []

    # 3. Agent _normalize_model_lists defense
    model_obj = ScoreCoachOutput(
        score=9,
        gaps=["   "],
        improvements=[""],
        preparation=["  "],
    )
    normalized = _normalize_model_lists(model_obj)
    assert normalized.gaps == []
    assert normalized.improvements == []
    assert normalized.preparation == []

    # 4. AnalysisResponse API contract
    api_resp = AnalysisResponse(
        score=10,
        gaps=["", " "],
        improvements=[{"Target Area": " "}],
        preparation=["null"],
    )
    assert api_resp.gaps == []
    assert api_resp.improvements == []
    assert api_resp.preparation == []


