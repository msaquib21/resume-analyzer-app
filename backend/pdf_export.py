import logging
from datetime import datetime
from typing import Any, Dict, List
from fpdf import FPDF

logger = logging.getLogger(__name__)


def generate_report(
    score: int,
    gaps: List[Any],
    improvements: List[Any],
    preparation: List[Any],
    keywords: List[Dict[str, Any]],
    resume_filename: str,
    jd_snippet: str,
    elapsed_seconds: float,
) -> bytes:
    """Generate a clean, professional PDF report from analysis results using fpdf2."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Safe text helper to avoid latin-1 encoding crashes
    def safe(text: Any) -> str:
        return str(text).encode("latin-1", "replace").decode("latin-1")

    epw = pdf.epw  # Effective page width (total width - margins)

    # ── Header ────────────────────────────────────────────────────────
    pdf.set_font("helvetica", "B", 18)
    pdf.set_text_color(20, 20, 30)
    pdf.cell(epw, 10, "Resume Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(100, 100, 120)
    pdf.cell(
        epw,
        7,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Resume: {safe(resume_filename)}",
        new_x="LMARGIN",
        new_y="NEXT",
        align="C",
    )
    pdf.ln(4)

    # ── Score Section ──────────────────────────────────────────────────
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(30, 40, 60)
    pdf.cell(epw, 8, "1. Match Assessment", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "B", 24)
    if score >= 8:
        pdf.set_text_color(0, 160, 90)
        interp = "Strong Match — Candidate aligns closely with JD requirements"
    elif score >= 5:
        pdf.set_text_color(220, 140, 0)
        interp = "Partial Match — Critical gaps need targeted resume adjustments"
    else:
        pdf.set_text_color(220, 40, 60)
        interp = "Low Match — Significant technical & domain gaps identified"

    pdf.cell(epw, 12, f"Score: {score}/10", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(80, 80, 90)
    pdf.cell(epw, 6, interp, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Keyword Match Section ──────────────────────────────────────────
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(30, 40, 60)
    pdf.cell(epw, 8, "2. JD Keyword Match (ATS Simulation)", new_x="LMARGIN", new_y="NEXT")

    if keywords:
        found_count = sum(1 for kw in keywords if kw.get("found_in_resume"))
        match_pct = (found_count / len(keywords)) * 100
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(60, 60, 70)
        pdf.cell(
            epw,
            6,
            f"Coverage: {found_count} of {len(keywords)} keywords present ({match_pct:.1f}%)",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(2)

        # Table Header
        col1 = epw * 0.50
        col2 = epw * 0.25
        col3 = epw * 0.25

        pdf.set_font("helvetica", "B", 9)
        pdf.set_fill_color(240, 243, 248)
        pdf.set_text_color(40, 40, 60)
        pdf.cell(col1, 7, "  Target Keyword / Skill", border=1, fill=True)
        pdf.cell(col2, 7, "  Category", border=1, fill=True)
        pdf.cell(col3, 7, "  Resume Match", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 9)
        for kw in keywords[:15]:
            is_found = bool(kw.get("found_in_resume"))
            status = "[FOUND]" if is_found else "[MISSING]"
            if is_found:
                pdf.set_text_color(0, 130, 60)
            else:
                pdf.set_text_color(180, 30, 40)

            pdf.cell(col1, 6, f"  {safe(kw.get('keyword', ''))}", border=1)
            pdf.cell(col2, 6, f"  {safe(kw.get('category', 'skill'))}", border=1)
            pdf.cell(col3, 6, f"  {status}", border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(50, 50, 60)
    else:
        pdf.set_font("helvetica", "", 10)
        pdf.cell(epw, 6, "No keywords analyzed.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Gap Analysis Section ───────────────────────────────────────────
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(30, 40, 60)
    pdf.cell(epw, 8, "3. Identified Skill Gaps", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 60)
    for i, gap in enumerate(gaps, 1):
        pdf.multi_cell(epw, 6, f"{i}. {safe(gap)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.ln(4)

    # ── Improvements Section ───────────────────────────────────────────
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(30, 40, 60)
    pdf.cell(epw, 8, "4. Recommended Actionable Improvements", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 60)
    for i, imp in enumerate(improvements, 1):
        pdf.multi_cell(epw, 6, f"{i}. {safe(imp)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.ln(4)

    # ── Preparation Section ────────────────────────────────────────────
    pdf.set_font("helvetica", "B", 13)
    pdf.set_text_color(30, 40, 60)
    pdf.cell(epw, 8, "5. Interview Preparation Roadmap", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(50, 50, 60)
    for i, prep in enumerate(preparation, 1):
        pdf.multi_cell(epw, 6, f"Step {i}: {safe(prep)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.ln(4)

    # Return pure bytes
    return bytes(pdf.output())
