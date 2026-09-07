from __future__ import annotations

import json
import re
import time
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic Resume Analyzer — AI Career Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ultra-Modern CSS Design System (Aurora + Cyberpunk Glassmorphism)
# ---------------------------------------------------------------------------
STYLES = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap" rel="stylesheet">

<style>
/* =====================================================
   0. RESET STREAMLIT CHROME & GUARANTEE SIDEBAR TOGGLE
   ===================================================== */
[data-testid="stHeader"] {
  background: transparent !important;
  z-index: 99999 !important;
}
[data-testid="stToolbar"] {
  display: none !important;
}
[data-testid="stDecoration"] {
  display: none !important;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* GUARANTEED SIDEBAR COLLAPSE / EXPAND TOGGLE */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapsedControl"],
button[aria-label="Expand sidebar"],
button[aria-label="Collapse sidebar"],
button[data-testid="baseButton-headerNoPadding"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  z-index: 9999999 !important;
  position: fixed !important;
  top: 14px !important;
  left: 14px !important;
  background: rgba(10, 15, 33, 0.95) !important;
  border: 1.5px solid #00FFA3 !important;
  border-radius: 10px !important;
  padding: 6px 10px !important;
  cursor: pointer !important;
  box-shadow: 0 0 20px rgba(0, 255, 163, 0.35), 0 4px 14px rgba(0, 0, 0, 0.6) !important;
  backdrop-filter: blur(16px) !important;
  pointer-events: auto !important;
  transition: all 0.25s ease !important;
}

[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover,
button[aria-label="Expand sidebar"]:hover {
  background: rgba(0, 255, 163, 0.18) !important;
  border-color: #00FFA3 !important;
  transform: scale(1.08) !important;
  box-shadow: 0 0 30px rgba(0, 255, 163, 0.5) !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg,
button[aria-label="Expand sidebar"] svg,
button[aria-label="Collapse sidebar"] svg {
  fill: #00FFA3 !important;
  color: #00FFA3 !important;
  stroke: #00FFA3 !important;
  width: 20px !important;
  height: 20px !important;
}

/* =====================================================
   1. DESIGN TOKENS
   ===================================================== */
:root {
  --bg1:      #060914;
  --bg2:      #0A0F22;
  --bg3:      #0E1630;

  --mint:     #00FFA3;
  --mint-dk:  #00CC84;
  --rose:     #FF4060;
  --violet:   #B44DFF;
  --amber:    #FFB800;
  --sky:      #38BDF8;

  --text:     #EDF2FF;
  --muted:    #8899B4;
  --dimmed:   #506282;

  --radius:    18px;
  --radius-sm: 12px;

  --font: 'Inter', system-ui, -apple-system, sans-serif;
  --mono: 'JetBrains Mono', monospace;
}

/* =====================================================
   2. ANIMATED AURORA CANVAS BACKGROUND
   ===================================================== */
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
  background: var(--bg1) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
  overflow-x: hidden !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  background:
    radial-gradient(ellipse 80% 55% at 5% 15%,   rgba(0, 255, 163, 0.08) 0%, transparent 60%),
    radial-gradient(ellipse 65% 70% at 85% 85%,  rgba(180, 77, 255, 0.09) 0%, transparent 55%),
    radial-gradient(ellipse 55% 45% at 50% 45%,  rgba(255, 64, 96, 0.05)  0%, transparent 50%);
  animation: auroraDrift 22s ease-in-out infinite alternate;
  pointer-events: none;
}

@keyframes auroraDrift {
  0%   { transform: translate(0, 0) scale(1); }
  50%  { transform: translate(-30px, 20px) scale(1.05); }
  100% { transform: translate(20px, -15px) scale(1.02); }
}

.block-container {
  padding-top: 1.2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1420px !important;
  position: relative;
  z-index: 1;
}

/* =====================================================
   3. HERO BANNER
   ===================================================== */
.hero-container {
  text-align: center;
  margin-bottom: 1.8rem;
  padding: 1.2rem 1rem 0.5rem;
}
.hero-title {
  font-size: 3rem;
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1.1;
  margin: 0;
  background: linear-gradient(135deg, #00FFA3 0%, #38BDF8 35%, #B44DFF 70%, #FF4060 100%);
  background-size: 250% 250%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 6s ease-in-out infinite;
}
@keyframes shimmer {
  0%, 100% { background-position: 0% 50%; }
  50%      { background-position: 100% 50%; }
}
.hero-subtitle {
  margin-top: 0.6rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  color: var(--muted);
  font-size: 0.92rem;
  font-weight: 500;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  background: rgba(0, 255, 163, 0.12);
  color: var(--mint);
  border: 1px solid rgba(0, 255, 163, 0.35);
  box-shadow: 0 0 12px rgba(0, 255, 163, 0.2);
}

/* =====================================================
   4. GLASS CONTAINER CARD STYLES
   ===================================================== */
.card-box {
  background: linear-gradient(180deg, rgba(13, 20, 42, 0.82) 0%, rgba(8, 12, 28, 0.92) 100%);
  border: 1.5px solid rgba(0, 255, 163, 0.22);
  border-radius: var(--radius);
  padding: 24px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  margin-bottom: 20px;
  transition: all 0.3s ease;
}
.card-box:hover {
  border-color: rgba(0, 255, 163, 0.40);
  box-shadow: 0 12px 50px rgba(0, 255, 163, 0.10), inset 0 1px 0 rgba(255, 255, 255, 0.12);
}

.card-header-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.card-header-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
  font-size: 1.05rem;
  color: #FFFFFF;
  letter-spacing: 0.01em;
}
.card-header-badge {
  font-family: var(--mono);
  font-size: 0.70rem;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 6px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.badge-mint {
  background: rgba(0, 255, 163, 0.12);
  color: var(--mint);
  border: 1px solid rgba(0, 255, 163, 0.3);
}
.badge-sky {
  background: rgba(56, 189, 248, 0.12);
  color: var(--sky);
  border: 1px solid rgba(56, 189, 248, 0.3);
}
.card-subtext {
  font-size: 0.84rem;
  color: var(--muted);
  margin-bottom: 12px;
  line-height: 1.5;
}

/* =====================================================
   5. TEXTAREA & UPLOADER OVERHAUL (VIBRANT & RICH)
   ===================================================== */
[data-testid="stTextArea"] > div > div {
  background: linear-gradient(180deg, rgba(14, 22, 46, 0.95) 0%, rgba(9, 14, 30, 0.98) 100%) !important;
  border: 1.5px solid rgba(0, 255, 163, 0.28) !important;
  border-radius: 14px !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
  transition: all 0.28s ease !important;
  padding: 4px !important;
}

[data-testid="stTextArea"] > div > div:focus-within {
  border-color: #00FFA3 !important;
  box-shadow: 0 0 0 3px rgba(0, 255, 163, 0.20), 0 0 30px rgba(0, 255, 163, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
  background: rgba(16, 26, 56, 0.98) !important;
}

[data-testid="stTextArea"] textarea {
  color: #FFFFFF !important;
  font-family: var(--font) !important;
  font-size: 0.94rem !important;
  font-weight: 500 !important;
  line-height: 1.65 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 12px 14px !important;
}

[data-testid="stTextArea"] textarea::placeholder {
  color: #7E94B8 !important;
  font-size: 0.88rem !important;
  font-style: italic !important;
  opacity: 0.85 !important;
}

/* UPLOADER */
[data-testid="stFileUploaderDropzone"] {
  background: linear-gradient(135deg, rgba(0, 255, 163, 0.04) 0%, rgba(180, 77, 255, 0.03) 100%) !important;
  border: 2px dashed rgba(0, 255, 163, 0.38) !important;
  border-radius: 14px !important;
  padding: 1.4rem !important;
  transition: all 0.3s ease !important;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
  background: rgba(0, 255, 163, 0.08) !important;
  border-color: #00FFA3 !important;
  box-shadow: 0 0 25px rgba(0, 255, 163, 0.22) !important;
  transform: translateY(-2px);
}

/* =====================================================
   6. PRIMARY CALL-TO-ACTION BUTTON
   ===================================================== */
div.stButton > button[kind="primary"] {
  width: 100% !important;
  border-radius: 14px !important;
  font-family: var(--font) !important;
  font-weight: 900 !important;
  font-size: 1.05rem !important;
  letter-spacing: 0.03em !important;
  padding: 0.95rem 1.8rem !important;
  border: none !important;
  color: #040814 !important;
  background: linear-gradient(135deg, #00FFA3 0%, #00D68A 50%, #00FFA3 100%) !important;
  background-size: 200% auto !important;
  box-shadow: 0 0 28px rgba(0, 255, 163, 0.45), 0 4px 18px rgba(0, 0, 0, 0.5) !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  text-transform: uppercase !important;
}

div.stButton > button[kind="primary"]:hover {
  background-position: right center !important;
  transform: translateY(-2px) scale(1.01) !important;
  box-shadow: 0 0 40px rgba(0, 255, 163, 0.65), 0 8px 25px rgba(0, 0, 0, 0.6) !important;
  color: #02050D !important;
}

/* =====================================================
   7. ONBOARDING & HOW IT WORKS (RIGHT COLUMN)
   ===================================================== */
.ob-title {
  font-size: 1.25rem;
  font-weight: 900;
  color: #FFFFFF;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}
.ob-step {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 16px;
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.025);
  border: 1px solid rgba(255, 255, 255, 0.07);
  margin-bottom: 12px;
  transition: all 0.25s ease;
}
.ob-step:hover {
  background: rgba(0, 255, 163, 0.04);
  border-color: rgba(0, 255, 163, 0.25);
  transform: translateX(4px);
}
.ob-num {
  width: 38px;
  height: 38px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  font-size: 0.90rem;
  flex-shrink: 0;
  color: #040814;
}
.ob-num-1 { background: linear-gradient(135deg, #00FFA3, #00CC84); box-shadow: 0 0 15px rgba(0, 255, 163, 0.35); }
.ob-num-2 { background: linear-gradient(135deg, #B44DFF, #8A2BE2); box-shadow: 0 0 15px rgba(180, 77, 255, 0.35); }
.ob-num-3 { background: linear-gradient(135deg, #FF4060, #CC2040); box-shadow: 0 0 15px rgba(255, 64, 96, 0.35); }

.ob-step-title {
  font-weight: 800;
  color: #FFFFFF;
  font-size: 0.95rem;
  margin-bottom: 4px;
}
.ob-step-desc {
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.5;
}

.feature-pill-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 18px 0 14px;
}
.feature-pill {
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 700;
  font-family: var(--mono);
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid rgba(255, 255, 255, 0.09);
  color: #E2E8F0;
}

.ob-tip-box {
  margin-top: 16px;
  padding: 14px 18px;
  border-radius: var(--radius-sm);
  background: linear-gradient(135deg, rgba(255, 184, 0, 0.06), rgba(0, 255, 163, 0.03));
  border: 1px solid rgba(255, 184, 0, 0.25);
  border-left: 4px solid var(--amber);
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.ob-tip-box strong {
  color: var(--amber);
}

/* =====================================================
   8. SCORE GAUGE & DASHBOARD STATS
   ===================================================== */
.score-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 10px 0 20px;
}
.score-ring-wrap {
  position: relative;
  width: 170px;
  height: 170px;
}
.score-ring-wrap svg {
  transform: rotate(-90deg);
  width: 170px;
  height: 170px;
}
.score-ring-bg {
  fill: none;
  stroke: rgba(255, 255, 255, 0.06);
  stroke-width: 11;
}
.score-ring-fill {
  fill: none;
  stroke-width: 11;
  stroke-linecap: round;
  transition: stroke-dashoffset 1.6s cubic-bezier(0.4, 0, 0.2, 1);
  filter: drop-shadow(0 0 10px var(--ring-color));
}
.score-center {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.score-value {
  font-family: var(--mono);
  font-weight: 900;
  font-size: 2.8rem;
  line-height: 1;
}
.score-label {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 4px;
}
.score-elapsed {
  margin-top: 8px;
  font-size: 0.78rem;
  color: var(--dimmed);
  font-family: var(--mono);
}

.stats-row {
  display: flex;
  gap: 12px;
  margin: 16px 0 22px;
}
.stat-card {
  flex: 1;
  padding: 14px 16px;
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.025);
  border: 1px solid rgba(255, 255, 255, 0.08);
  text-align: center;
}
.stat-value {
  font-family: var(--mono);
  font-weight: 900;
  font-size: 1.4rem;
}
.stat-label {
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 4px;
}

/* =====================================================
   9. RESULT TABS & CARDS
   ===================================================== */
[data-baseweb="tab-list"] {
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 14px !important;
  padding: 5px !important;
  background: rgba(255, 255, 255, 0.02) !important;
  gap: 6px !important;
}
[data-baseweb="tab"] {
  color: var(--muted) !important;
  background: transparent !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  font-size: 0.86rem !important;
  padding: 8px 16px !important;
  transition: all 0.25s ease !important;
}
[data-baseweb="tab"][aria-selected="true"] {
  color: #040814 !important;
  background: var(--mint) !important;
  box-shadow: 0 0 15px rgba(0, 255, 163, 0.4) !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

.tab-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.tab-header-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
}
.tab-header-icon.gaps { background: rgba(255, 64, 96, 0.12); border: 1px solid rgba(255, 64, 96, 0.25); }
.tab-header-icon.imps { background: rgba(0, 255, 163, 0.12); border: 1px solid rgba(0, 255, 163, 0.25); }
.tab-header-icon.prep { background: rgba(180, 77, 255, 0.12); border: 1px solid rgba(180, 77, 255, 0.25); }
.tab-header-title { font-weight: 800; font-size: 1.05rem; color: #FFFFFF; }
.tab-header-count {
  font-family: var(--mono);
  font-size: 0.75rem;
  font-weight: 800;
  padding: 2px 10px;
  border-radius: 6px;
  color: var(--mint);
  background: rgba(0, 255, 163, 0.1);
  border: 1px solid rgba(0, 255, 163, 0.2);
}

.result-scroll { max-height: 540px; overflow-y: auto; padding: 4px; }
.result-scroll::-webkit-scrollbar { width: 5px; }
.result-scroll::-webkit-scrollbar-thumb { background: rgba(0, 255, 163, 0.25); border-radius: 999px; }

/* GAP CARD */
.gap-card {
  padding: 14px 18px;
  border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-left: 4px solid;
  margin-bottom: 12px;
  transition: all 0.25s ease;
}
.gap-card:hover {
  background: rgba(255, 255, 255, 0.035);
  transform: translateX(4px);
}
.gap-card.severity-high { border-left-color: #FF4060; }
.gap-card.severity-med  { border-left-color: #FFB800; }
.gap-card.severity-low  { border-left-color: #00FFA3; }
.gap-header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.gap-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 7px; font-weight: 900; font-size: 0.75rem;
}
.gap-badge-high { background: rgba(255, 64, 96, 0.15); color: #FF4060; border: 1px solid rgba(255, 64, 96, 0.3); }
.gap-badge-med  { background: rgba(255, 184, 0, 0.15); color: #FFB800; border: 1px solid rgba(255, 184, 0, 0.3); }
.gap-badge-low  { background: rgba(0, 255, 163, 0.15); color: #00FFA3; border: 1px solid rgba(0, 255, 163, 0.3); }
.gap-skill-name { font-weight: 800; font-size: 0.98rem; color: #FFFFFF; }
.gap-body { color: var(--muted); font-size: 0.88rem; line-height: 1.55; padding-left: 36px; }
.gap-jd-req {
  margin-top: 8px; padding: 7px 12px; border-radius: 7px;
  background: rgba(0, 255, 163, 0.05); border: 1px solid rgba(0, 255, 163, 0.18);
  font-size: 0.82rem; color: #A8FFD8;
}

/* IMPROVEMENT CARD */
.imp-card {
  display: flex; gap: 14px; align-items: flex-start; padding: 14px 18px;
  border-radius: var(--radius-sm); background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06); margin-bottom: 12px;
  transition: all 0.25s ease;
}
.imp-card:hover {
  background: rgba(0, 255, 163, 0.03); border-color: rgba(0, 255, 163, 0.2);
  transform: translateX(4px);
}
.imp-check {
  width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center;
  background: rgba(0, 255, 163, 0.10); border: 1px solid rgba(0, 255, 163, 0.3);
  color: var(--mint); font-weight: 900; font-size: 0.85rem; flex-shrink: 0;
}
.imp-num { font-weight: 900; font-size: 0.78rem; color: var(--mint); font-family: var(--mono); }
.imp-text { color: #FFFFFF; font-size: 0.92rem; font-weight: 600; line-height: 1.5; margin: 4px 0; }

/* PREPARATION TIMELINE */
.prep-card {
  padding: 14px 18px; border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.06);
  margin-bottom: 14px; transition: all 0.25s ease;
}
.prep-card:hover {
  background: rgba(180, 77, 255, 0.04); border-color: rgba(180, 77, 255, 0.25);
  transform: translateX(4px);
}
.prep-num {
  font-family: var(--mono); font-weight: 800; font-size: 0.75rem; color: var(--violet);
  padding: 3px 9px; border-radius: 6px; background: rgba(180, 77, 255, 0.12);
  border: 1px solid rgba(180, 77, 255, 0.25);
}
.prep-gap-title { font-weight: 800; font-size: 0.95rem; color: #E4D5FF; }
.prep-sec {
  display: flex; flex-direction: column; gap: 3px; padding: 9px 12px;
  border-radius: 7px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05);
  margin-top: 6px;
}
.prep-tag {
  display: inline-block; font-size: 0.70rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: 0.06em; padding: 2px 7px; border-radius: 4px; width: fit-content;
}
.tag-study    { background: rgba(56, 189, 248, 0.14); color: #38BDF8; }
.tag-practice { background: rgba(0, 255, 163, 0.12); color: #00FFA3; }
.tag-angle    { background: rgba(255, 184, 0, 0.14); color: #FFB800; }
.prep-sec-text { font-size: 0.88rem; color: var(--muted); line-height: 1.55; }

/* KEYWORD HEATMAP */
.keyword-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.kw-chip {
  padding: 6px 14px; border-radius: 20px; font-size: 0.80rem; font-weight: 700;
  font-family: var(--mono); transition: all 0.2s ease;
}
.kw-found {
  background: rgba(0, 255, 163, 0.14); color: #00FFA3; border: 1.5px solid rgba(0, 255, 163, 0.35);
}
.kw-missing {
  background: rgba(255, 64, 96, 0.12); color: #FF4060; border: 1.5px solid rgba(255, 64, 96, 0.25);
}

/* HISTORY CARDS */
.history-card {
  padding: 16px 20px; border-radius: var(--radius-sm);
  background: rgba(255, 255, 255, 0.025); border: 1px solid rgba(255, 255, 255, 0.07);
  margin-bottom: 12px; transition: all 0.25s;
}
.history-card:hover {
  background: rgba(0, 255, 163, 0.035); border-color: rgba(0, 255, 163, 0.25);
  transform: translateX(4px);
}
.history-score { font-family: var(--mono); font-weight: 900; font-size: 1.6rem; }
.history-filename { color: #FFFFFF; font-weight: 700; font-size: 0.96rem; margin-bottom: 3px; }
.history-meta { color: var(--muted); font-size: 0.82rem; }

/* SIDEBAR STYLING */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #090E20 0%, #050814 100%) !important;
  border-right: 1px solid rgba(0, 255, 163, 0.12) !important;
}

/* LOADING CONTAINER */
.loading-container { display: flex; flex-direction: column; align-items: center; padding: 3.5rem 1rem; }
.loading-spinner {
  width: 60px; height: 60px; border-radius: 50%;
  border: 3px solid rgba(0, 255, 163, 0.1);
  border-top-color: var(--mint); border-right-color: var(--violet);
  animation: spin 0.9s linear infinite; margin-bottom: 1.5rem;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text { font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 6px; }
.loading-sub  { font-size: 0.82rem; color: var(--muted); font-family: var(--mono); }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "analysis"
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "out" not in st.session_state:
    st.session_state.out = None
if "error" not in st.session_state:
    st.session_state.error = None
if "active_jd_input" not in st.session_state:
    st.session_state.active_jd_input = ""

# ---------------------------------------------------------------------------
# Sidebar (Always Re-openable & Informative)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 18px 0 10px;">
          <div style="font-size: 1.65rem; font-weight: 900; background: linear-gradient(135deg, #00FFA3, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🤖 Resume AI
          </div>
          <div style="color: #8899B4; font-size: 0.78rem; margin-top: 4px; font-weight: 600;">
            Agentic Resume & JD Matcher
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<div style='font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; color: #506282; font-weight: 800; margin-bottom: 8px;'>Navigation</div>", unsafe_allow_html=True)
    if st.button("🔬  New Resume Analysis", use_container_width=True, type="primary" if st.session_state.page == "analysis" else "secondary"):
        st.session_state.page = "analysis"
        st.rerun()

    if st.button("📊  Analysis History & Trends", use_container_width=True, type="primary" if st.session_state.page == "history" else "secondary"):
        st.session_state.page = "history"
        st.rerun()

    st.divider()

    # System Status Indicator & Model Selector
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=2.5).json()
        provider = health.get("llm_provider", "ollama").lower()
        if provider == "groq":
            status_color = "#00FFA3" if health.get("groq_configured") else "#FFB800"
            status_text = "Groq Cloud API (Connected)" if health.get("groq_configured") else "Groq API (No Key)"
            provider_label = "Cloud AI (Groq)"
        else:
            status_color = "#00FFA3" if health.get("ollama_reachable") else "#FF4060"
            status_text = "Ollama (Connected)" if health.get("ollama_reachable") else "Ollama Offline"
            provider_label = "Local AI (Ollama)"

        models_list = health.get("available_models") or ["qwen2.5:3b", "qwen3.5:9b", "qwen2.5:7b", "qwen3:8b", "llama3.1:8b"]
        current_model = health.get("model", "qwen2.5:3b")
        default_idx = models_list.index(current_model) if current_model in models_list else 0

        st.markdown(
            f"""
            <div style="padding: 12px; border-radius: 10px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); margin-bottom: 12px;">
              <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:8px; height:8px; border-radius:50%; background:{status_color}; box-shadow: 0 0 10px {status_color};"></div>
                <span style="color: #EDF2FF; font-size: 0.82rem; font-weight: 700;">{provider_label}: {status_text}</span>
              </div>
              <div style="color: #8899B4; font-size: 0.74rem; margin-top: 6px; font-family: var(--mono);">
                Model: {current_model}
              </div>
              <div style="color: #506282; font-size: 0.70rem; margin-top: 4px;">
                Retrieval: Hybrid ChromaDB + BM25
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if provider == "ollama":
            selected_model = st.selectbox(
                "🧠 Select Ollama Model",
                options=models_list,
                index=default_idx,
                help="Swap LLM models easily: qwen2.5:3b, qwen3.5:9b, qwen2.5:7b, qwen3:8b, or llama3.1:8b",
                key="ollama_model_select",
            )
            st.session_state.selected_model = selected_model
        else:
            st.session_state.selected_model = current_model
    except Exception:
        st.markdown(
            """
            <div style="padding: 12px; border-radius: 10px; background: rgba(255,64,96,0.05); border: 1px solid rgba(255,64,96,0.2);">
              <div style="color: #FF4060; font-size: 0.82rem; font-weight: 700;">⚠️ Backend Server Not Detected</div>
              <div style="color: #8899B4; font-size: 0.72rem; margin-top: 4px;">Run `uvicorn backend.server:app --port 8000`</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# View 1: New Resume Analysis
# ---------------------------------------------------------------------------
def render_analysis_page():
    # Hero Title
    st.markdown(
        """
        <div class="hero-container">
          <h1 class="hero-title">Agentic Resume Analyzer</h1>
          <div class="hero-subtitle">
            <span class="hero-badge">⚡ 100% PRIVATE & LOCAL</span>
            Ollama · LangGraph · ChromaDB RAG · sentence-transformers
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Navigation Switcher on Page
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav2:
        nav_mode = st.radio(
            "Page Switcher",
            ["🔬 New Resume Analysis", "📊 History & Analytics"],
            horizontal=True,
            label_visibility="collapsed",
            index=0 if st.session_state.page == "analysis" else 1,
            key="top_nav_radio",
        )
        if nav_mode == "📊 History & Analytics" and st.session_state.page != "history":
            st.session_state.page = "history"
            st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 3], gap="large")

    # ── Left Column: Inputs ──────────────────────────────────────────────
    with col_left:
        # Step 1: Job Description Card Header
        st.markdown(
            """
            <div class="card-header-bar">
              <div class="card-header-title">
                <span style="font-size: 1.25rem;">📋</span> Target Job Description
              </div>
              <span class="card-header-badge badge-mint">STEP 1 OF 2</span>
            </div>
            <div class="card-subtext">
              Paste the full job requirements, technical skills, and core responsibilities for deep semantic matching.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick sample JD filler & clear controls
        col_jd_act1, col_jd_act2 = st.columns([3, 1])
        with col_jd_act1:
            if st.button("✨ Load Sample AI Engineer JD", use_container_width=True):
                st.session_state["active_jd_input"] = (
                    "Role: Senior AI / MLOps Engineer\n"
                    "Requirements:\n"
                    "- 3+ years experience with Python, FastAPI, and asynchronous backend microservices.\n"
                    "- Hands-on production experience building RAG pipelines using LangChain, LangGraph, and Vector Databases (ChromaDB, Pinecone, FAISS).\n"
                    "- Experience fine-tuning and deploying open-source LLMs (Llama, Qwen, Mistral) on AWS SageMaker or GCP.\n"
                    "- Strong software engineering fundamentals: automated testing (pytest), Docker containerization, and CI/CD automation.\n"
                    "- Strong communication skills and cross-functional leadership."
                )
                st.rerun()
        with col_jd_act2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["active_jd_input"] = ""
                st.rerun()

        job_description = st.text_area(
            "Job Description",
            key="active_jd_input",
            height=240,
            placeholder="Paste the target Job Description here (skills, tools, responsibilities, years of experience)...",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Step 2: Resume PDF Card Header
        st.markdown(
            """
            <div class="card-header-bar">
              <div class="card-header-title">
                <span style="font-size: 1.25rem;">📄</span> Candidate Resume (PDF)
              </div>
              <span class="card-header-badge badge-sky">STEP 2 OF 2</span>
            </div>
            <div class="card-subtext">
              Upload your PDF resume. Text is chunked, embedded, and analyzed strictly on your local machine.
            </div>
            """,
            unsafe_allow_html=True,
        )

        resume_file = st.file_uploader(
            "Resume PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help="Upload your resume in PDF format. 100% on-device processing.",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        submitted = st.button("🚀  Run Agentic Resume Analysis", type="primary", use_container_width=True)

    # ── Right Column: Results & Dashboard ────────────────────────────────
    with col_right:
        if submitted:
            st.session_state.submitted = True
            st.session_state.out = None
            st.session_state.error = None

        # Onboarding state when no analysis is active
        if not st.session_state.submitted:
            st.markdown(
                """
                <div class="card-box">
                  <div class="ob-title">
                    <span>✨</span> Multi-Agent Resume Matching Engine
                  </div>
                  <div style="color: #8899B4; font-size: 0.88rem; line-height: 1.55; margin-bottom: 20px;">
                    This application deploys a local <strong>LangGraph state machine</strong> to perform deep multi-query RAG retrieval against your resume, eliminating keyword hallucinations and generating concrete, actionable career guidance.
                  </div>

                  <div class="ob-step">
                    <div class="ob-num ob-num-1">01</div>
                    <div>
                      <div class="ob-step-title">Role Target Definition</div>
                      <div class="ob-step-desc">Extracts core technical proficiencies, required tooling, and expected seniority benchmarks from the target job posting.</div>
                    </div>
                  </div>

                  <div class="ob-step">
                    <div class="ob-num ob-num-2">02</div>
                    <div>
                      <div class="ob-step-title">Neural Resume Ingestion (RAG)</div>
                      <div class="ob-step-desc">Parses your PDF and embeds chunks using MiniLM embeddings into an in-memory ChromaDB vector store with reciprocal rank fusion (RRF).</div>
                    </div>
                  </div>

                  <div class="ob-step">
                    <div class="ob-num ob-num-3">03</div>
                    <div>
                      <div class="ob-step-title">Deep Gap Analysis & Action Directives</div>
                      <div class="ob-step-desc">Evaluates fit, generates past-tense bullet additions, surfaces interview prep angles, and outputs an executive PDF report.</div>
                    </div>
                  </div>

                  <div style="font-size: 0.80rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: #8899B4; margin-top: 20px;">
                    Included Analysis Deliverables
                  </div>
                  <div class="feature-pill-grid">
                    <span class="feature-pill">🎯 0–10 Match Score</span>
                    <span class="feature-pill">🔍 Evidence-Linked Skill Gaps</span>
                    <span class="feature-pill">⚡ Action-Oriented Resume Bullets</span>
                    <span class="feature-pill">🗺️ 3-Tier Interview Prep Roadmap</span>
                    <span class="feature-pill">🔑 ATS Keyword Heatmap</span>
                    <span class="feature-pill">📄 1-Click PDF Report Export</span>
                  </div>

                  <div class="ob-tip-box">
                    <div style="font-size: 1.2rem;">💡</div>
                    <div style="font-size: 0.84rem; color: #EDF2FF; line-height: 1.5;">
                      <strong>Pro tip:</strong> Paste detailed job descriptions specifying real frameworks, platforms, and years of experience. The neural agent cross-checks every single claim against your resume.
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Input Validation — guarantees active UI state is read
        active_jd = st.session_state.get("active_jd_input", "").strip()
        if not active_jd:
            st.error("⚠️ Please paste a Job Description before running the analysis.")
            st.session_state.error = "Missing job description."
        elif resume_file is None:
            st.error("⚠️ Please upload a Resume PDF file before running the analysis.")
            st.session_state.error = "Missing resume PDF."
        else:
            STAGE_LABELS = {
                "started":        ("⚙️", "Initializing LangGraph agent pipeline…"),
                "extracting":     ("📄", "Extracting PDF text & building vector embeddings…"),
                "retrieved":      ("🔍", "Multi-query RAG semantic search complete"),
                "analyzing_gaps": ("🧠", "Evaluating candidate experience against JD requirements…"),
                "gaps_found":     ("📋", "Skill gaps classified with JD evidence"),
                "scoring":        ("📊", "Synthesizing match score, improvements & prep roadmap…"),
                "complete":       ("✅", "Analysis successfully completed!"),
                "error":          ("❌", "Pipeline execution error"),
            }

            def _loading_html(stage_key: str, message: str) -> str:
                icon, _ = STAGE_LABELS.get(stage_key, ("⚙️", message))
                all_stages = [
                    ("extracting",     "Parsing PDF & Vector Store"),
                    ("retrieved",      "Multi-query RAG Retrieval"),
                    ("analyzing_gaps", "Candidate Gap Identification"),
                    ("scoring",        "Synthesizing Strategic Score & Roadmap"),
                ]
                stage_keys = [s[0] for s in all_stages]
                current_idx = stage_keys.index(stage_key) if stage_key in stage_keys else -1
                stages_html = ""
                for i, (skey, slabel) in enumerate(all_stages):
                    if i < current_idx:
                        dot_style, text_style = "background:#00FFA3; box-shadow: 0 0 8px #00FFA3;", "color:#00FFA3;"
                    elif i == current_idx:
                        dot_style, text_style = "background:#38BDF8; box-shadow: 0 0 10px #38BDF8; animation:pulse 1s infinite alternate;", "color:#FFFFFF; font-weight:700;"
                    else:
                        dot_style, text_style = "background:rgba(255,255,255,0.12);", "color:#506282;"
                    stages_html += f'<div style="display:flex; align-items:center; gap:10px; font-size:0.82rem; padding:8px 12px; border-radius:8px; background:rgba(255,255,255,0.02); margin-bottom:6px;"><div style="width:8px; height:8px; border-radius:50%; {dot_style}"></div><span style="{text_style}">{slabel}</span></div>'

                return f"""
                <div class="card-box" style="text-align:center;">
                  <div class="loading-spinner" style="margin: 0 auto 18px;"></div>
                  <div class="loading-text">{icon} {message}</div>
                  <div class="loading-sub">Running 100% locally on Ollama — Zero cloud telemetry</div>
                  <div style="max-width: 360px; margin: 20px auto 0; text-align:left;">{stages_html}</div>
                </div>
                """

            loading_placeholder = st.empty()
            if not st.session_state.out and not st.session_state.error:
                loading_placeholder.markdown(_loading_html("started", "Starting analysis pipeline…"), unsafe_allow_html=True)

            t0 = time.perf_counter()
            out = st.session_state.out
            error_msg = st.session_state.error
            elapsed = 0.0

            # Execute SSE Stream with verified active JD payload
            if not out and not error_msg:
                try:
                    resume_bytes = resume_file.getvalue()
                    files = {"resume_pdf": (resume_file.name, resume_bytes, "application/pdf")}
                    data = {
                        "job_description": active_jd,
                        "model": st.session_state.get("selected_model", "qwen2.5:3b"),
                    }

                    with requests.post(f"{BACKEND_URL}/analyze/stream", data=data, files=files, stream=True, timeout=420) as resp:
                        resp.raise_for_status()
                        for raw_line in resp.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                            if not line.startswith("data:"):
                                continue
                            payload_str = line[len("data:"):].strip()
                            try:
                                evt = json.loads(payload_str)
                            except json.JSONDecodeError:
                                continue

                            event_type = evt.get("event", "")
                            event_msg = evt.get("message", "")

                            if event_type == "result":
                                out = evt.get("data", {})
                            elif event_type == "error":
                                error_msg = event_msg
                            else:
                                loading_placeholder.markdown(_loading_html(event_type, event_msg), unsafe_allow_html=True)

                    st.session_state.out = out
                    st.session_state.error = error_msg
                except requests.exceptions.ConnectionError:
                    st.session_state.error = "Cannot reach backend server at http://localhost:8000. Ensure uvicorn is running."
                except requests.exceptions.RequestException as exc:
                    st.session_state.error = f"Analysis request failed: {exc}"
                except Exception as exc:
                    st.session_state.error = f"Unexpected error: {exc}"
                elapsed = time.perf_counter() - t0

            loading_placeholder.empty()

            if st.session_state.error:
                st.error(st.session_state.error)
            elif out:
                score = int(out.get("score", 0) or 0)
                gaps = out.get("gaps", []) or []
                improvements = out.get("improvements", []) or []
                preparation = out.get("preparation", []) or []
                keywords = out.get("keywords", []) or []
                analysis_id = out.get("analysis_id", "")
                elapsed_display = out.get("elapsed_seconds", elapsed)

                keyword_total = len(keywords)
                keyword_found = sum(1 for k in keywords if k.get("found_in_resume"))
                kw_pct = round((keyword_found / keyword_total * 100) if keyword_total > 0 else 0)

                # Color Schemes
                if score >= 8:
                    ring_color, score_fg = "#00FFA3", "#00FFA3"
                    glow_bg = "rgba(0,255,163,0.12)"
                    score_title = "Strong Alignment"
                elif score >= 5:
                    ring_color, score_fg = "#FFB800", "#FFB800"
                    glow_bg = "rgba(255,184,0,0.12)"
                    score_title = "Moderate Match"
                else:
                    ring_color, score_fg = "#FF4060", "#FF4060"
                    glow_bg = "rgba(255,64,96,0.12)"
                    score_title = "Critical Gaps Present"

                radius = 64
                circumference = 2 * 3.14159 * radius
                dash_offset = circumference * (1 - (score / 10))

                # Score Section & Gauge
                st.markdown(
                    f"""
                    <div class="score-section">
                      <div class="score-ring-wrap">
                        <svg viewBox="0 0 170 170">
                          <circle class="score-ring-bg" cx="85" cy="85" r="{radius}" />
                          <circle class="score-ring-fill" cx="85" cy="85" r="{radius}"
                            stroke="{ring_color}"
                            stroke-dasharray="{circumference}"
                            stroke-dashoffset="{dash_offset}"
                            style="--ring-color: {ring_color};"
                          />
                        </svg>
                        <div class="score-center">
                          <div class="score-value" style="color: {score_fg};">{score}<span style="font-size:1.4rem; color: #506282;">/10</span></div>
                          <div class="score-label">{score_title}</div>
                        </div>
                      </div>
                      <div class="score-elapsed">Pipeline completed in {elapsed_display:.1f}s · In-Memory ChromaDB</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Dashboard Quick Stats Row
                st.markdown(
                    f"""
                    <div class="stats-row">
                      <div class="stat-card">
                        <div class="stat-value" style="color: var(--rose);">{len(gaps)}</div>
                        <div class="stat-label">Identified Gaps</div>
                      </div>
                      <div class="stat-card">
                        <div class="stat-value" style="color: var(--mint);">{kw_pct}%</div>
                        <div class="stat-label">ATS Keyword Match</div>
                      </div>
                      <div class="stat-card">
                        <div class="stat-value" style="color: var(--sky);">{len(improvements)}</div>
                        <div class="stat-label">Action Items</div>
                      </div>
                      <div class="stat-card">
                        <div class="stat-value" style="color: var(--violet);">{len(preparation)}</div>
                        <div class="stat-label">Prep Roadmaps</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Tabs Interface
                gaps_tab, imp_tab, prep_tab, kw_tab = st.tabs([
                    "🔍  Skill Gaps",
                    "⚡  Actionable Improvements",
                    "🎯  Interview Prep",
                    "🔑  ATS Keywords",
                ])

                # ── Gaps Tab ──
                with gaps_tab:
                    st.markdown(
                        f'<div class="tab-header"><div class="tab-header-icon gaps">🔍</div><div class="tab-header-title">Identified Technical Gaps</div><div class="tab-header-count">{len(gaps)} found</div></div>',
                        unsafe_allow_html=True,
                    )
                    if gaps:
                        st.markdown('<div class="result-scroll">', unsafe_allow_html=True)
                        for i, g in enumerate(gaps, start=1):
                            sev_class = "severity-high" if score < 5 else ("severity-med" if score < 8 else "severity-low")
                            badge_class = "gap-badge-high" if score < 5 else ("gap-badge-med" if score < 8 else "gap-badge-low")

                            text = g.strip()
                            colon_idx = text.find(":")
                            if 0 < colon_idx < 60:
                                skill_name = text[:colon_idx].strip()
                                explanation = text[colon_idx+1:].strip()
                            else:
                                skill_name = f"Requirement Gap {i}"
                                explanation = text

                            jd_parts = re.split(r"\s*[—-]\s*JD requires:\s*", explanation, flags=re.IGNORECASE)
                            body_text = jd_parts[0].strip()
                            jd_req = jd_parts[1].strip() if len(jd_parts) > 1 and jd_parts[1].strip() else ""
                            jd_html = f'<div class="gap-jd-req"><strong style="color:var(--mint)">📌 JD Benchmark:</strong> {jd_req}</div>' if jd_req else ''

                            st.markdown(
                                f'<div class="gap-card {sev_class}"><div class="gap-header"><div class="gap-badge {badge_class}">{i}</div><div class="gap-skill-name">{skill_name}</div></div><div class="gap-body"><div>{body_text}</div>{jd_html}</div></div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align:center; padding:2rem; color:var(--mint);">✅ No critical skill gaps detected. Strong match!</div>', unsafe_allow_html=True)

                # ── Improvements Tab ──
                with imp_tab:
                    st.markdown(
                        f'<div class="tab-header"><div class="tab-header-icon imps">⚡</div><div class="tab-header-title">Targeted Resume Directives</div><div class="tab-header-count">{len(improvements)} recommendations</div></div>',
                        unsafe_allow_html=True,
                    )
                    if improvements:
                        st.markdown('<div class="result-scroll">', unsafe_allow_html=True)
                        for i, imp in enumerate(improvements, start=1):
                            text = imp.strip()
                            text = re.sub(r"^Add bullet:\s*", "", text, flags=re.IGNORECASE).strip()
                            area_val, action_val, align_val = "", "", ""

                            m_area = re.search(r"(?:Target\s*Area|Area|Target)\s*:\s*(.*?)(?=\s*(?:[|→—]|\bAction\s*Required:|\bAction:|\bJD\s*Alignment:|\bWhy:)|$)", text, re.IGNORECASE)
                            if m_area: area_val = m_area.group(1).strip(" |→—")

                            m_act = re.search(r"(?:Action\s*Required|Action)\s*:\s*(.*?)(?=\s*(?:[|→—]|\bJD\s*Alignment:|\bWhy:)|$)", text, re.IGNORECASE)
                            if m_act: action_val = m_act.group(1).strip(" |→—")

                            m_align = re.search(r"(?:JD\s*Alignment|Alignment|Why)\s*:\s*(.*)$", text, re.IGNORECASE)
                            if m_align: align_val = m_align.group(1).strip(" |→—")

                            if not action_val and not area_val:
                                text_clean = re.sub(r"^Add bullet:\s*", "", text, flags=re.IGNORECASE)
                                parts = re.split(r"\s*[|—-]\s*(?:addresses gap|Why|JD Alignment):\s*", text_clean, flags=re.IGNORECASE)
                                action_val = parts[0].strip()
                                align_val = parts[1].strip() if len(parts) > 1 and parts[1].strip() else ""

                            area_html = f'<div style="font-size: 0.82rem; color: #A8FFD8; font-family: var(--mono); font-weight:700;">📌 Section: {area_val}</div>' if area_val else ''
                            align_html = f'<div class="prep-sec" style="margin-top:6px; border-color:rgba(0,255,163,0.2);"><span class="prep-tag tag-practice">🎯 JD Alignment</span><div class="prep-sec-text">{align_val}</div></div>' if align_val else ''

                            st.markdown(
                                f'<div class="imp-card"><div class="imp-check">✓</div><div style="flex:1;"><div style="display:flex; justify-content:space-between; align-items:center;"><div class="imp-num">ACTION #{i:02d}</div>{area_html}</div><div class="imp-text">{action_val}</div>{align_html}</div></div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown('</div>', unsafe_allow_html=True)

                # ── Preparation Tab ──
                with prep_tab:
                    st.markdown(
                        f'<div class="tab-header"><div class="tab-header-icon prep">🎯</div><div class="tab-header-title">3-Tier Interview Preparation Roadmap</div><div class="tab-header-count">{len(preparation)} steps</div></div>',
                        unsafe_allow_html=True,
                    )
                    if preparation:
                        st.markdown('<div class="result-scroll">', unsafe_allow_html=True)
                        for i, p in enumerate(preparation, start=1):
                            text = p.strip()
                            gap_val, study_val, practice_val, angle_val = "", "", "", ""
                            m_gap = re.search(r"(?:Target\s*Gap|Gap)\s*:\s*(.*?)(?=\s*(?:[|→—]|\bStudy:|\bPractice:|\bInterview [Aa]ngle:)|$)", text, re.IGNORECASE)
                            if m_gap: gap_val = m_gap.group(1).strip(" |→—")

                            m_study = re.search(r"\bStudy\s*:\s*(.*?)(?=\s*(?:[|→—]|\bPractice:|\bInterview [Aa]ngle:)|$)", text, re.IGNORECASE)
                            if m_study: study_val = m_study.group(1).strip(" |→—")

                            m_prac = re.search(r"\bPractice\s*:\s*(.*?)(?=\s*(?:[|→—]|\bInterview [Aa]ngle:)|$)", text, re.IGNORECASE)
                            if m_prac: practice_val = m_prac.group(1).strip(" |→—")

                            m_angle = re.search(r"\bInterview\s*[Aa]ngle\s*:\s*(.*)$", text, re.IGNORECASE)
                            if m_angle: angle_val = m_angle.group(1).strip(" |→—")

                            gap_title_html = f'<div class="prep-gap-title">Target: {gap_val}</div>' if gap_val else ''
                            study_html = f'<div class="prep-sec"><span class="prep-tag tag-study">📖 Conceptual Study</span><div class="prep-sec-text">{study_val}</div></div>' if study_val else ''
                            practice_html = f'<div class="prep-sec"><span class="prep-tag tag-practice">🛠️ Hands-On Project</span><div class="prep-sec-text">{practice_val}</div></div>' if practice_val else ''
                            angle_html = f'<div class="prep-sec"><span class="prep-tag tag-angle">💬 Interview Question</span><div class="prep-sec-text">{angle_val}</div></div>' if angle_val else ''

                            st.markdown(
                                f'<div class="prep-card"><div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;"><div class="prep-num">STEP {i:02d}</div>{gap_title_html}</div>{study_html}{practice_html}{angle_html}</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown('</div>', unsafe_allow_html=True)

                # ── Keywords Tab ──
                with kw_tab:
                    st.markdown(
                        f'<div class="tab-header"><div class="tab-header-icon" style="background:rgba(56,189,248,0.12); color:#38BDF8;">🔑</div><div class="tab-header-title">ATS Keyword Heatmap</div><div class="tab-header-count">{keyword_found}/{keyword_total} matched</div></div>',
                        unsafe_allow_html=True,
                    )
                    found_chips = "".join(f'<span class="kw-chip kw-found">✓ {k["keyword"]}</span>' for k in keywords if k.get("found_in_resume"))
                    missing_chips = "".join(f'<span class="kw-chip kw-missing">✗ {k["keyword"]}</span>' for k in keywords if not k.get("found_in_resume"))

                    if found_chips:
                        st.markdown(f'<div style="color:var(--mint); font-size:0.84rem; font-weight:700; margin-bottom:6px;">✅ Verified Resume Keywords ({keyword_found})</div><div class="keyword-grid">{found_chips}</div>', unsafe_allow_html=True)
                    if missing_chips:
                        st.markdown(f'<div style="color:var(--rose); font-size:0.84rem; font-weight:700; margin-top:16px; margin-bottom:6px;">❌ Missing JD Keywords ({keyword_total - keyword_found})</div><div class="keyword-grid">{missing_chips}</div>', unsafe_allow_html=True)

                # PDF Export Action
                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                if analysis_id:
                    try:
                        pdf_resp = requests.post(f"{BACKEND_URL}/export/pdf", data={"analysis_id": analysis_id}, timeout=30)
                        if pdf_resp.status_code == 200:
                            st.download_button(
                                label="📥  Download Executive PDF Report",
                                data=pdf_resp.content,
                                file_name=f"resume_analysis_{analysis_id}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    except Exception:
                        pass

# ---------------------------------------------------------------------------
# View 2: Analysis History & Analytics
# ---------------------------------------------------------------------------
def render_history_page():
    st.markdown(
        """
        <div class="hero-container">
          <h1 class="hero-title">Analysis History & Trends</h1>
          <div class="hero-subtitle">
            Historical progression and score analytics stored in local SQLite
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Navigation Switcher
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav2:
        nav_mode = st.radio(
            "Page Switcher Hist",
            ["🔬 New Resume Analysis", "📊 History & Analytics"],
            horizontal=True,
            label_visibility="collapsed",
            index=1,
            key="top_nav_hist_radio",
        )
        if nav_mode == "🔬 New Resume Analysis":
            st.session_state.page = "analysis"
            st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    try:
        # Score Trend Chart
        trend_resp = requests.get(f"{BACKEND_URL}/history/trend", timeout=10)
        if trend_resp.status_code == 200:
            trend_data = trend_resp.json()
            if len(trend_data) >= 2:
                df = pd.DataFrame(trend_data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.markdown('<div style="color:#00FFA3; font-size:0.95rem; font-weight:800; margin-bottom:8px;">📈 Match Score Progression Over Time</div>', unsafe_allow_html=True)
                st.line_chart(df.set_index("timestamp")["score"], color="#00FFA3", height=220)

        # History List
        history_resp = requests.get(f"{BACKEND_URL}/history", timeout=10)
        if history_resp.status_code == 200:
            history = history_resp.json()
            if not history:
                st.markdown(
                    """
                    <div class="card-box" style="text-align:center; padding:3rem;">
                      <div style="font-size:2.5rem; margin-bottom:10px;">📋</div>
                      <div style="font-size:1.1rem; font-weight:700; color:#FFFFFF;">No Prior Analyses Found</div>
                      <div style="color:#8899B4; font-size:0.86rem; margin-top:6px;">Run an analysis on the main tab to begin tracking score trends.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<div style="font-weight:800; font-size:1.05rem; color:#FFFFFF; margin-bottom:14px;">Past Analyses ({len(history)})</div>', unsafe_allow_html=True)
                for item in history:
                    score = item.get("score", 0)
                    score_color = "#00FFA3" if score >= 8 else ("#FFB800" if score >= 5 else "#FF4060")
                    ts = item.get("timestamp", "")[:16].replace("T", " ")
                    fname = item.get("resume_filename", "unknown.pdf")
                    jd = item.get("jd_snippet", "")[:120]
                    gaps = item.get("gap_count", 0)
                    kw_pct = item.get("keyword_match_pct", 0)
                    elapsed = item.get("elapsed_seconds", 0)
                    aid = item.get("id", "")

                    col_hist_left, col_hist_right = st.columns([5, 1])
                    with col_hist_left:
                        st.markdown(
                            f"""
                            <div class="history-card">
                              <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                  <div class="history-filename">📄 {fname}</div>
                                  <div class="history-meta">{ts} · {gaps} Gaps Found · {kw_pct}% Keywords · {elapsed:.1f}s</div>
                                  <div class="history-meta" style="margin-top:6px; color:#CAD5E8;">"{jd}..."</div>
                                </div>
                                <div class="history-score" style="color:{score_color};">{score}/10</div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with col_hist_right:
                        if st.button("🗑️ Delete", key=f"del_{aid}", use_container_width=True):
                            requests.delete(f"{BACKEND_URL}/history/{aid}", timeout=5)
                            st.rerun()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend server at http://localhost:8000")
    except Exception as e:
        st.error(f"Error loading history data: {e}")

# ---------------------------------------------------------------------------
# Page Router
# ---------------------------------------------------------------------------
if st.session_state.page == "history":
    render_history_page()
else:
    render_analysis_page()
