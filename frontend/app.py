from __future__ import annotations

import json
import re
import time
import requests
import streamlit as st
import pandas as pd

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic Resume Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# Premium UI — CSS Design System
# ---------------------------------------------------------------------------
STYLES = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">

<style>
/* =====================================================
   0.  RESET STREAMLIT CHROME
   ===================================================== */
[data-testid="stHeader"]     { background:transparent!important; height:0!important; }
[data-testid="stHeader"] *   { display:none!important; }
[data-testid="stDecoration"] { display:none!important; }
#MainMenu { visibility:hidden; }
footer    { visibility:hidden; }

/* =====================================================
   1.  DESIGN TOKENS — UNIQUE PALETTE
   ===================================================== */
:root {
  --bg1:  #060914;
  --bg2:  #0A0F1F;
  --bg3:  #0D1326;

  --glass:   rgba(255,255,255,0.035);
  --glass-2: rgba(255,255,255,0.02);
  --border:  rgba(255,255,255,0.065);

  --text:    #EDF2FF;
  --muted:   #8899B4;
  --dimmed:  #4D5F7A;

  --mint:    #00FFA3;
  --mint-dk: #00CC84;
  --rose:    #FF4060;
  --violet:  #B44DFF;
  --amber:   #FFB800;
  --sky:     #38BDF8;
  --accent:  #00FFA3;

  --radius:    18px;
  --radius-sm: 12px;
  --radius-xs: 8px;

  --font: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  --mono: 'JetBrains Mono', 'Fira Code', monospace;
}

/* =====================================================
   2.  ANIMATED AURORA BACKGROUND
   ===================================================== */
[data-testid="stAppViewContainer"] {
  background: var(--bg1);
  color: var(--text);
  position: relative;
  overflow: hidden;
}
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  background:
    radial-gradient(ellipse 80% 55% at 5% 15%,  rgba(0,255,163,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 70% at 85% 80%,  rgba(180,77,255,0.08) 0%, transparent 55%),
    radial-gradient(ellipse 55% 45% at 50% 45%,  rgba(255,64,96,0.05)  0%, transparent 50%);
  animation: auroraDrift 22s ease-in-out infinite alternate;
  pointer-events: none;
}
@keyframes auroraDrift {
  0%   { transform: translate(0,    0)    scale(1);    }
  50%  { transform: translate(-35px, 25px) scale(1.06); }
  100% { transform: translate(18px, -18px) scale(1.02); }
}

[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed;
  width: 360px;
  height: 360px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,255,163,0.07) 0%, transparent 70%);
  top: 55%;
  left: 65%;
  animation: orbFloat 14s ease-in-out infinite alternate;
  pointer-events: none;
  z-index: 0;
}
@keyframes orbFloat {
  0%   { transform: translate(0, 0) scale(1); }
  100% { transform: translate(-70px, -55px) scale(1.25); }
}

.block-container {
  padding-top: 1rem;
  padding-bottom: 2rem;
  position: relative;
  z-index: 1;
}

/* =====================================================
   3.  TYPOGRAPHY
   ===================================================== */
html, body, [class*="stMarkdown"], [class*="stText"] {
  font-family: var(--font) !important;
}
h1, h2, h3 { color: var(--text) !important; }
p, label, .stMarkdown p { color: var(--muted) !important; }

/* =====================================================
   4.  HERO HEADER
   ===================================================== */
.hero-container {
  text-align: center;
  margin-bottom: 2.5rem;
  padding: 2.2rem 1rem 1.5rem;
}
.hero-title {
  font-size: 2.75rem;
  font-weight: 900;
  letter-spacing: -0.035em;
  line-height: 1.08;
  margin: 0;
  background: linear-gradient(135deg, #00FFA3 0%, #B44DFF 45%, #FF4060 85%, #00FFA3 100%);
  background-size: 220% 220%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: shimmer 5s ease-in-out infinite;
}
@keyframes shimmer {
  0%, 100% { background-position: 0% 50%; }
  50%       { background-position: 100% 50%; }
}
.hero-subtitle {
  margin-top: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 500;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 11px;
  border-radius: 999px;
  font-size: 0.70rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  background: rgba(0,255,163,0.10);
  color: var(--mint);
  border: 1px solid rgba(0,255,163,0.28);
}
.hero-badge::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--mint);
  animation: dotPulse 2s ease-in-out infinite;
}

/* =====================================================
   5.  GLASSMORPHISM CARD
   ===================================================== */
.premium-card {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.055);
  border-radius: var(--radius);
  padding: 26px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
  box-shadow: 0 4px 28px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.04);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}
.premium-card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: var(--radius);
  padding: 1px;
  background: linear-gradient(135deg, rgba(0,255,163,0.12), transparent 50%, rgba(180,77,255,0.08));
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
}
.premium-card:hover {
  border-color: rgba(0,255,163,0.14);
  box-shadow: 0 8px 44px rgba(0,255,163,0.07), inset 0 1px 0 rgba(255,255,255,0.06);
}

.section-label {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 0.92rem;
  color: var(--text);
  margin-bottom: 12px;
  letter-spacing: 0.01em;
}
.section-label .label-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; border-radius: 10px; font-size: 1rem; flex-shrink: 0;
}
.label-icon-jd  { background: rgba(0,255,163,0.10); color: var(--mint); border: 1px solid rgba(0,255,163,0.20); }
.label-icon-pdf { background: rgba(255,64,96,0.10); color: var(--rose); border: 1px solid rgba(255,64,96,0.20); }

/* =====================================================
   6.  INPUTS
   ===================================================== */
textarea {
  border-radius: var(--radius-sm) !important; border: 1px solid rgba(255,255,255,0.08) !important;
  background: rgba(255,255,255,0.022) !important; padding: 1rem !important; color: var(--text) !important;
  font-family: var(--font) !important; font-size: 0.88rem !important;
  transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
textarea:focus {
  border-color: rgba(0,255,163,0.50) !important;
  box-shadow: 0 0 0 4px rgba(0,255,163,0.08), 0 0 20px rgba(0,255,163,0.04) !important;
  background: rgba(255,255,255,0.03) !important; outline: none !important;
}

div.stFileUploader > label {
  border-radius: var(--radius-sm) !important; border: 2px dashed rgba(255,255,255,0.10) !important;
  padding: 1.3rem !important; background: rgba(0,255,163,0.025) !important;
  color: var(--muted) !important; transition: all 0.3s ease !important;
}
div.stFileUploader > label:hover {
  border-color: rgba(0,255,163,0.40) !important; background: rgba(0,255,163,0.05) !important;
  box-shadow: 0 0 0 4px rgba(0,255,163,0.06) !important;
}

/* =====================================================
   7.  CTA BUTTON
   ===================================================== */
div.stButton > button[kind="primary"] {
  position: relative; border-radius: var(--radius-sm); font-family: var(--font); font-weight: 800;
  font-size: 0.96rem; letter-spacing: 0.015em; padding: 0.92rem 1.5rem;
  border: 1px solid rgba(0,255,163,0.45) !important; color: var(--mint) !important;
  background: rgba(0,255,163,0.06) !important;
  box-shadow: 0 0 0 1px rgba(0,255,163,0.12) inset, 0 4px 20px rgba(0,255,163,0.08);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); animation: btnPulse 3.5s ease-in-out infinite; overflow: hidden;
}
@keyframes btnPulse {
  0%, 100% { box-shadow: 0 0 0 1px rgba(0,255,163,0.12) inset, 0 4px 20px rgba(0,255,163,0.08); }
  50%       { box-shadow: 0 0 0 1px rgba(0,255,163,0.22) inset, 0 6px 32px rgba(0,255,163,0.18), 0 0 60px rgba(0,255,163,0.06); }
}
div.stButton > button[kind="primary"]::before {
  content: ''; position: absolute; top: 0; left: -80%; width: 50%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(0,255,163,0.12), transparent);
  transform: skewX(-22deg); transition: left 0.65s ease;
}
div.stButton > button[kind="primary"]:hover {
  background: rgba(0,255,163,0.10) !important; border-color: rgba(0,255,163,0.70) !important;
  transform: translateY(-2px); box-shadow: 0 0 0 1px rgba(0,255,163,0.30) inset, 0 10px 40px rgba(0,255,163,0.20), 0 0 80px rgba(0,255,163,0.08) !important;
  color: #CAFFEC !important;
}
div.stButton > button[kind="primary"]:hover::before { left: 130%; }
div.stButton > button[kind="primary"]:active { transform: translateY(0) scale(0.99); }

/* =====================================================
   8.  ONBOARDING STEPS
   ===================================================== */
.onboarding { padding: 2rem 0 0; }
.onboarding-title { font-weight: 800; font-size: 1.2rem; color: var(--text); margin-bottom: 1.4rem; display: flex; align-items: center; gap: 10px; }
.onboarding-steps { display: flex; flex-direction: column; gap: 14px; }
.ob-step {
  display: flex; align-items: flex-start; gap: 16px; padding: 16px 18px; border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.055);
  transition: all 0.3s ease; animation: fadeSlideUp 0.5s ease-out both;
}
.ob-step:nth-child(1) { animation-delay: 0.08s; }
.ob-step:nth-child(2) { animation-delay: 0.18s; }
.ob-step:nth-child(3) { animation-delay: 0.28s; }
.ob-step:hover { background: rgba(0,255,163,0.03); border-color: rgba(0,255,163,0.12); transform: translateX(4px); }
.ob-num {
  width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.85rem; flex-shrink: 0; color: #060914;
}
.ob-num-1 { background: linear-gradient(135deg, #00FFA3, #00CC84); }
.ob-num-2 { background: linear-gradient(135deg, #B44DFF, #8A2BE2); }
.ob-num-3 { background: linear-gradient(135deg, #FF4060, #CC2040); }
.ob-step-content { flex: 1; }
.ob-step-title { font-weight: 700; color: var(--text); font-size: 0.92rem; margin-bottom: 3px; }
.ob-step-desc  { color: var(--muted); font-size: 0.84rem; line-height: 1.55; }
.ob-tip {
  margin-top: 1.5rem; padding: 14px 18px; border-radius: var(--radius-sm);
  background: linear-gradient(135deg, rgba(0,255,163,0.05), rgba(180,77,255,0.03));
  border: 1px solid rgba(0,255,163,0.14); border-left: 3px solid var(--mint);
  display: flex; align-items: flex-start; gap: 10px; animation: fadeSlideUp 0.5s ease-out 0.42s both;
}
.ob-tip-icon  { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.ob-tip-text  { color: var(--muted); font-size: 0.84rem; line-height: 1.55; }
.ob-tip-text strong { color: var(--mint); }

@keyframes fadeSlideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

/* =====================================================
   9.  SCORE GAUGE
   ===================================================== */
.score-section { display: flex; flex-direction: column; align-items: center; margin: 8px 0 22px; animation: fadeSlideUp 0.6s ease-out both; }
.score-ring-wrap { position: relative; width: 164px; height: 164px; }
.score-ring-wrap svg { transform: rotate(-90deg); width: 164px; height: 164px; }
.score-ring-bg   { fill: none; stroke: rgba(255,255,255,0.055); stroke-width: 10; }
.score-ring-fill {
  fill: none; stroke-width: 10; stroke-linecap: round;
  transition: stroke-dashoffset 1.6s cubic-bezier(0.4, 0, 0.2, 1);
  filter: drop-shadow(0 0 9px var(--ring-color));
}
.score-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.score-value { font-family: var(--mono); font-weight: 800; font-size: 2.5rem; line-height: 1; letter-spacing: -0.02em; }
.score-label { font-size: 0.70rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); margin-top: 4px; }
.score-ring-glow {
  position: absolute; width: 120px; height: 120px; border-radius: 50%; top: 50%; left: 50%;
  transform: translate(-50%, -50%); animation: scoreGlow 3s ease-in-out infinite; pointer-events: none;
}
@keyframes scoreGlow { 0%, 100% { opacity: 0.35; transform: translate(-50%,-50%) scale(1); } 50% { opacity: 0.65; transform: translate(-50%,-50%) scale(1.12); } }
.score-elapsed { margin-top: 10px; font-size: 0.76rem; color: var(--dimmed); font-family: var(--mono); font-weight: 500; }

/* =====================================================
   10. TAB STYLING
   ===================================================== */
[data-baseweb="tab-list"] {
  border: 1px solid rgba(255,255,255,0.055) !important; border-radius: 14px !important; padding: 5px !important;
  background: rgba(255,255,255,0.02) !important; backdrop-filter: blur(12px); gap: 4px !important;
}
[data-baseweb="tab"] {
  color: var(--dimmed) !important; background: transparent !important; border-radius: 10px !important;
  font-weight: 600 !important; font-size: 0.82rem !important; padding: 8px 14px !important; transition: all 0.25s ease !important;
}
[data-baseweb="tab"]:hover { color: var(--muted) !important; background: rgba(255,255,255,0.03) !important; }
[data-baseweb="tab"][aria-selected="true"] {
  color: var(--mint) !important; background: rgba(0,255,163,0.08) !important; box-shadow: 0 0 0 1px rgba(0,255,163,0.22) inset !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* =====================================================
   11. SCROLLABLE CARD CONTAINER
   ===================================================== */
.result-scroll { max-height: 520px; overflow-y: auto; padding: 4px 2px; scroll-behavior: smooth; }
.result-scroll::-webkit-scrollbar { width: 5px; }
.result-scroll::-webkit-scrollbar-track { background: transparent; }
.result-scroll::-webkit-scrollbar-thumb { background: rgba(0,255,163,0.22); border-radius: 999px; }
.result-scroll::-webkit-scrollbar-thumb:hover { background: rgba(0,255,163,0.40); }

/* =====================================================
   12. GAP CARDS
   ===================================================== */
.gap-card {
  padding: 12px 16px; border-radius: var(--radius-sm); background: rgba(255,255,255,0.018); border: 1px solid rgba(255,255,255,0.055);
  border-left: 4px solid; margin-bottom: 10px; transition: all 0.25s ease; animation: fadeSlideUp 0.4s ease-out both;
}
.gap-card:hover { background: rgba(255,255,255,0.03); transform: translateX(4px); box-shadow: 0 4px 24px rgba(0,0,0,0.18); }
.gap-card.severity-high { border-left-color: #FF4060; } .gap-card.severity-med { border-left-color: #FFB800; } .gap-card.severity-low { border-left-color: #00FFA3; }
.gap-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.gap-badge { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 6px; font-weight: 800; font-size: 0.72rem; flex-shrink: 0; }
.gap-badge-high { background: rgba(255,64,96,0.12); color: #FF4060; border: 1px solid rgba(255,64,96,0.22); }
.gap-badge-med  { background: rgba(255,184,0,0.12); color: #FFB800; border: 1px solid rgba(255,184,0,0.22); }
.gap-badge-low  { background: rgba(0,255,163,0.10); color: #00FFA3; border: 1px solid rgba(0,255,163,0.20); }
.gap-skill-name { font-weight: 800; font-size: 0.92rem; color: var(--text); letter-spacing: -0.01em; }
.gap-body { color: var(--muted); font-size: 0.85rem; line-height: 1.5; padding-left: 32px; white-space: normal; }
.gap-jd-req { margin-top: 6px; padding: 6px 10px; border-radius: 6px; background: rgba(0,255,163,0.05); border: 1px solid rgba(0,255,163,0.14); font-size: 0.80rem; color: #A8FFD8; }
.jd-req-label { font-weight: 700; color: var(--mint); }

/* =====================================================
   13. IMPROVEMENT CARDS
   ===================================================== */
.imp-card {
  display: flex; gap: 12px; align-items: flex-start; padding: 12px 16px; border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.018); border: 1px solid rgba(255,255,255,0.055); margin-bottom: 10px; transition: all 0.25s ease; animation: fadeSlideUp 0.4s ease-out both;
}
.imp-card:hover { background: rgba(0,255,163,0.025); border-color: rgba(0,255,163,0.14); transform: translateX(4px); }
.imp-check { width: 26px; height: 26px; border-radius: 8px; display: flex; align-items: center; justify-content: center; background: rgba(0,255,163,0.08); border: 1px solid rgba(0,255,163,0.22); color: var(--mint); font-size: 0.80rem; font-weight: 800; flex-shrink: 0; }
.imp-content { flex: 1; }
.imp-num { font-weight: 800; font-size: 0.74rem; color: var(--mint); font-family: var(--mono); letter-spacing: 0.03em; margin-bottom: 2px; }
.imp-text { color: var(--text); font-size: 0.88rem; line-height: 1.5; white-space: normal; font-weight: 500; }

/* =====================================================
   14. PREPARATION TIMELINE
   ===================================================== */
.timeline { position: relative; padding-left: 22px; }
.timeline::before { content: ''; position: absolute; left: 9px; top: 18px; bottom: 18px; width: 2px; background: linear-gradient(180deg, rgba(180,77,255,0.45) 0%, rgba(180,77,255,0.06) 100%); border-radius: 999px; }
.prep-card { position: relative; padding: 12px 16px; border-radius: var(--radius-sm); background: rgba(255,255,255,0.018); border: 1px solid rgba(255,255,255,0.055); margin-bottom: 12px; margin-left: 16px; transition: all 0.25s ease; animation: fadeSlideUp 0.4s ease-out both; }
.prep-card:hover { background: rgba(180,77,255,0.03); border-color: rgba(180,77,255,0.16); transform: translateX(4px); }
.prep-dot { position: absolute; left: -25px; top: 16px; width: 12px; height: 12px; border-radius: 50%; background: var(--violet); border: 3px solid var(--bg1); box-shadow: 0 0 0 3px rgba(180,77,255,0.22); }
.prep-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.prep-num { font-family: var(--mono); font-weight: 700; font-size: 0.72rem; color: var(--violet); padding: 2px 8px; border-radius: 6px; background: rgba(180,77,255,0.10); border: 1px solid rgba(180,77,255,0.20); }
.prep-gap-title { font-weight: 800; font-size: 0.90rem; color: #E0CCFF; letter-spacing: -0.01em; }
.prep-text { color: var(--muted); font-size: 0.85rem; line-height: 1.5; white-space: normal; }
.prep-sections { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
.prep-sec { display: flex; flex-direction: column; gap: 2px; padding: 8px 10px; border-radius: 6px; background: rgba(255,255,255,0.018); border: 1px solid rgba(255,255,255,0.045); }
.prep-tag { display: inline-block; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; width: fit-content; padding: 2px 6px; border-radius: 4px; }
.tag-study    { background: rgba(56,189,248,0.12); color: #38BDF8; }
.tag-practice { background: rgba(0,255,163,0.10); color: #00FFA3; }
.tag-angle    { background: rgba(255,184,0,0.12); color: #FFB800; }
.prep-sec-text { font-size: 0.86rem; color: var(--muted); line-height: 1.55; }

/* STAGGERED ANIMATION DELAYS */
.gap-card:nth-child(1),.imp-card:nth-child(1),.prep-card:nth-child(1) { animation-delay:0.04s; }
.gap-card:nth-child(2),.imp-card:nth-child(2),.prep-card:nth-child(2) { animation-delay:0.10s; }
.gap-card:nth-child(3),.imp-card:nth-child(3),.prep-card:nth-child(3) { animation-delay:0.16s; }
.gap-card:nth-child(4),.imp-card:nth-child(4),.prep-card:nth-child(4) { animation-delay:0.22s; }

/* DIVIDER */
hr { border: none !important; height: 1px !important; background: linear-gradient(90deg, transparent, rgba(0,255,163,0.12), transparent) !important; margin: 1.2rem 0 !important; }

/* EMPTY STATE */
.empty-state { text-align: center; padding: 2.5rem 1rem; color: var(--dimmed); }
.empty-state-icon { font-size: 2rem; margin-bottom: 8px; opacity: 0.45; }
.empty-state-text { font-size: 0.88rem; }

/* LOADING ANIMATION */
.loading-container { display: flex; flex-direction: column; align-items: center; padding: 3.5rem 1rem; animation: fadeSlideUp 0.5s ease-out both; }
.loading-spinner { width: 58px; height: 58px; border-radius: 50%; border: 3px solid rgba(0,255,163,0.08); border-top-color: var(--mint); border-right-color: var(--violet); animation: spin 0.9s linear infinite; margin-bottom: 1.5rem; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text  { font-size: 0.94rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.loading-sub   { font-size: 0.8rem; color: var(--dimmed); font-family: var(--mono); }
.loading-stages { display: flex; flex-direction: column; gap: 10px; margin-top: 1.6rem; width: 100%; max-width: 340px; }
.loading-stage { display: flex; align-items: center; gap: 10px; font-size: 0.8rem; padding: 9px 13px; border-radius: 9px; background: rgba(255,255,255,0.018); border: 1px solid rgba(255,255,255,0.05); animation: fadeSlideUp 0.4s ease-out both; }
.loading-stage:nth-child(1) { animation-delay: 0.0s; } .loading-stage:nth-child(2) { animation-delay: 0.25s; } .loading-stage:nth-child(3) { animation-delay: 0.5s; } .loading-stage:nth-child(4) { animation-delay: 0.75s; }
.stage-dot { width: 8px; height: 8px; border-radius: 50%; }
.stage-text { color: var(--muted); font-family: var(--mono); font-weight: 500; }

/* TAB SECTION HEADERS */
.tab-header { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; padding-bottom: 13px; border-bottom: 1px solid rgba(255,255,255,0.045); }
.tab-header-icon { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1rem; }
.tab-header-icon.gaps { background: rgba(255,64,96,0.10); border: 1px solid rgba(255,64,96,0.18); }
.tab-header-icon.imps { background: rgba(0,255,163,0.10); border: 1px solid rgba(0,255,163,0.18); }
.tab-header-icon.prep { background: rgba(180,77,255,0.10); border: 1px solid rgba(180,77,255,0.18); }
.tab-header-title { font-weight: 700; font-size: 0.96rem; color: var(--text); }
.tab-header-count { font-family: var(--mono); font-size: 0.72rem; font-weight: 700; padding: 2px 9px; border-radius: 6px; color: var(--muted); background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.055); }

/* STREAMLIT-SPECIFIC OVERRIDES */
[data-testid="stApp"] { background: var(--bg1) !important; }
[data-testid="column"] { background: transparent !important; }
[data-testid="stMainBlockContainer"], .main .block-container { background: transparent !important; max-width: 1400px !important; padding-left: 2rem !important; padding-right: 2rem !important; }
[data-testid="stTextArea"] > div > div { background: rgba(255,255,255,0.022) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
[data-testid="stTextArea"] textarea { background: transparent !important; border: none !important; box-shadow: none !important; }
[data-testid="stTextArea"] textarea:focus { border: none !important; box-shadow: none !important; }
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploaderDropzone"] { background: rgba(0,255,163,0.025) !important; border: 2px dashed rgba(0,255,163,0.25) !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(0,255,163,0.05) !important; border-color: rgba(0,255,163,0.50) !important; }
[data-testid="stDivider"] hr { border-color: rgba(255,255,255,0.06) !important; }
[data-testid="stAlert"] { background: rgba(255,64,96,0.06) !important; border: 1px solid rgba(255,64,96,0.22) !important; border-radius: 12px !important; color: #FFB3BE !important; }

/* Sidebar Styling */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0A0F1F 0%, #060914 100%) !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }

/* Keyword Heatmap */
.keyword-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.kw-chip { padding: 6px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; font-family: var(--mono); transition: all 0.2s ease; }
.kw-found { background: rgba(0,255,163,0.12); color: #00FFA3; border: 1px solid rgba(0,255,163,0.25); }
.kw-missing { background: rgba(255,64,96,0.10); color: #FF4060; border: 1px solid rgba(255,64,96,0.20); }
.kw-chip:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }

/* Feedback Buttons */
.feedback-row { display: flex; gap: 8px; margin-top: 8px; }
.fb-btn { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 4px 10px; cursor: pointer; font-size: 0.8rem; transition: all 0.2s; }
.fb-btn:hover { background: rgba(255,255,255,0.08); }
.fb-btn.active-up { background: rgba(0,255,163,0.15); border-color: rgba(0,255,163,0.3); }
.fb-btn.active-down { background: rgba(255,64,96,0.15); border-color: rgba(255,64,96,0.3); }

/* History Table */
.history-card { padding: 14px 18px; border-radius: var(--radius-sm); background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); margin-bottom: 10px; transition: all 0.25s; cursor: pointer; }
.history-card:hover { background: rgba(0,255,163,0.03); border-color: rgba(0,255,163,0.15); transform: translateX(4px); }
.history-score { font-family: var(--mono); font-weight: 800; font-size: 1.4rem; }
.history-meta { color: var(--muted); font-size: 0.82rem; }
.history-filename { color: var(--text); font-weight: 600; font-size: 0.9rem; }

/* Download Button */
.download-btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 20px; border-radius: 10px; background: linear-gradient(135deg, rgba(0,255,163,0.15), rgba(0,255,163,0.05)); border: 1px solid rgba(0,255,163,0.25); color: #00FFA3; font-weight: 700; font-size: 0.88rem; cursor: pointer; transition: all 0.3s; text-decoration: none; }
.download-btn:hover { background: linear-gradient(135deg, rgba(0,255,163,0.25), rgba(0,255,163,0.10)); transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,255,163,0.15); }

/* Stats Row */
.stats-row { display: flex; gap: 12px; margin: 16px 0; }
.stat-card { flex: 1; padding: 14px 16px; border-radius: var(--radius-sm); background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); text-align: center; }
.stat-value { font-family: var(--mono); font-weight: 800; font-size: 1.3rem; }
.stat-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Init
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "analysis"
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "out" not in st.session_state:
    st.session_state.out = None
if "error" not in st.session_state:
    st.session_state.error = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:20px 0 10px"><div style="font-size:1.6rem;font-weight:900;background:linear-gradient(135deg,#00FFA3,#38BDF8);-webkit-background-clip:text;-webkit-text-fill-color:transparent">🤖 Resume Analyzer</div><div style="color:var(--muted);font-size:0.78rem;margin-top:4px">v2.0 · Powered by AI</div></div>', unsafe_allow_html=True)
    st.markdown('---')
    
    if st.button("🔬 New Analysis", use_container_width=True, type="primary" if st.session_state.page == "analysis" else "secondary"):
        st.session_state.page = "analysis"
        st.rerun()
    if st.button("📊 History & Trends", use_container_width=True, type="primary" if st.session_state.page == "history" else "secondary"):
        st.session_state.page = "history"
        st.rerun()
    
    st.markdown('---')
    
    # System Status
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        status_color = '#00FFA3' if health.get('ollama_reachable') else '#FF4060'
        status_text = 'Online' if health.get('ollama_reachable') else 'Offline'
        st.markdown(f'<div style="padding:10px;border-radius:8px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06)"><div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;border-radius:50%;background:{status_color}"></div><span style="color:var(--muted);font-size:0.8rem">Ollama: {status_text}</span></div><div style="color:var(--dimmed);font-size:0.72rem;margin-top:6px">Model: {health.get("model", "unknown")}</div></div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div style="padding:10px;border-radius:8px;background:rgba(255,64,96,0.05);border:1px solid rgba(255,64,96,0.15)"><div style="color:#FF4060;font-size:0.8rem">⚠️ Backend offline</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def render_analysis_page():
    st.markdown('<div class="hero-container"><h1 class="hero-title">Agentic Resume Analyzer</h1><div class="hero-subtitle"><span class="hero-badge">LOCAL</span>Ollama · LangGraph · ChromaDB · sentence-transformers — no API keys needed</div></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label"><span class="label-icon label-icon-jd">📋</span>Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area("Job Description", height=240, placeholder="Paste the full Job Description here — skills, tools, responsibilities…", label_visibility="collapsed")
        st.divider()
        st.markdown('<div class="section-label"><span class="label-icon label-icon-pdf">📄</span>Resume PDF</div>', unsafe_allow_html=True)
        resume_file = st.file_uploader("Resume PDF", type=["pdf"], accept_multiple_files=False, help="Upload a PDF resume. Everything runs 100% locally on your machine.", label_visibility="collapsed")
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        submitted = st.button("🚀  Analyze Resume", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)

        if submitted:
            st.session_state.submitted = True
            st.session_state.out = None
            st.session_state.error = None

        if not st.session_state.submitted:
            st.markdown('<div class="onboarding"><div class="onboarding-title">✨ How it works</div><div class="onboarding-steps"><div class="ob-step"><div class="ob-num ob-num-1">1</div><div class="ob-step-content"><div class="ob-step-title">Paste the Job Description</div><div class="ob-step-desc">Copy the JD from the job listing — include skills, tools, and requirements for best results.</div></div></div><div class="ob-step"><div class="ob-num ob-num-2">2</div><div class="ob-step-content"><div class="ob-step-title">Upload your Resume</div><div class="ob-step-desc">Drop your resume as a PDF. The AI will parse every section and extract your skills automatically.</div></div></div><div class="ob-step"><div class="ob-num ob-num-3">3</div><div class="ob-step-content"><div class="ob-step-title">Get Instant Analysis</div><div class="ob-step-desc">Receive a match score, identified gaps, actionable improvements, and interview prep — all locally.</div></div></div></div><div class="ob-tip"><div class="ob-tip-icon">💡</div><div class="ob-tip-text"><strong>Pro tip:</strong> Include specific tools, frameworks, and years of experience in the JD for more precise gap detection and targeted improvement suggestions.</div></div></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if not job_description.strip():
            st.error("Please paste a Job Description.")
            st.session_state.error = "Missing job description."
        elif resume_file is None:
            st.error("Please upload a Resume PDF.")
            st.session_state.error = "Missing resume PDF."
        else:
            STAGE_LABELS = {
                "started":          ("⚙️",  "Starting analysis pipeline…"),
                "extracting":       ("📄",  "Parsing PDF and building embeddings…"),
                "retrieved":        ("🔍",  "Multi-query RAG retrieval complete"),
                "analyzing_gaps":   ("🧠",  "Identifying skill gaps…"),
                "gaps_found":       ("📋",  "Gaps identified"),
                "scoring":          ("📊",  "Generating score and recommendations…"),
                "complete":         ("✅",  "Analysis complete!"),
                "error":            ("❌",  "An error occurred"),
            }

            def _loading_html(stage_key: str, message: str) -> str:
                icon, _ = STAGE_LABELS.get(stage_key, ("⚙️", message))
                all_stages = [("extracting", "Parsing PDF & embeddings"), ("retrieved", "Multi-query RAG retrieval"), ("analyzing_gaps", "Identifying skill gaps"), ("scoring", "Generating recommendations")]
                stage_keys = [s[0] for s in all_stages]
                current_idx = stage_keys.index(stage_key) if stage_key in stage_keys else -1
                stages_html = ""
                for i, (skey, slabel) in enumerate(all_stages):
                    if i < current_idx: dot_style, text_style = "background:#34D399;", "color:#34D399;"
                    elif i == current_idx: dot_style, text_style = "background:#818CF8; animation:dotPulse 1.2s ease-in-out infinite;", "color:#F1F5F9;"
                    else: dot_style, text_style = "background:rgba(255,255,255,0.12);", "color:#64748B;"
                    stages_html += f'<div class="loading-stage"><div class="stage-dot" style="{dot_style}"></div><span class="stage-text" style="{text_style}">{slabel}</span></div>'
                return f'<div class="loading-container"><div class="loading-spinner"></div><div class="loading-text">{icon} {message}</div><div class="loading-sub">Running local pipeline — no data leaves your machine</div><div class="loading-stages">{stages_html}</div></div>'

            loading_placeholder = st.empty()
            if not st.session_state.out and not st.session_state.error:
                loading_placeholder.markdown(_loading_html("started", "Starting analysis pipeline…"), unsafe_allow_html=True)

            t0 = time.perf_counter()
            out = st.session_state.out
            error_msg = st.session_state.error
            elapsed = 0.0

            if not out and not error_msg:
                try:
                    resume_bytes = resume_file.getvalue()
                    files = {"resume_pdf": (resume_file.name, resume_bytes, "application/pdf")}
                    data = {"job_description": job_description}
                    with requests.post(f"{BACKEND_URL}/analyze/stream", data=data, files=files, stream=True, timeout=420) as resp:
                        resp.raise_for_status()
                        for raw_line in resp.iter_lines():
                            if not raw_line: continue
                            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                            if not line.startswith("data:"): continue
                            payload_str = line[len("data:"):].strip()
                            try: evt = json.loads(payload_str)
                            except json.JSONDecodeError: continue
                            event_type = evt.get("event", "")
                            event_msg  = evt.get("message", "")
                            if event_type == "result": out = evt.get("data", {})
                            elif event_type == "error": error_msg = event_msg
                            else: loading_placeholder.markdown(_loading_html(event_type, event_msg), unsafe_allow_html=True)
                    st.session_state.out = out
                    st.session_state.error = error_msg
                except requests.exceptions.ConnectionError: st.session_state.error = "Cannot reach the backend at http://localhost:8000 — is it running?"
                except requests.exceptions.RequestException as exc: st.session_state.error = f"Backend request failed: {exc}"
                except Exception as exc: st.session_state.error = f"Unexpected error: {exc}"
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
                
                keyword_total = len(keywords)
                keyword_found = sum(1 for k in keywords if k.get('found_in_resume'))
                kw_pct = round((keyword_found / keyword_total * 100) if keyword_total > 0 else 0)

                if score >= 8: ring_color, score_fg, glow_bg = "#00FFA3", "#00FFA3", "rgba(0,255,163,0.10)"
                elif score >= 5: ring_color, score_fg, glow_bg = "#FFB800", "#FFB800", "rgba(255,184,0,0.10)"
                else: ring_color, score_fg, glow_bg = "#FF4060", "#FF4060", "rgba(255,64,96,0.10)"

                radius, circumference = 62, 2 * 3.14159 * 62
                progress = score / 10
                dash_offset = circumference * (1 - progress)

                st.markdown(f'<div class="score-section"><div class="score-ring-wrap"><div class="score-ring-glow" style="background: radial-gradient(circle, {glow_bg}, transparent 70%);"></div><svg viewBox="0 0 160 160"><circle class="score-ring-bg" cx="80" cy="80" r="{radius}" /><circle class="score-ring-fill" cx="80" cy="80" r="{radius}" stroke="{ring_color}" stroke-dasharray="{circumference}" stroke-dashoffset="{dash_offset}" style="--ring-color: {ring_color};" /></svg><div class="score-center"><div class="score-value" style="color: {score_fg};">{score}<span style="font-size:1.2rem; color: var(--dimmed);">/10</span></div><div class="score-label">Match Score</div></div></div><div class="score-elapsed">completed in {elapsed:.1f}s</div></div>', unsafe_allow_html=True)
                
                # Stats Row
                st.markdown(f'<div class="stats-row"><div class="stat-card"><div class="stat-value" style="color: var(--rose)">{len(gaps)}</div><div class="stat-label">Skill Gaps</div></div><div class="stat-card"><div class="stat-value" style="color: var(--mint)">{kw_pct}%</div><div class="stat-label">Keyword Match</div></div><div class="stat-card"><div class="stat-value" style="color: var(--sky)">{elapsed:.1f}s</div><div class="stat-label">Analysis Time</div></div></div>', unsafe_allow_html=True)

                gaps_tab, imp_tab, prep_tab, kw_tab = st.tabs(["🔍  Identified Gaps", "⚡  Improvements", "🎯  Preparation", "🔑 Keywords"])

                with gaps_tab:
                    st.markdown(f'<div class="tab-header"><div class="tab-header-icon gaps">🔍</div><div class="tab-header-title">Identified Skill Gaps</div><div class="tab-header-count">{len(gaps)} found</div></div>', unsafe_allow_html=True)
                    if gaps:
                        st.markdown('<div class="result-scroll">', unsafe_allow_html=True)
                        for i, g in enumerate(gaps, start=1):
                            if score < 5: sev_class, badge_class = "severity-high", "gap-badge-high"
                            elif score < 8: sev_class, badge_class = "severity-med", "gap-badge-med"
                            else: sev_class, badge_class = "severity-low", "gap-badge-low"
                            text = g.strip()
                            colon_idx = text.find(":")
                            if 0 < colon_idx < 60: skill_name, explanation = text[:colon_idx].strip(), text[colon_idx+1:].strip()
                            else: skill_name, explanation = f"Gap {i}", text
                            jd_parts = re.split(r"\s*[—-]\s*JD requires:\s*", explanation, flags=re.IGNORECASE)
                            body_text = jd_parts[0].strip()
                            jd_req = jd_parts[1].strip() if len(jd_parts) > 1 and jd_parts[1].strip() else ""
                            jd_html = f'<div class="gap-jd-req"><span class="jd-req-label">📌 JD Requires:</span> {jd_req}</div>' if jd_req else ''
                            st.markdown(f'<div class="gap-card {sev_class}"><div class="gap-header"><div class="gap-badge {badge_class}">{i}</div><div class="gap-skill-name">{skill_name}</div></div><div class="gap-body"><div>{body_text}</div>{jd_html}</div></div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else: st.markdown('<div class="empty-state"><div class="empty-state-icon">✅</div><div class="empty-state-text">No gaps detected — great match!</div></div>', unsafe_allow_html=True)

                with imp_tab:
                    st.markdown(f'<div class="tab-header"><div class="tab-header-icon imps">⚡</div><div class="tab-header-title">Actionable Improvements</div><div class="tab-header-count">{len(improvements)} items</div></div>', unsafe_allow_html=True)
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
                            area_html = f'<div class="prep-gap-title" style="color: #A8FFD8; font-size: 0.88rem;">📌 Area: {area_val}</div>' if area_val else ''
                            align_html = f'<div class="prep-sec" style="margin-top: 6px; border-color: rgba(0,255,163,0.18); background: rgba(0,255,163,0.02);"><span class="prep-tag tag-practice">🎯 JD Alignment</span><div class="prep-sec-text">{align_val}</div></div>' if align_val else ''
                            st.markdown(f'<div class="imp-card"><div class="imp-check">✓</div><div class="imp-content"><div class="prep-header"><div class="imp-num">#{i:02d}</div>{area_html}</div><div class="imp-text" style="font-weight: 600; color: #FFFFFF;">{action_val}</div>{align_html}</div></div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else: st.markdown('<div class="empty-state"><div class="empty-state-icon">🎉</div><div class="empty-state-text">No improvements needed — your resume is strong!</div></div>', unsafe_allow_html=True)

                with prep_tab:
                    st.markdown(f'<div class="tab-header"><div class="tab-header-icon prep">🎯</div><div class="tab-header-title">Interview Preparation</div><div class="tab-header-count">{len(preparation)} steps</div></div>', unsafe_allow_html=True)
                    if preparation:
                        st.markdown('<div class="result-scroll"><div class="timeline">', unsafe_allow_html=True)
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
                            if gap_val or study_val or practice_val or angle_val:
                                gap_title_html = f'<div class="prep-gap-title">Target: {gap_val}</div>' if gap_val else ''
                                study_html = f'<div class="prep-sec"><span class="prep-tag tag-study">📖 Study</span><div class="prep-sec-text">{study_val}</div></div>' if study_val else ''
                                practice_html = f'<div class="prep-sec"><span class="prep-tag tag-practice">🛠️ Practice</span><div class="prep-sec-text">{practice_val}</div></div>' if practice_val else ''
                                angle_html = f'<div class="prep-sec"><span class="prep-tag tag-angle">💬 Interview Angle</span><div class="prep-sec-text">{angle_val}</div></div>' if angle_val else ''
                                card_inner = f'<div class="prep-header"><div class="prep-num">Step {i}</div>{gap_title_html}</div><div class="prep-sections">{study_html}{practice_html}{angle_html}</div>'
                            else: card_inner = f'<div class="prep-header"><div class="prep-num">Step {i}</div></div><div class="prep-text">{text}</div>'
                            st.markdown(f'<div class="prep-card"><div class="prep-dot"></div>{card_inner}</div>', unsafe_allow_html=True)
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    else: st.markdown('<div class="empty-state"><div class="empty-state-icon">📚</div><div class="empty-state-text">No preparation topics generated.</div></div>', unsafe_allow_html=True)

                with kw_tab:
                    st.markdown(f'<div class="tab-header"><div class="tab-header-icon" style="background:rgba(56,189,248,0.12);color:#38BDF8">🔑</div><div class="tab-header-title">JD Keyword Match</div><div class="tab-header-count">{keyword_found}/{keyword_total} matched</div></div>', unsafe_allow_html=True)
                    found_chips = ''.join(f'<span class="kw-chip kw-found">✓ {k["keyword"]}</span>' for k in keywords if k.get('found_in_resume'))
                    missing_chips = ''.join(f'<span class="kw-chip kw-missing">✗ {k["keyword"]}</span>' for k in keywords if not k.get('found_in_resume'))
                    if found_chips: st.markdown(f'<div style="margin-bottom:6px;color:var(--muted);font-size:0.82rem;font-weight:600">✅ Found in Resume</div><div class="keyword-grid">{found_chips}</div>', unsafe_allow_html=True)
                    if missing_chips: st.markdown(f'<div style="margin-top:14px;margin-bottom:6px;color:var(--muted);font-size:0.82rem;font-weight:600">❌ Missing from Resume</div><div class="keyword-grid">{missing_chips}</div>', unsafe_allow_html=True)

                if analysis_id:
                    try:
                        pdf_resp = requests.post(f"{BACKEND_URL}/export/pdf", data={"analysis_id": analysis_id}, timeout=30)
                        if pdf_resp.status_code == 200:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_resp.content,
                                file_name=f"resume_analysis_{analysis_id}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    except Exception:
                        pass

        st.markdown("</div>", unsafe_allow_html=True)

def render_history_page():
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="tab-header"><div class="tab-header-icon" style="background:rgba(180,77,255,0.12);color:#B44DFF">📊</div><div class="tab-header-title">Analysis History</div></div>', unsafe_allow_html=True)
    
    try:
        trend_resp = requests.get(f"{BACKEND_URL}/history/trend", timeout=10)
        if trend_resp.status_code == 200:
            trend_data = trend_resp.json()
            if len(trend_data) >= 2:
                df = pd.DataFrame(trend_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.markdown('<div style="color:var(--muted);font-size:0.82rem;font-weight:600;margin-bottom:8px">📈 Score Trend</div>', unsafe_allow_html=True)
                st.line_chart(df.set_index('timestamp')['score'], color='#00FFA3', height=200)
        
        history_resp = requests.get(f"{BACKEND_URL}/history", timeout=10)
        if history_resp.status_code == 200:
            history = history_resp.json()
            if not history:
                st.markdown('<div class="empty-state"><div class="empty-state-icon">📝</div><div class="empty-state-text">No analyses yet. Run your first analysis!</div></div>', unsafe_allow_html=True)
            else:
                for item in history:
                    score = item.get('score', 0)
                    score_color = '#00FFA3' if score >= 8 else '#FFB800' if score >= 5 else '#FF4060'
                    ts = item.get('timestamp', '')[:10]
                    fname = item.get('resume_filename', 'unknown')
                    jd = item.get('jd_snippet', '')[:100]
                    gaps = item.get('gap_count', 0)
                    kw_pct = item.get('keyword_match_pct', 0)
                    elapsed = item.get('elapsed_seconds', 0)
                    aid = item.get('id', '')
                    
                    st.markdown(f'<div class="history-card"><div style="display:flex;justify-content:space-between;align-items:center"><div><div class="history-filename">📄 {fname}</div><div class="history-meta">{ts} · {gaps} gaps · {kw_pct}% keywords · {elapsed:.1f}s</div><div class="history-meta" style="margin-top:4px">{jd}...</div></div><div class="history-score" style="color:{score_color}">{score}/10</div></div></div>', unsafe_allow_html=True)
                    
                    if st.button(f"🗑️ Delete", key=f"del_{aid}", help="Delete this analysis"):
                        requests.delete(f"{BACKEND_URL}/history/{aid}", timeout=5)
                        st.rerun()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend at http://localhost:8000")
    except Exception as e:
        st.error(f"Error loading history: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page Router
# ---------------------------------------------------------------------------
if st.session_state.page == "history":
    render_history_page()
else:
    render_analysis_page()
