"""
Dead Reckoning — Spec Inference Engine for Undocumented Industrial Parts
========================================================================
An AI system that reconstructs complete product specifications for industrial
parts where NO external data exists — using engineering domain reasoning,
physics-based inference, and vision analysis.

Built for UniHack 2026 | Powered by Groq
"""

import os
import streamlit as st
import json
import base64
import io
import csv
import time
from datetime import datetime

import pandas as pd
from PIL import Image
from groq import Groq
import re
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# ─── Load environment variables ─────────────────────────────────────────────

load_dotenv()

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dead Reckoning — Spec Inference Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Fully Bright Professional Theme ────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global Overrides ───────────────────────────── */
html, body {
    font-family: 'Inter', sans-serif;
}

/* Exempt Material icons from the font override to prevent ligature text rendering */
.material-symbols-rounded, 
[class*="Icon"], 
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

.stApp {
    background: #F8FAFC;
}

/* ── Hide default Streamlit branding ───────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Sidebar — Bright White/Blue ────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F0F4FF 100%);
    border-right: 1px solid #E2E8F0;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #475569 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1E293B !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #E2E8F0;
}

/* ── Custom Header — Bright Gradient ────────────── */
.app-header {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0F4FF 30%, #EDE9FE 70%, #F5F3FF 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid #DBEAFE;
}

.app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(37,99,235,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.app-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%);
    border-radius: 50%;
}

.app-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2563EB, #7C3AED, #2563EB);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
}

.app-header p {
    margin: 0.5rem 0 0 0;
    color: #475569;
    font-size: 1.05rem;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

.header-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}

/* ── Cards ──────────────────────────────────────── */
.spec-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.spec-card:hover {
    box-shadow: 0 10px 25px rgba(37,99,235,0.06), 0 4px 10px rgba(0,0,0,0.03);
    transform: translateY(-1px);
}

.spec-card h3 {
    margin: 0 0 1rem 0;
    color: #1E293B;
    font-weight: 700;
    font-size: 1.15rem;
}

/* ── Badges ─────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-measured {
    background: #D1FAE5;
    color: #065F46;
    border: 1px solid #A7F3D0;
}

.badge-derived {
    background: #DBEAFE;
    color: #1E40AF;
    border: 1px solid #BFDBFE;
}

.badge-inferred {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}

.badge-assumed {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FECACA;
}

/* ── Confidence bar ─────────────────────────────── */
.confidence-bar-container {
    background: #F1F5F9;
    border-radius: 10px;
    height: 12px;
    overflow: hidden;
    margin: 0.3rem 0;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.8s ease;
}

.confidence-high { background: linear-gradient(90deg, #059669, #34D399); }
.confidence-mid { background: linear-gradient(90deg, #D97706, #FBBF24); }
.confidence-low { background: linear-gradient(90deg, #DC2626, #F87171); }

/* ── Overall Confidence ─────────────────────────── */
.overall-confidence {
    background: linear-gradient(135deg, #F0F4FF 0%, #EDE9FE 100%);
    border: 1px solid #C7D2FE;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

.overall-confidence .score {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.overall-confidence .label {
    font-size: 0.9rem;
    color: #64748B;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.overall-bar {
    background: #E2E8F0;
    border-radius: 12px;
    height: 16px;
    overflow: hidden;
    margin-top: 0.75rem;
}

.overall-bar-fill {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    transition: width 1s ease;
}

/* ── Result Header — Bright Blue Gradient ───────── */
.result-header {
    background: linear-gradient(135deg, #2563EB 0%, #3B82F6 50%, #7C3AED 100%);
    color: black;
    padding: 1.2rem 1.5rem;
    border-radius: 12px 12px 0 0;
    margin-bottom: 0;
}

.result-header h2 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 700;
    color: #000000;
}

.result-header p {
    margin: 0.3rem 0 0 0;
    font-size: 0.85rem;
    color: rgba(0,0,0,0.85);
}

.result-body {
    border: 1px solid #E2E8F0;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 1.5rem;
    background: #FFFFFF;
}

/* ── Spec Table ─────────────────────────────────── */
.spec-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.9rem;
}

.spec-table thead th {
    background: #F0F4FF;
    color: #374151;
    font-weight: 600;
    text-align: left;
    padding: 12px 14px;
    border-bottom: 2px solid #DBEAFE;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.spec-table tbody td {
    padding: 11px 14px;
    border-bottom: 1px solid #F1F5F9;
    color: #334155;
    vertical-align: top;
}

.spec-table tbody tr:last-child td {
    border-bottom: none;
}

.spec-table tbody tr:hover {
    background: #F8FAFC;
}

.spec-field {
    font-weight: 600;
    color: #1E293B;
}

.spec-value {
    color: #2563EB;
    font-weight: 500;
}

.spec-reasoning {
    font-size: 0.8rem;
    color: #64748B;
    max-width: 300px;
}

/* ── Flow Steps (About tab) ─────────────────────── */
.flow-step {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
    height: 100%;
}

.flow-step:hover {
    border-color: #2563EB;
    box-shadow: 0 4px 15px rgba(37,99,235,0.12);
    transform: translateY(-2px);
}

.flow-step .step-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.flow-step .step-num {
    font-size: 0.7rem;
    font-weight: 700;
    color: #2563EB;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.flow-step h4 {
    margin: 0.4rem 0;
    color: #1E293B;
    font-size: 0.95rem;
}

.flow-step p {
    margin: 0;
    font-size: 0.8rem;
    color: #64748B;
    line-height: 1.4;
}

/* ── Comparison Table ───────────────────────────── */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
    margin: 1rem 0;
}

.comparison-table th {
    padding: 14px 18px;
    font-weight: 700;
    font-size: 0.9rem;
}

.comparison-table th:first-child {
    background: #F8FAFC;
    color: #475569;
    text-align: left;
}

.comparison-table th:nth-child(2) {
    background: linear-gradient(135deg, #EDE9FE, #DBEAFE);
    color: #1E293B;
}

.comparison-table th:nth-child(3) {
    background: #F1F5F9;
    color: #64748B;
}

.comparison-table td {
    padding: 12px 18px;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.88rem;
    color: #334155;
}

.comparison-table td:first-child {
    font-weight: 600;
    color: #334155;
}

/* ── Section Divider ────────────────────────────── */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
    margin: 2rem 0;
}

/* ── Info Box Sidebar — Bright ──────────────────── */
.sidebar-info {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.82rem;
    color: #1E40AF;
    line-height: 1.6;
}

/* ── Status Indicator ───────────────────────────── */
.status-connected {
    background: #D1FAE5;
    color: #065F46;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid #A7F3D0;
    text-align: center;
}

.status-disconnected {
    background: #FEE2E2;
    color: #991B1B;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid #FECACA;
    text-align: center;
}

/* ── Stat Box — Bright ──────────────────────────── */
.stat-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.stat-box .stat-label {
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
}

.stat-box .stat-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #2563EB;
}

/* ── Streamlit button overrides ─────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s ease;
    border: none;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.25);
}

div[data-testid="stForm"] {
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    background: #FFFFFF;
}

/* ── Download buttons ───────────────────────────── */
.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

/* ── Tabs ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #FFFFFF;
    padding: 0.5rem;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    color: #64748B;
    padding: 0.5rem 1.2rem;
}

.stTabs [aria-selected="true"] {
    background: #EFF6FF;
    color: #2563EB;
    box-shadow: 0 1px 3px rgba(37,99,235,0.1);
}

/* ── Expander ───────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    color: #1E293B;
}

/* ── Text input overrides ───────────────────────── */
.stTextArea textarea,
.stTextInput input {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    color: #1E293B;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #2563EB;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.12);
}

/* ── Select box ─────────────────────────────────── */
.stSelectbox > div > div {
    background: #FFFFFF;
    border-radius: 8px;
}

/* ── Warning card ───────────────────────────────── */
.warning-card {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #92400E;
    font-size: 0.9rem;
}

/* ── Nameplate decoder layout ───────────────────── */
.decoder-upload {
    background: #F0F4FF;
    border: 2px dashed #93C5FD;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: border-color 0.2s;
}

.decoder-upload:hover {
    border-color: #2563EB;
    background: #EFF6FF;
}

/* ── Sidebar branding ───────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 0.8rem 0 1.2rem 0;
}

.sidebar-brand .brand-icon {
    font-size: 2.5rem;
    margin-bottom: 0.3rem;
}

.sidebar-brand h2 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-brand .tagline {
    margin: 0;
    font-size: 0.72rem;
    color: #64748B;
    letter-spacing: 0.06em;
    font-weight: 600;
    text-transform: uppercase;
}

/* ── Sidebar footer ─────────────────────────────── */
.sidebar-footer {
    text-align: center;
    font-size: 0.72rem;
    color: #94A3B8;
    padding: 0.5rem 0;
}

.sidebar-footer strong {
    color: #64748B;
}

/* ── Data frame overrides ───────────────────────── */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── API Key Loading (from .env) ─────────────────────────────────────────────


def _load_api_key() -> str:
    """Load API key from environment, st.secrets, or .env file."""
    # Priority 1: Streamlit secrets (for Streamlit Cloud deployment)
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # Priority 2: Environment variable (from .env via dotenv or system env)
    key = os.getenv("GROQ_API_KEY", "")
    return key if key else ""


# ─── Session State Initialization ────────────────────────────────────────────

DEFAULTS = {
    "api_key": _load_api_key(),
    "inference_count": 0,
    "parts_processed": 0,
    "last_result": None,
    "batch_results": None,
    "nameplate_result": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ─── Helper Functions ────────────────────────────────────────────────────────


def get_groq_client():
    """Return an initialised Groq client or None."""
    if st.session_state.api_key:
        return Groq(api_key=st.session_state.api_key)
    return None


def validate_api_key(key: str) -> bool:
    """Quick validation: try a tiny API call."""
    try:
        client = Groq(api_key=key)
        client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True
    except Exception:
        return False


def encode_image_b64(image: Image.Image, max_size: int = 1024) -> str:
    """Resize image if needed and return base64 string."""
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def source_badge(source_type: str) -> str:
    """Return HTML badge for a given source type."""
    mapping = {
        "MEASURED": ("🟢 MEASURED", "measured"),
        "DERIVED": ("🔵 DERIVED", "derived"),
        "INFERRED": ("🟡 INFERRED", "inferred"),
        "ASSUMED": ("🔴 ASSUMED", "assumed"),
    }
    label, cls = mapping.get(source_type.upper(), ("⚪ UNKNOWN", "assumed"))
    return f'<span class="badge badge-{cls}">{label}</span>'


def confidence_bar_html(confidence: int) -> str:
    """Return an HTML micro-progress-bar."""
    if confidence >= 70:
        cls = "confidence-high"
    elif confidence >= 40:
        cls = "confidence-mid"
    else:
        cls = "confidence-low"
    return (
        f'<div class="confidence-bar-container">'
        f'<div class="confidence-bar-fill {cls}" style="width:{confidence}%;"></div>'
        f"</div>"
        f'<span style="font-size:0.78rem;color:#64748B;">{confidence}%</span>'
    )


def build_spec_table_html(specs: list) -> str:
    """Build a full HTML table for the specification list."""
    rows = ""
    for s in specs:
        field = s.get("field", "—")
        value = s.get("value", "—")
        source = s.get("source_type", "ASSUMED")
        conf = int(s.get("confidence", 0))
        reasoning = s.get("reasoning", "")
        rows += (
            "<tr>"
            f'<td class="spec-field">{field}</td>'
            f'<td class="spec-value">{value}</td>'
            f"<td>{source_badge(source)}</td>"
            f"<td>{confidence_bar_html(conf)}</td>"
            f'<td class="spec-reasoning">{reasoning}</td>'
            "</tr>"
        )
    return (
        '<div style="max-height: 400px; overflow-y: auto; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1rem;">'
        '<table class="spec-table"><thead><tr>'
        "<th>Field</th><th>Value</th><th>Source Type</th>"
        "<th>Confidence</th><th>Reasoning</th>"
        "</tr></thead><tbody>"
        f"{rows}</tbody></table>"
        "</div>"
    )


def specs_to_csv_string(result: dict) -> str:
    """Convert result dict to a CSV string for download."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Field", "Value", "Source Type", "Confidence", "Reasoning"])
    for s in result.get("specifications", []):
        writer.writerow([
            s.get("field", ""),
            s.get("value", ""),
            s.get("source_type", ""),
            s.get("confidence", ""),
            s.get("reasoning", ""),
        ])
    return output.getvalue()


# ─── Groq API Prompts ────────────────────────────────────────────────────────

SPEC_SYSTEM_PROMPT = """You are an expert industrial engineer and materials scientist with 30+ years \
of experience in manufacturing, standards (IS/DIN/ANSI/ISO), and reverse \
engineering of industrial components. You specialize in reconstructing complete \
product specifications from partial information using physics-based reasoning \
and engineering domain knowledge.

When given partial product information, you DERIVE specifications from first \
principles — NOT from database lookup. You reason like an experienced engineer \
examining an unknown part.

You must respond ONLY in valid JSON with this exact structure:
{
  "part_summary": "brief description of identified part",
  "overall_confidence": 0-100,
  "specifications": [
    {
      "field": "spec name",
      "value": "spec value with unit",
      "source_type": "MEASURED|DERIVED|INFERRED|ASSUMED",
      "confidence": 0-100,
      "reasoning": "engineering explanation of how this was determined"
    }
  ],
  "reasoning_chain": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "applicable_standards": ["standard 1", "standard 2"],
  "verification_tests": ["test 1", "test 2"],
  "alternative_interpretations": ["if X then Y", "if A then B"],
  "warnings": ["caution 1", "caution 2"]
}"""


def build_user_prompt(part_desc: str, context: str, domain: str, standards: str, vision_text: str | None = None) -> str:
    vision_section = vision_text if vision_text else "No image provided"
    return (
        f"Part Fragments: {part_desc}\n"
        f"Application Context: {context}\n"
        f"Industry Domain: {domain}\n"
        f"Standards Preference: {standards}\n"
        f"Image Analysis Results: {vision_section}\n\n"
        "Reconstruct the complete specification sheet using engineering reasoning. "
        "Derive every possible specification from the given fragments and context. "
        "Be thorough — generate at least 12-15 specification fields."
    )


NAMEPLATE_SYSTEM_PROMPT = (
    "You are an expert at reading industrial nameplates, part markings, "
    "and technical labels. Extract ALL visible text, numbers, symbols, and markings. "
    "Then parse them into structured fields. Return JSON only with this structure:\n"
    "{\n"
    '  "raw_text": "all text seen on the nameplate",\n'
    '  "parsed_fields": [\n'
    "    {\n"
    '      "field": "field name",\n'
    '      "value": "extracted value",\n'
    '      "confidence": 0-100\n'
    "    }\n"
    "  ],\n"
    '  "overall_confidence": 0-100,\n'
    '  "notes": "any observations about condition, legibility, etc."\n'
    "}"
)

# ─── API Call Wrappers ───────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def call_vision(client: Groq, image_b64: str, prompt: str | None = None) -> str:
    """Analyse an image via the vision model and return the raw response text."""
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
        {
            "type": "text",
            "text": prompt or "Describe everything you see on this industrial part or nameplate. Extract all text, numbers, symbols, and markings.",
        },
    ]
    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {"role": "system", "content": NAMEPLATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=4000,
        temperature=0.2,
    )
    return response.choices[0].message.content

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def call_spec_inference(client: Groq, user_prompt: str) -> str:
    """Run the main spec inference and return raw response text."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SPEC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4000,
        temperature=0.3,
    )
    return response.choices[0].message.content


def safe_parse_json(text: str) -> dict | None:
    """Try to parse JSON from the model response, handling markdown fences and think blocks."""
    cleaned = text.strip()
    
    # Remove <think>...</think> blocks using regex
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
    
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON block in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                return None
    return None


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="brand-icon">⚙️</div>'
        "<h2>Dead Reckoning</h2>"
        '<p class="tagline">Spec Inference Engine</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # API Key status (loaded from .env — no input field)
    if st.session_state.api_key:
        st.markdown(
            '<div class="status-connected">✅ Groq API Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-disconnected">❌ API Key Missing</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.78rem;color:#64748B;margin-top:0.5rem;">'
            "Add <code>GROQ_API_KEY</code> to your <code>.env</code> file or "
            "Streamlit secrets to connect.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="stat-box">
            <span class="stat-label">Inferences Run</span>
            <span class="stat-value">{st.session_state.inference_count}</span>
        </div>
        <div class="stat-box">
            <span class="stat-label">Parts Processed</span>
            <span class="stat-value">{st.session_state.parts_processed}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-info">'
        "<strong>What is Dead Reckoning?</strong><br>"
        "An AI engine that reconstructs product specs for industrial parts "
        "where <em>no documentation exists</em>. It uses physics-based reasoning "
        "and engineering domain knowledge to derive specs from first principles."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-footer">'
        "Built for <strong>UniHack 2026</strong><br>Powered by Groq ⚡"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="app-header">'
    '<div class="header-icon">⚙️</div>'
    "<h1>Dead Reckoning</h1>"
    "<p>Spec Inference Engine for Undocumented Industrial Parts — "
    "derive what datasheets can't tell you.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_spec, tab_nameplate, tab_batch, tab_about = st.tabs([
    "🔍 Spec Inference",
    "📷 Nameplate Decoder",
    "🔄 Batch Inference",
    "📊 About / How It Works",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — SPEC INFERENCE
# ═══════════════════════════════════════════════════════════════════════

with tab_spec:

    st.markdown(
        '<div class="spec-card"><h3>🔍 Part Information Input</h3>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        part_desc = st.text_area(
            "Part Description / Fragments",
            placeholder="e.g., partial part number, material type, worn nameplate text",
            height=120,
            key="spec_part_desc",
        )
        app_context = st.text_area(
            "Application Context",
            placeholder="e.g., used in 5HP pump motor coupling, operating temp 80°C",
            height=120,
            key="spec_app_context",
        )

    with col_right:
        industry_domain = st.selectbox(
            "Industry Domain",
            [
                "Pumps & Valves",
                "Motors & Drives",
                "Fasteners & Fittings",
                "Bearings & Couplings",
                "Electrical Components",
                "Hydraulics & Pneumatics",
                "Piping & Flanges",
                "Custom/Other",
            ],
            key="spec_domain",
        )
        standards_pref = st.selectbox(
            "Standards Preference",
            ["Auto-detect", "IS (Indian Standards)", "DIN", "ANSI/ASME", "ISO", "JIS"],
            key="spec_standards",
        )
        uploaded_image = st.file_uploader(
            "Upload Part Photo (optional)",
            type=["jpg", "jpeg", "png"],
            help="Upload worn nameplate, part image, or any visual reference",
            key="spec_image",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Infer button ──
    infer_clicked = st.button("🧠 Infer Specifications", type="primary", use_container_width=True)

    if infer_clicked:
        # Validation
        if not st.session_state.api_key:
            st.markdown(
                '<div class="warning-card">⚠️ <strong>API key required.</strong> '
                "Add <code>GROQ_API_KEY</code> to your <code>.env</code> file and restart the app.</div>",
                unsafe_allow_html=True,
            )
        elif not part_desc.strip() and not uploaded_image:
            st.warning("Please provide at least a part description or upload an image.")
        else:
            client = get_groq_client()
            vision_output = None

            try:
                # Step 1 — Vision analysis (if image provided)
                if uploaded_image:
                    with st.spinner("📷 Analysing uploaded image with vision AI..."):
                        pil_img = Image.open(uploaded_image).convert("RGB")
                        img_b64 = encode_image_b64(pil_img)
                        vision_output = call_vision(client, img_b64)

                # Step 2 — Build prompt
                with st.spinner("🧠 Applying engineering domain reasoning..."):
                    user_prompt = build_user_prompt(
                        part_desc, app_context, industry_domain, standards_pref, vision_output
                    )
                    time.sleep(0.3)  # tiny visual pause for UX

                # Step 3 — Call inference
                with st.spinner("⚙️ Deriving specifications from first principles..."):
                    raw_response = call_spec_inference(client, user_prompt)

                # Step 4 — Parse
                with st.spinner("📐 Cross-referencing industrial standards..."):
                    result = safe_parse_json(raw_response)

                if result is None:
                    st.error("Could not parse the AI response as structured JSON. Raw response below:")
                    st.code(raw_response, language="json")
                else:
                    st.session_state.last_result = result
                    st.session_state.inference_count += 1
                    st.session_state.parts_processed += 1
                    st.rerun()

            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    st.error("⏳ **Rate limit reached.** Groq's free tier has usage limits. Please wait a moment and try again.")
                elif "authentication" in error_msg.lower() or "401" in error_msg:
                    st.error("🔑 **Invalid API key.** Please check your GROQ_API_KEY in the .env file.")
                else:
                    st.error(f"❌ **API Error:** {error_msg}")
                st.button("🔄 Retry", key="retry_spec")

    # ── Display last result ──
    if st.session_state.last_result:
        result = st.session_state.last_result

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        st.markdown(
            '<div class="result-header">'
            "<h2>📋 RECONSTRUCTED SPECIFICATION SHEET</h2>"
            f'<p>{result.get("part_summary", "Industrial Part Analysis")}</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="result-body">', unsafe_allow_html=True)

        # Overall confidence
        overall = int(result.get("overall_confidence", 0))
        st.markdown(
            f'<div class="overall-confidence">'
            f'<div class="label">Overall Confidence Score</div>'
            f'<div class="score">{overall}%</div>'
            f'<div class="overall-bar">'
            f'<div class="overall-bar-fill" style="width:{overall}%;"></div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # Spec table
        specs = result.get("specifications", [])
        if specs:
            st.markdown(build_spec_table_html(specs), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Expandable sections
        if result.get("reasoning_chain"):
            with st.expander("🔗 Engineering Reasoning Chain", expanded=False):
                for step in result["reasoning_chain"]:
                    st.markdown(f"- {step}")

        if result.get("applicable_standards"):
            with st.expander("📏 Applicable Standards", expanded=False):
                for std in result["applicable_standards"]:
                    st.markdown(f"- {std}")

        if result.get("verification_tests"):
            with st.expander("🧪 Recommended Verification Tests", expanded=False):
                for test in result["verification_tests"]:
                    st.markdown(f"- {test}")

        if result.get("alternative_interpretations"):
            with st.expander("🔀 Alternative Interpretations", expanded=False):
                for alt in result["alternative_interpretations"]:
                    st.markdown(f"- {alt}")

        if result.get("warnings"):
            with st.expander("⚠️ Warnings & Cautions", expanded=False):
                for warn in result["warnings"]:
                    st.warning(warn)

        # Download buttons
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        dl_col1, dl_col2, _ = st.columns([1, 1, 2])
        with dl_col1:
            st.download_button(
                "📥 Download as JSON",
                data=json.dumps(result, indent=2),
                file_name="dead_reckoning_specs.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "📥 Download as CSV",
                data=specs_to_csv_string(result),
                file_name="dead_reckoning_specs.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — NAMEPLATE DECODER
# ═══════════════════════════════════════════════════════════════════════

with tab_nameplate:

    st.markdown(
        '<div class="spec-card"><h3>📷 Industrial Nameplate Decoder</h3>'
        "<p style='color:#64748B;font-size:0.9rem;'>"
        "Upload a photo of a worn, damaged, or partially legible nameplate. "
        "The AI will extract and structure all visible information."
        "</p></div>",
        unsafe_allow_html=True,
    )

    np_col_left, np_col_right = st.columns(2, gap="large")

    with np_col_left:
        st.markdown(
            '<div class="decoder-upload">'
            '<p style="font-size:2.5rem;margin:0;">📷</p>'
            '<p style="color:#64748B;font-size:0.9rem;">Upload nameplate image below</p>'
            "</div>",
            unsafe_allow_html=True,
        )
        np_image = st.file_uploader(
            "Choose nameplate image",
            type=["jpg", "jpeg", "png"],
            key="nameplate_upload",
            label_visibility="collapsed",
        )

        if np_image:
            pil_np = Image.open(np_image).convert("RGB")
            st.image(pil_np, caption="Uploaded Nameplate", use_container_width=True)

    with np_col_right:
        decode_clicked = st.button(
            "🔍 Decode Nameplate", type="primary", use_container_width=True, key="btn_decode"
        )

        if decode_clicked:
            if not st.session_state.api_key:
                st.markdown(
                    '<div class="warning-card">⚠️ <strong>API key required.</strong> '
                    "Add <code>GROQ_API_KEY</code> to your <code>.env</code> file.</div>",
                    unsafe_allow_html=True,
                )
            elif not np_image:
                st.warning("Please upload a nameplate image first.")
            else:
                client = get_groq_client()
                try:
                    with st.spinner("🔍 Decoding nameplate with vision AI..."):
                        pil_np = Image.open(np_image).convert("RGB")
                        img_b64 = encode_image_b64(pil_np)
                        raw = call_vision(
                            client,
                            img_b64,
                            "Read this industrial nameplate thoroughly. Extract every piece of text, "
                            "number, symbol, and marking you can see. Parse them into structured fields "
                            "with confidence scores. Return JSON only.",
                        )
                        np_result = safe_parse_json(raw)

                    if np_result is None:
                        st.error("Could not parse structured data. Raw AI output:")
                        st.code(raw, language="json")
                    else:
                        st.session_state.nameplate_result = np_result
                        st.session_state.parts_processed += 1
                        st.rerun()

                except Exception as e:
                    error_msg = str(e)
                    if "rate_limit" in error_msg.lower() or "429" in error_msg:
                        st.error("⏳ **Rate limit reached.** Please wait and retry.")
                    elif "authentication" in error_msg.lower() or "401" in error_msg:
                        st.error("🔑 **Invalid API key.** Check your .env file.")
                    else:
                        st.error(f"❌ **Error:** {error_msg}")

        # Display nameplate results
        if st.session_state.nameplate_result:
            np_res = st.session_state.nameplate_result

            st.markdown(
                '<div class="result-header">'
                "<h2>📋 Decoded Nameplate Data</h2>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="result-body">', unsafe_allow_html=True)

            # Raw text
            if np_res.get("raw_text"):
                st.markdown("**Extracted Raw Text:**")
                st.code(np_res["raw_text"], language="text")

            # Parsed fields table
            fields = np_res.get("parsed_fields", [])
            if fields:
                rows_html = ""
                for f in fields:
                    conf = int(f.get("confidence", 0))
                    rows_html += (
                        f'<tr><td class="spec-field">{f.get("field", "—")}</td>'
                        f'<td class="spec-value">{f.get("value", "—")}</td>'
                        f"<td>{confidence_bar_html(conf)}</td></tr>"
                    )
                st.markdown(
                    '<div style="max-height: 400px; overflow-y: auto; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 1rem;">'
                    '<table class="spec-table"><thead><tr>'
                    "<th>Field</th><th>Value</th><th>Confidence</th>"
                    f"</tr></thead><tbody>{rows_html}</tbody></table>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Overall confidence
            np_conf = int(np_res.get("overall_confidence", 0))
            st.markdown(
                f'<div class="overall-confidence" style="margin-top:1rem;">'
                f'<div class="label">Decoding Confidence</div>'
                f'<div class="score">{np_conf}%</div>'
                f'<div class="overall-bar">'
                f'<div class="overall-bar-fill" style="width:{np_conf}%;"></div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

            if np_res.get("notes"):
                st.info(f"📝 **Notes:** {np_res['notes']}")

            st.markdown("</div>", unsafe_allow_html=True)

            # Download
            st.download_button(
                "📥 Download Decoded Data (JSON)",
                data=json.dumps(np_res, indent=2),
                file_name="nameplate_decoded.json",
                mime="application/json",
                key="dl_nameplate",
            )

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH INFERENCE
# ═══════════════════════════════════════════════════════════════════════

with tab_batch:

    st.markdown(
        '<div class="spec-card"><h3>🔄 Batch Specification Inference</h3>'
        "<p style='color:#64748B;font-size:0.9rem;'>"
        "Upload a CSV with multiple parts to infer specifications in bulk. "
        "Required columns: <code>part_id</code>, <code>description</code>, "
        "<code>context</code>, <code>domain</code>."
        "</p></div>",
        unsafe_allow_html=True,
    )

    batch_csv = st.file_uploader("Upload CSV file", type=["csv"], key="batch_csv")

    if batch_csv:
        try:
            df = pd.read_csv(batch_csv)
            st.markdown("**📋 Uploaded Data Preview:**")
            st.dataframe(df, use_container_width=True)

            required_cols = {"part_id", "description", "context", "domain"}
            if not required_cols.issubset(set(df.columns)):
                missing = required_cols - set(df.columns)
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
            else:
                batch_clicked = st.button(
                    "🚀 Run Batch Inference",
                    type="primary",
                    use_container_width=True,
                    key="btn_batch",
                )

                if batch_clicked:
                    if not st.session_state.api_key:
                        st.markdown(
                            '<div class="warning-card">⚠️ <strong>API key required.</strong> '
                            "Add <code>GROQ_API_KEY</code> to your <code>.env</code> file.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        client = get_groq_client()
                        all_results = []
                        progress = st.progress(0, text="Starting batch inference...")
                        total = len(df)
                        errors = 0

                        for idx, row in df.iterrows():
                            progress.progress(
                                (idx) / total,
                                text=f"⚙️ Processing part {idx + 1}/{total}: {row['part_id']}",
                            )
                            try:
                                user_prompt = build_user_prompt(
                                    str(row["description"]),
                                    str(row["context"]),
                                    str(row["domain"]),
                                    "Auto-detect",
                                )
                                raw = call_spec_inference(client, user_prompt)
                                parsed = safe_parse_json(raw)
                                if parsed:
                                    parsed["part_id"] = row["part_id"]
                                    all_results.append(parsed)
                                else:
                                    all_results.append({
                                        "part_id": row["part_id"],
                                        "error": "JSON parse failed",
                                        "raw": raw[:500],
                                    })
                                    errors += 1

                                # Small delay to respect rate limits
                                time.sleep(1.0)

                            except Exception as e:
                                all_results.append({
                                    "part_id": row["part_id"],
                                    "error": str(e),
                                })
                                errors += 1

                        progress.progress(1.0, text="✅ Batch inference complete!")
                        st.session_state.batch_results = all_results
                        st.session_state.inference_count += total - errors
                        st.session_state.parts_processed += total - errors

                        if errors:
                            st.warning(f"⚠️ {errors}/{total} parts encountered errors.")

                        st.rerun()

        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")

    # Display batch results
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-header">'
            f"<h2>📊 Batch Results — {len(results)} Parts</h2>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="result-body">', unsafe_allow_html=True)

        # Summary table
        summary_rows = []
        for r in results:
            if "error" in r:
                summary_rows.append({
                    "Part ID": r.get("part_id", "—"),
                    "Summary": f"❌ Error: {r['error'][:80]}",
                    "Confidence": "—",
                    "Specs Count": "—",
                })
            else:
                summary_rows.append({
                    "Part ID": r.get("part_id", "—"),
                    "Summary": r.get("part_summary", "—"),
                    "Confidence": f'{r.get("overall_confidence", 0)}%',
                    "Specs Count": len(r.get("specifications", [])),
                })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Expand individual results
        for r in results:
            if "error" not in r:
                with st.expander(f"📋 {r.get('part_id', 'Unknown')} — {r.get('part_summary', '')}", expanded=False):
                    specs = r.get("specifications", [])
                    if specs:
                        st.markdown(build_spec_table_html(specs), unsafe_allow_html=True)

        # Download full results
        st.download_button(
            "📥 Download All Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name="batch_results.json",
            mime="application/json",
            key="dl_batch_json",
            use_container_width=True,
        )

        # Build flat CSV for batch
        flat_rows = []
        for r in results:
            pid = r.get("part_id", "")
            for s in r.get("specifications", []):
                flat_rows.append({
                    "part_id": pid,
                    "field": s.get("field", ""),
                    "value": s.get("value", ""),
                    "source_type": s.get("source_type", ""),
                    "confidence": s.get("confidence", ""),
                    "reasoning": s.get("reasoning", ""),
                })
        if flat_rows:
            batch_df = pd.DataFrame(flat_rows)
            st.download_button(
                "📥 Download All Results (CSV)",
                data=batch_df.to_csv(index=False),
                file_name="batch_results.csv",
                mime="text/csv",
                key="dl_batch_csv",
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT / HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════

with tab_about:

    st.markdown(
        '<div class="spec-card"><h3>📊 How Dead Reckoning Works</h3>'
        "<p style='color:#64748B;font-size:0.9rem;'>"
        "A five-stage AI pipeline that reconstructs specifications from fragments.</p></div>",
        unsafe_allow_html=True,
    )

    # ── Flow Steps ──
    flow_cols = st.columns(5, gap="medium")
    steps = [
        ("📥", "Step 1", "Input Fragments", "User provides partial part info, application context, and optional images."),
        ("📷", "Step 2", "Vision Analysis", "AI reads nameplates, markings, and part geometry from uploaded photos."),
        ("🧠", "Step 3", "Domain Reasoning", "LLM applies decades of engineering expertise to interpret fragments."),
        ("⚙️", "Step 4", "Spec Derivation", "Physics-based inference reconstructs the full specification sheet."),
        ("📊", "Step 5", "Confidence Scoring", "Each field is rated by derivation method and certainty level."),
    ]
    for col, (icon, num, title, desc) in zip(flow_cols, steps):
        with col:
            st.markdown(
                f'<div class="flow-step">'
                f'<div class="step-icon">{icon}</div>'
                f'<div class="step-num">{num}</div>'
                f"<h4>{title}</h4>"
                f"<p>{desc}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Comparison Table ──
    st.markdown(
        '<div class="spec-card"><h3>⚖️ Dead Reckoning vs Traditional Tools</h3></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>⚙️ Dead Reckoning</th>
                    <th>🔎 Traditional Lookup</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Data Source</td>
                    <td>Derives from first principles</td>
                    <td>Requires existing database/catalog</td>
                </tr>
                <tr>
                    <td>Undocumented Parts</td>
                    <td>✅ Core capability</td>
                    <td>❌ Cannot handle</td>
                </tr>
                <tr>
                    <td>Damaged Nameplates</td>
                    <td>✅ Vision AI + domain reasoning</td>
                    <td>❌ Needs legible text</td>
                </tr>
                <tr>
                    <td>Confidence Tracking</td>
                    <td>✅ Per-field with reasoning chain</td>
                    <td>❌ Binary match/no-match</td>
                </tr>
                <tr>
                    <td>Standards Mapping</td>
                    <td>✅ IS / DIN / ANSI / ISO / JIS</td>
                    <td>⚠️ Single standard only</td>
                </tr>
                <tr>
                    <td>Engineering Reasoning</td>
                    <td>✅ Step-by-step derivation chain</td>
                    <td>❌ No reasoning</td>
                </tr>
                <tr>
                    <td>Verification Guidance</td>
                    <td>✅ Suggested tests & checks</td>
                    <td>❌ Not provided</td>
                </tr>
                <tr>
                    <td>Batch Processing</td>
                    <td>✅ CSV upload + bulk inference</td>
                    <td>⚠️ One-at-a-time</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Use Case Examples ──
    st.markdown(
        '<div class="spec-card"><h3>🔬 Example Use Cases</h3></div>',
        unsafe_allow_html=True,
    )

    uc_col1, uc_col2 = st.columns(2, gap="large")

    with uc_col1:
        st.markdown(
            '<div class="spec-card">'
            '<h3 style="color:#D97706;">🔧 Before — The Problem</h3>'
            '<p style="color:#64748B;font-size:0.88rem;line-height:1.6;">'
            "A maintenance engineer finds a corroded coupling in a legacy pump system. "
            "The nameplate is illegible. No manuals exist. The plant was built in 1987. "
            "Ordering a replacement requires exact specs — material, bore size, RPM rating, "
            "torque capacity — none of which are available."
            "</p></div>",
            unsafe_allow_html=True,
        )

    with uc_col2:
        st.markdown(
            '<div class="spec-card" style="border-color:#059669;">'
            '<h3 style="color:#059669;">✅ After — Dead Reckoning</h3>'
            '<p style="color:#64748B;font-size:0.88rem;line-height:1.6;">'
            "The engineer uploads a photo and enters: <em>\"Jaw coupling, connects 5HP motor "
            "to centrifugal pump, ~1450 RPM, appears to be cast iron.\"</em> "
            "Dead Reckoning derives 15+ specifications including bore range, torque capacity, "
            "material grade (IS 210 Grade FG200), keyway dimensions, and applicable IS standards — "
            "all with confidence scores and reasoning chains."
            "</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    uc_col3, uc_col4 = st.columns(2, gap="large")

    with uc_col3:
        st.markdown(
            '<div class="spec-card">'
            '<h3 style="color:#D97706;">🔧 Before — Electrical Component</h3>'
            '<p style="color:#64748B;font-size:0.88rem;line-height:1.6;">'
            "A plant electrician needs to replace a damaged contactor. Only fragments remain: "
            '"LC1-D??A, 3-pole, used on a 15kW motor starter panel." '
            "The exact model suffix is unreadable."
            "</p></div>",
            unsafe_allow_html=True,
        )

    with uc_col4:
        st.markdown(
            '<div class="spec-card" style="border-color:#059669;">'
            '<h3 style="color:#059669;">✅ After — Dead Reckoning</h3>'
            '<p style="color:#64748B;font-size:0.88rem;line-height:1.6;">'
            "Dead Reckoning infers: LC1-D32A (32A contactor), coil voltage likely 240V AC, "
            "thermal rating 15kW at 415V, auxiliary contacts 1NO+1NC, DIN rail mount. "
            "Derives compatible overload relay range (23-32A) and suggests verification steps."
            "</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Source Type Legend ──
    st.markdown(
        '<div class="spec-card"><h3>🏷️ Source Type Legend</h3></div>',
        unsafe_allow_html=True,
    )

    legend_cols = st.columns(4, gap="medium")
    legend_items = [
        ("🟢 MEASURED", "measured", "Direct measurement from image analysis or explicit user input."),
        ("🔵 DERIVED", "derived", "Mathematically calculated from known values using engineering formulas."),
        ("🟡 INFERRED", "inferred", "Determined using engineering domain knowledge and contextual reasoning."),
        ("🔴 ASSUMED", "assumed", "Standard default value applied when no specific data is available."),
    ]
    for col, (label, cls, desc) in zip(legend_cols, legend_items):
        with col:
            st.markdown(
                f'<div class="flow-step">'
                f'<span class="badge badge-{cls}" style="font-size:0.85rem;">{label}</span>'
                f'<p style="margin-top:0.6rem;">{desc}</p></div>',
                unsafe_allow_html=True,
            )
