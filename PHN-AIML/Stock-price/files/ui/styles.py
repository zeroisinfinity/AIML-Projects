# =============================================================================
# ui/styles.py  —  Global CSS injection
# Call inject_styles() once at the top of app.py.
# Isolated here so visual tweaks never touch logic files.
# =============================================================================

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Global font — NEVER target div/span/[class*=css] ────────────────────────
   Streamlit renders its collapse-arrow as a Unicode glyph in a <span> using
   a Material-Icons font.  Overriding font-family on all <span> elements
   replaces that glyph with its literal ligature name ("arrow_drop_down").
   We scope overrides to semantic text tags only. */
html, body, .stMarkdown, p, label,
.stTextInput input, .stSelectbox select,
button[kind], .stSlider, .stDateInput {
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ────────────────────────────────────────────────────────── */
.stApp { background: #080c14; }

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b101c 0%, #080c14 100%);
    border-right: 1px solid #1a2535;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stDateInput label {
    color: #8b949e !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #0d1927 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 0.875rem !important;
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0d1927 0%, #0a1116 100%);
    border: 1px solid #1e2d40;
    border-radius: 14px;
    padding: 18px 20px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: #2d4a6a;
    box-shadow: 0 10px 30px rgba(88,166,255,0.1);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.85rem !important; font-weight: 800 !important;
    color: #58a6ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important; color: #8b949e !important;
    text-transform: uppercase !important; letter-spacing: 0.09em !important;
    font-weight: 700 !important;
}

/* ── Section headers ───────────────────────────────────────────────────────── */
.sec-hdr {
    font-size: 1.1rem; font-weight: 700; color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding: 4px 0 4px 12px; margin: 26px 0 14px;
    font-family: 'Inter', sans-serif; letter-spacing: -0.01em;
}

/* ── Info / math cards ─────────────────────────────────────────────────────── */
.info-card {
    background: linear-gradient(135deg, #0d1927 0%, #080e18 100%);
    border-left: 3px solid #58a6ff;
    border-top: 1px solid #1e2d40; border-right: 1px solid #1e2d40;
    border-bottom: 1px solid #1e2d40;
    border-radius: 0 10px 10px 0;
    padding: 13px 16px; margin: 8px 0;
    font-size: 0.875rem; color: #c9d1d9; line-height: 1.65;
}
.info-card b { color: #79c0ff; }
.info-card code {
    background: #1a2a3a; color: #79c0ff;
    padding: 1px 5px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}

/* ── Expander — prevent arrow-glyph / title collision ─────────────────────
   Layout the summary as flex so the SVG arrow and the <p> label are
   always separated by a gap.  We override font ONLY on <p>, never on
   <span>, which is where the Material-Icons glyph lives. */
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 12px 16px !important;
    background: linear-gradient(135deg,#0d1927,#080e18) !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 10px !important;
    cursor: pointer !important;
}
[data-testid="stExpander"] summary svg {
    flex-shrink: 0 !important;
    width: 16px !important; height: 16px !important;
    color: #58a6ff !important;
}
[data-testid="stExpander"] summary p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #c9d1d9 !important;
    margin: 0 !important;
}
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
    border: 1px solid #1e2d40 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: #080c14 !important;
    padding: 16px !important;
}

/* ── Divider ───────────────────────────────────────────────────────────────── */
.div-line { border: none; border-top: 1px solid #1a2535; margin: 18px 0; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
