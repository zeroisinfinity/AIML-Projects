# =============================================================================
# ui/sidebar.py  —  Sidebar layout and widget state
# render_sidebar() returns a plain dict — app.py reads it, no sidebar logic
# leaks into data or model layers.
# =============================================================================

import streamlit as st
import pandas as pd
from config import STOCKS, ALGORITHMS


def render_sidebar() -> dict:
    """
    Render sidebar and return a config dict with keys:
        ticker, start_date, algo, train_split, conf_thresh, n_estimators
    """
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:22px 0 16px;">
            <div style="font-size:2.4rem;line-height:1;">📈</div>
            <div style="font-size:1.3rem;font-weight:900;color:#e6edf3;
                        letter-spacing:0.06em;margin-top:6px;">QuantML</div>
            <div style="font-size:0.65rem;color:#58a6ff;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.18em;margin-top:2px;">
                Stock Direction Predictor</div>
        </div>
        """, unsafe_allow_html=True)

        # Live badge
        st.markdown("""
        <div style="background:linear-gradient(90deg,#0d2137,#091520);
                    border:1px solid #1e3a5f;border-radius:8px;
                    padding:10px 14px;margin-bottom:16px;text-align:center;">
            <span style="font-size:0.7rem;color:#58a6ff;font-weight:600;
                         text-transform:uppercase;letter-spacing:0.1em;">
                Live Data · Refreshes every 60 s
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Asset selection ───────────────────────────────────────────────────
        _label("Asset Selection", "#58a6ff")
        category     = st.selectbox("Category", list(STOCKS.keys()),
                                    label_visibility="collapsed")
        stock_opts   = STOCKS[category]
        selected_lbl = st.selectbox("Select Asset", list(stock_opts.keys()),
                                    label_visibility="collapsed")
        ticker       = stock_opts[selected_lbl]

        custom = st.text_input("Or type any ticker",
                               placeholder="e.g. BABA, RIVN, DOGE-USD")
        if custom.strip():
            ticker = custom.strip().upper()

        _divider()

        # ── Date ──────────────────────────────────────────────────────────────
        _label("Date Range", "#58a6ff")
        start_date = st.date_input("Start", value=pd.to_datetime("2015-01-01"),
                                   label_visibility="collapsed")

        _divider()

        # ── Model ─────────────────────────────────────────────────────────────
        _label("Model Settings", "#a371f7")
        algo         = st.selectbox("Algorithm", ALGORITHMS,
                                    help="XGBoost is usually fastest & most precise.")
        train_split  = st.slider("Training Data %",     60, 90, 80, 5)
        conf_thresh  = st.slider("Confidence Threshold %", 50, 75, 55)
        n_estimators = st.slider("Trees / Rounds",     100, 500, 200, 50)

        _divider()

        st.markdown("""
        <div style="background:#120a02;border:1px solid #3a2000;border-radius:8px;
                    padding:10px 13px;font-size:0.7rem;color:#8b949e;line-height:1.6;">
            <b style="color:#ffa657;">Educational only.</b><br>
            Not financial advice. Past performance does not guarantee future results.
        </div>
        """, unsafe_allow_html=True)

    return dict(
        ticker=ticker,
        start_date=str(start_date),
        algo=algo,
        train_split=train_split,
        conf_thresh=conf_thresh,
        n_estimators=n_estimators,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label(text: str, color: str) -> None:
    st.markdown(f"""
    <div style="font-size:0.65rem;font-weight:700;color:{color};
                text-transform:uppercase;letter-spacing:0.12em;
                margin-bottom:6px;">{text}</div>
    """, unsafe_allow_html=True)


def _divider() -> None:
    st.markdown('<hr class="div-line">', unsafe_allow_html=True)
