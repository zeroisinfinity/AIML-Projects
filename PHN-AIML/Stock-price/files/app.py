# =============================================================================
# app.py  —  QuantML entry point
# Run with:  streamlit run app.py
#
# This file is intentionally thin — it only orchestrates.
# All logic lives in the modules below:
#
#   config.py            Constants, feature registry, stock universe
#   data/loader.py       yfinance fetch, MultiIndex fix, live quote
#   data/features.py     Technical indicator computation
#   models/trainer.py    XGBoost / RF / GBM training pipeline
#   models/evaluator.py  Backtest metrics, Sharpe, drawdown
#   ui/styles.py         CSS injection
#   ui/sidebar.py        Sidebar widgets → config dict
#   ui/charts.py         Plotly figure builders
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # ensure local imports work

import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="QuantML · Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Module imports ────────────────────────────────────────────────────────────
from ui.styles  import inject_styles
from ui.sidebar import render_sidebar
from ui         import charts
from data.loader   import fetch_ohlcv, get_live_quote
from data.features import build_features
from models.trainer   import train
from models.evaluator import evaluate

# ── Styles ────────────────────────────────────────────────────────────────────
inject_styles()

# ── Sidebar → config ──────────────────────────────────────────────────────────
cfg = render_sidebar()
ticker      = cfg['ticker']
start_date  = cfg['start_date']
algo        = cfg['algo']
train_split = cfg['train_split']
conf_thresh = cfg['conf_thresh']
n_est       = cfg['n_estimators']

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:12px 0 6px;">
  <div style="font-size:0.72rem;font-weight:700;color:#58a6ff;
              text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px;">
      Live · Real-Time Data via yfinance &nbsp;|&nbsp; {algo}
  </div>
  <div style="font-size:2.4rem;font-weight:900;line-height:1.1;letter-spacing:-0.03em;
              background:linear-gradient(90deg,#e6edf3 20%,#58a6ff 55%,#a371f7 100%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      QuantML · Stock Direction Predictor
  </div>
  <div style="font-size:0.9rem;color:#8b949e;margin-top:8px;font-weight:400;
              display:flex;gap:16px;flex-wrap:wrap;">
      <span>Log Returns</span><span>RSI</span><span>MACD</span>
      <span>Bollinger Bands</span><span>Momentum</span>
  </div>
</div>
<hr class="div-line">
""", unsafe_allow_html=True)

# ── Live quote badge ──────────────────────────────────────────────────────────
quote = get_live_quote(ticker)
last  = quote.get('last_price') or 0
prev  = quote.get('prev_close') or last
chg   = (last - prev) / prev * 100 if prev else 0
col   = "#3fb950" if chg >= 0 else "#f85149"
arr   = "▲" if chg >= 0 else "▼"

st.markdown(f"""
<div style="display:flex;gap:0;align-items:stretch;
            background:linear-gradient(135deg,#0d1927,#080e18);
            border:1px solid #1e2d40;border-radius:14px;
            overflow:hidden;margin-bottom:22px;">
  <div style="padding:16px 24px;border-right:1px solid #1e2d40;">
    <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:4px;">Ticker</div>
    <div style="font-size:1.7rem;font-weight:900;color:#e6edf3;
                font-family:'JetBrains Mono',monospace;">{ticker}</div>
  </div>
  <div style="padding:16px 24px;border-right:1px solid #1e2d40;">
    <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:4px;">Live Price</div>
    <div style="font-size:1.5rem;font-weight:800;color:#58a6ff;
                font-family:'JetBrains Mono',monospace;">${last:,.2f}</div>
  </div>
  <div style="padding:16px 24px;border-right:1px solid #1e2d40;">
    <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:4px;">Day Change</div>
    <div style="font-size:1.5rem;font-weight:800;color:{col};
                font-family:'JetBrains Mono',monospace;">{arr} {abs(chg):.2f}%</div>
  </div>
  <div style="padding:16px 24px;border-right:1px solid #1e2d40;">
    <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:4px;">Algorithm</div>
    <div style="font-size:1rem;font-weight:700;color:#a371f7;margin-top:6px;">{algo}</div>
  </div>
  <div style="padding:16px 24px;">
    <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:4px;">Confidence Filter</div>
    <div style="font-size:1.5rem;font-weight:800;color:#ffa657;
                font-family:'JetBrains Mono',monospace;">&gt; {conf_thresh}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load + feature-engineer data ──────────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker}…"):
    raw = fetch_ohlcv(ticker, start_date)

if raw is None:
    st.error(f"Could not fetch data for **{ticker}**. Check the ticker and try again.")
    st.stop()

df = build_features(raw)

# ── Chart 1: price history ────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Price History</div>', unsafe_allow_html=True)
st.plotly_chart(charts.price_chart(df, ticker), use_container_width=True)

# ── Chart 2: RSI + MACD (collapsible) ────────────────────────────────────────
with st.expander("RSI & MACD Technical Indicators", expanded=False):
    st.plotly_chart(charts.rsi_macd_chart(df), use_container_width=True)

# ── Train model ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Model Training & Evaluation</div>',
            unsafe_allow_html=True)
with st.spinner(f"Training {algo} on {len(df):,} samples…"):
    result = train(df, train_split, n_est, algo)

ev = evaluate(result, df, conf_thresh)

# ── Metrics row 1 ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision",    f"{ev.precision:.2%}", help="Correct % when model says UP")
m2.metric("Accuracy",     f"{ev.accuracy:.2%}",  help="Overall correct predictions")
m3.metric("Recall",       f"{ev.recall:.2%}",    help="% of actual UP days caught")
m4.metric("Trades Found", f"{ev.n_trades}",      help=f"Days above {conf_thresh}% confidence")
st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics row 2 ─────────────────────────────────────────────────────────────
m5, m6, m7, m8 = st.columns(4)
m5.metric("Strat Return",   f"{ev.ret_strategy:.2%}", delta=f"Mkt {ev.ret_market:.2%}")
m6.metric("Sharpe Ratio",   f"{ev.sharpe:.2f}",       help=">1 good · >2 great")
m7.metric("Max Drawdown",   f"{ev.max_drawdown:.2%}", help="Worst peak-to-trough loss")
m8.metric("Outperformance", f"{ev.ret_strategy - ev.ret_market:.2%}",
          help="Strategy minus Buy & Hold")

# ── Chart 3: equity curve ─────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Backtest: Growth of $1</div>',
            unsafe_allow_html=True)
st.plotly_chart(
    charts.equity_curve(ev.test_df, ev.cum_strat, ev.cum_market,
                        ev.ret_strategy, ev.ret_market),
    use_container_width=True,
)

# ── Chart 4: feature importance ───────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Signal Logic: Feature Importance</div>',
            unsafe_allow_html=True)
st.plotly_chart(charts.feature_importance(result.model), use_container_width=True)

# ── Charts 5 & 6: confidence histogram + monthly heatmap ─────────────────────
col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="sec-hdr" style="margin-top:8px;">Confidence Distribution</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        charts.confidence_histogram(ev.probas, result.y_test, conf_thresh),
        use_container_width=True,
    )
with col_b:
    st.markdown('<div class="sec-hdr" style="margin-top:8px;">Monthly P&L Heatmap</div>',
                unsafe_allow_html=True)
    st.plotly_chart(charts.monthly_heatmap(ev.test_df), use_container_width=True)

# ── Math explainer ────────────────────────────────────────────────────────────
st.markdown('<hr class="div-line">', unsafe_allow_html=True)
with st.expander("Math-Proof Logic Behind the Model", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="info-card"><b>① Stationarity — Log Returns</b><br>
Raw prices trend exponentially, violating ML assumptions.<br>
<code>r_t = ln(P_t / P_{t-1})</code><br>
Log returns are mean-reverting, time-additive, and centered near 0.</div>

<div class="info-card"><b>② RSI — Bounded Momentum Oscillator</b><br>
<code>RSI = 100 – 100 / (1 + AvgGain / AvgLoss)</code><br>
Bounded [0–100]. Overbought >70 → pullback. Oversold <30 → bounce.</div>

<div class="info-card"><b>③ MACD — Trend & Momentum</b><br>
<code>MACD = EMA(12) – EMA(26)</code> | Signal = EMA(9) of MACD<br>
Crossovers flag regime changes. Histogram = momentum strength.</div>

<div class="info-card"><b>④ Bollinger Bands — Volatility Context</b><br>
<code>BB_Width = (Upper – Lower) / SMA(20)</code><br>
Wide = high volatility. %B shows price position in bands [0–1].</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="info-card"><b>⑤ No Look-Ahead Bias</b><br>
Target = <code>(Close[t+1] > Close[t])</code> via <code>.shift(-1)</code><br>
Train/test split is strictly chronological — no random shuffling.</div>

<div class="info-card"><b>⑥ XGBoost — Regularised Gradient Boosting</b><br>
Obj = Σ L(y, ŷ) + Σ Ω(tree)<br>
Adds L1/L2 penalties to each tree. Faster & higher precision than RF.</div>

<div class="info-card"><b>⑦ Confidence Threshold → Precision Filter</b><br>
Trade only when <code>P(UP|X) > threshold</code><br>
Fewer trades, higher win rate. Precision = TP / (TP + FP).</div>

<div class="info-card"><b>⑧ Sharpe & Max Drawdown</b><br>
<code>Sharpe = (μ_daily / σ_daily) × √252</code><br>
Max drawdown = worst peak-to-trough loss on equity curve.</div>
""", unsafe_allow_html=True)
