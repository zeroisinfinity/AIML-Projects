import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# STOCK UNIVERSE  (real-time via yfinance)
# ══════════════════════════════════════════════════════════════════════════════
STOCKS = {
    "🇺🇸 US Large-Cap": {
        "SPY – S&P 500 ETF":    "SPY",
        "QQQ – Nasdaq-100 ETF": "QQQ",
        "AAPL – Apple":         "AAPL",
        "MSFT – Microsoft":     "MSFT",
        "GOOGL – Alphabet":     "GOOGL",
        "AMZN – Amazon":        "AMZN",
        "NVDA – NVIDIA":        "NVDA",
        "META – Meta":          "META",
        "TSLA – Tesla":         "TSLA",
        "BRK-B – Berkshire B":  "BRK-B",
    },
    "📈 Growth & Tech": {
        "AMD – AMD":            "AMD",
        "NFLX – Netflix":       "NFLX",
        "CRM – Salesforce":     "CRM",
        "PLTR – Palantir":      "PLTR",
        "SNOW – Snowflake":     "SNOW",
        "UBER – Uber":          "UBER",
        "SHOP – Shopify":       "SHOP",
    },
    "🏦 Finance & Value": {
        "JPM – JPMorgan":        "JPM",
        "BAC – Bank of America": "BAC",
        "GS – Goldman Sachs":    "GS",
        "V – Visa":              "V",
        "MA – Mastercard":       "MA",
        "WMT – Walmart":         "WMT",
    },
    "🌍 Global ETFs": {
        "EEM – Emerging Mkts":   "EEM",
        "EFA – Developed Mkts":  "EFA",
        "FXI – China Large-Cap": "FXI",
        "INDA – India ETF":      "INDA",
        "EWJ – Japan ETF":       "EWJ",
    },
    "₿ Crypto": {
        "BTC-USD – Bitcoin":  "BTC-USD",
        "ETH-USD – Ethereum": "ETH-USD",
        "SOL-USD – Solana":   "SOL-USD",
        "BNB-USD – BNB":      "BNB-USD",
    },
    "🛢 Commodities": {
        "GLD – Gold ETF":     "GLD",
        "USO – Oil ETF":      "USO",
        "SLV – Silver ETF":   "SLV",
        "PDBC – Commodities": "PDBC",
    },
}

FEATURES = ['Log_Returns', 'RSI', 'Volatility', 'MACD', 'MACD_Signal',
            'BB_Width', 'BB_pct', 'Momentum_5d']
FEATURE_LABELS = {
    'Log_Returns':  'Log Returns',
    'RSI':          'RSI (14d)',
    'Volatility':   'Volatility (21d)',
    'MACD':         'MACD Line',
    'MACD_Signal':  'MACD Signal',
    'BB_Width':     'BB Width',
    'BB_pct':       'BB %B',
    'Momentum_5d':  '5d Momentum',
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantML · Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Target text elements only — never touch SVG/icon elements or Streamlit internal
   classes that use glyph fonts. The old rule `[class*="css"], div, span` was
   overriding Streamlit's Material-icon font, causing the arrow glyph to render
   as the literal text "arrow_down". */
html, body, .stMarkdown, p, label,
.stTextInput input, .stSelectbox select,
button[kind], .stSlider, .stDateInput {
    font-family: 'Inter', sans-serif !important;
}
/* Sidebar text — scoped so we never bleed into icon nodes */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    font-family: 'Inter', sans-serif !important;
}

.stApp { background: #080c14; }

/* ── Sidebar ──────────────────────────────────────────────── */
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

/* Selectbox dropdown */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #0d1927 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 0.875rem !important;
}

/* Metric cards */
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

/* Section headers */
.sec-hdr {
    font-size: 1.1rem; font-weight: 700; color: #e6edf3;
    border-left: 3px solid #58a6ff;
    padding: 4px 0 4px 12px; margin: 26px 0 14px;
    font-family: 'Inter', sans-serif; letter-spacing: -0.01em;
}

/* Info cards */
.info-card {
    background: linear-gradient(135deg, #0d1927 0%, #080e18 100%);
    border-left: 3px solid #58a6ff;
    border-top: 1px solid #1e2d40;
    border-right: 1px solid #1e2d40;
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

/* ── Expander: prevent arrow glyph / title collision ──────── */
/* The summary row is a flex container; gap ensures the arrow
   and the label text are always separated. */
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
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
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

.div-line { border: none; border-top: 1px solid #1a2535; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Branding
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

    st.markdown("""
    <div style="background:linear-gradient(90deg,#0d2137,#091520);
                border:1px solid #1e3a5f;border-radius:8px;
                padding:10px 14px;margin-bottom:16px;text-align:center;">
        <span style="font-size:0.7rem;color:#58a6ff;font-weight:600;
                     text-transform:uppercase;letter-spacing:0.1em;">
            🟢 Live Data · Refreshes every 5 min
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Asset Selection ───────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem;font-weight:700;color:#58a6ff;
                text-transform:uppercase;letter-spacing:0.12em;
                margin-bottom:6px;">🔍 Asset Selection</div>
    """, unsafe_allow_html=True)

    category = st.selectbox("Category", list(STOCKS.keys()),
                            label_visibility="collapsed")

    stock_opts    = STOCKS[category]
    selected_lbl  = st.selectbox("Select Asset", list(stock_opts.keys()),
                                 label_visibility="collapsed")
    ticker = stock_opts[selected_lbl]

    custom = st.text_input("✏️ Or type any ticker",
                           placeholder="e.g. BABA, RIVN, DOGE-USD",
                           label_visibility="visible")
    if custom.strip():
        ticker = custom.strip().upper()

    st.markdown('<hr class="div-line">', unsafe_allow_html=True)

    # ── Date ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem;font-weight:700;color:#58a6ff;
                text-transform:uppercase;letter-spacing:0.12em;
                margin-bottom:6px;">📅 Date Range</div>
    """, unsafe_allow_html=True)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"),
                               label_visibility="collapsed")

    st.markdown('<hr class="div-line">', unsafe_allow_html=True)

    # ── Model ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:0.65rem;font-weight:700;color:#a371f7;
                text-transform:uppercase;letter-spacing:0.12em;
                margin-bottom:6px;">🤖 Model Settings</div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting"],
        help="Gradient Boosting is slower but often higher precision."
    )
    train_split  = st.slider("Training Data %", 60, 90, 80, 5)
    conf_thresh  = st.slider("Confidence Threshold %", 50, 75, 55)
    n_estimators = st.slider("Number of Trees", 100, 400, 200, 50)

    st.markdown('<hr class="div-line">', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#120a02;border:1px solid #3a2000;border-radius:8px;
                padding:10px 13px;font-size:0.7rem;color:#8b949e;line-height:1.6;">
        ⚠️ <b style="color:#ffa657;">Educational only.</b><br>
        Not financial advice. Past performance ≠ future results.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="padding:12px 0 6px;">
    <div style="font-size:0.72rem;font-weight:700;color:#58a6ff;
                text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px;">
        Live · Real-Time Data via yfinance
    </div>
    <div style="font-size:2.4rem;font-weight:900;line-height:1.1;letter-spacing:-0.03em;
                background:linear-gradient(90deg,#e6edf3 20%,#58a6ff 55%,#a371f7 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        QuantML · Stock Direction Predictor
    </div>
    <div style="font-size:0.9rem;color:#8b949e;margin-top:8px;font-weight:400;
                display:flex;gap:16px;flex-wrap:wrap;">
        <span>📐 <span style="color:#58a6ff;font-weight:600;">Log Returns</span></span>
        <span>📊 <span style="color:#a371f7;font-weight:600;">RSI</span></span>
        <span>📉 <span style="color:#3fb950;font-weight:600;">MACD</span></span>
        <span>🎯 <span style="color:#ffa657;font-weight:600;">Bollinger Bands</span></span>
        <span>🚀 <span style="color:#f78166;font-weight:600;">Momentum</span></span>
    </div>
</div>
<hr class="div-line">
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_and_process_data(symbol, start):
    df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
    if df.empty:
        return None
    # Fix yfinance ≥0.2 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).normalize()   # strip intraday timestamps

    df['Log_Returns']  = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility']   = df['Log_Returns'].rolling(21).std()

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df['BB_Width']    = (upper - lower) / sma20
    df['BB_pct']      = (df['Close'] - lower) / (upper - lower)
    df['Momentum_5d'] = df['Close'].pct_change(5)

    df['SMA_50']  = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Target']  = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# ML ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def train_model(df, split_pct, n_est, algo):
    X = df[FEATURES]; y = df['Target']
    idx = int(len(df) * split_pct / 100)
    X_tr, X_te = X.iloc[:idx], X.iloc[idx:]
    y_tr, y_te = y.iloc[:idx], y.iloc[idx:]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    if algo == "Random Forest":
        m = RandomForestClassifier(n_estimators=n_est, min_samples_split=50,
                                   max_features='sqrt', random_state=42, n_jobs=-1)
    else:
        m = GradientBoostingClassifier(n_estimators=n_est, learning_rate=0.05,
                                       max_depth=4, random_state=42)
    m.fit(X_tr_s, y_tr)
    return m, X_te_s, y_te, sc


# ── Shared Plotly layout ──────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor='#080c14', plot_bgcolor='#0c1220',
    font=dict(family='Inter', color='#c9d1d9', size=12),
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode='x unified',
)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner(f"⏳ Fetching real-time data for **{ticker}**…"):
    df = load_and_process_data(ticker, start_date)

if df is None:
    st.error(f"⚠️ Could not fetch data for **{ticker}**. Check the ticker and try again.")
    st.stop()

latest  = float(df['Close'].iloc[-1])
prev    = float(df['Close'].iloc[-2])
chg     = (latest - prev) / prev * 100
chg_col = "#3fb950" if chg >= 0 else "#f85149"
arr     = "▲" if chg >= 0 else "▼"

# ── Ticker stats bar ─────────────────────────────────────────────────────────
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
                    letter-spacing:0.1em;margin-bottom:4px;">Last Close</div>
        <div style="font-size:1.5rem;font-weight:800;color:#58a6ff;
                    font-family:'JetBrains Mono',monospace;">${latest:,.2f}</div>
    </div>
    <div style="padding:16px 24px;border-right:1px solid #1e2d40;">
        <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:4px;">Day Change</div>
        <div style="font-size:1.5rem;font-weight:800;color:{chg_col};
                    font-family:'JetBrains Mono',monospace;">{arr} {abs(chg):.2f}%</div>
    </div>
    <div style="padding:16px 24px;border-right:1px solid #1e2d40;">
        <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:4px;">Data Points</div>
        <div style="font-size:1.5rem;font-weight:800;color:#a371f7;
                    font-family:'JetBrains Mono',monospace;">{len(df):,}</div>
    </div>
    <div style="padding:16px 24px;">
        <div style="font-size:0.6rem;color:#8b949e;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:4px;">Period</div>
        <div style="font-size:1rem;font-weight:700;color:#c9d1d9;margin-top:6px;">
            {df.index[0].strftime('%b %Y')} → {df.index[-1].strftime('%b %Y')}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — CANDLESTICK + VOLUME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">📊 Price History</div>', unsafe_allow_html=True)

fig_p = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      row_heights=[0.72, 0.28], vertical_spacing=0.02)
fig_p.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='OHLC',
    increasing=dict(line=dict(color='#3fb950'), fillcolor='rgba(63,185,80,0.75)'),
    decreasing=dict(line=dict(color='#f85149'), fillcolor='rgba(248,81,73,0.75)')
), row=1, col=1)
fig_p.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
    line=dict(color='#ffa657', width=1.4, dash='dot')), row=1, col=1)
fig_p.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200',
    line=dict(color='#a371f7', width=1.4, dash='dash')), row=1, col=1)

vc = ['rgba(63,185,80,0.5)' if c >= o else 'rgba(248,81,73,0.5)'
      for c, o in zip(df['Close'], df['Open'])]
fig_p.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
    marker_color=vc, showlegend=False), row=2, col=1)

fig_p.update_layout(**PL, height=500,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                bgcolor='rgba(8,12,20,0.85)', bordercolor='#1e2d40', borderwidth=1))
fig_p.update_yaxes(gridcolor='#121c2b', zerolinecolor='#1e2d40',
                    tickfont=dict(family='JetBrains Mono'))
fig_p.update_xaxes(gridcolor='#121c2b')
st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — RSI + MACD (collapsible)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("RSI & MACD Technical Indicators", expanded=False):
    fig_i = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[0.48, 0.52], vertical_spacing=0.04,
                          subplot_titles=('RSI (14)', 'MACD'))
    fig_i.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#58a6ff', width=1.6)), row=1, col=1)
    fig_i.add_hrect(y0=70, y1=100, fillcolor='rgba(248,81,73,0.06)', line_width=0, row=1, col=1)
    fig_i.add_hrect(y0=0,  y1=30,  fillcolor='rgba(63,185,80,0.06)',  line_width=0, row=1, col=1)
    fig_i.add_hline(y=70, line=dict(color='#f85149', dash='dash', width=1), row=1, col=1)
    fig_i.add_hline(y=30, line=dict(color='#3fb950', dash='dash', width=1), row=1, col=1)

    hv = df['MACD'] - df['MACD_Signal']
    hc = ['rgba(63,185,80,0.6)' if v >= 0 else 'rgba(248,81,73,0.6)' for v in hv]
    fig_i.add_trace(go.Bar(x=df.index, y=hv, marker_color=hc,
        name='Histogram', showlegend=False), row=2, col=1)
    fig_i.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
        line=dict(color='#58a6ff', width=1.4)), row=2, col=1)
    fig_i.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
        line=dict(color='#ffa657', width=1.4)), row=2, col=1)

    fig_i.update_layout(**PL, height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(8,12,20,0.85)', bordercolor='#1e2d40', borderwidth=1))
    fig_i.update_yaxes(gridcolor='#121c2b'); fig_i.update_xaxes(gridcolor='#121c2b')
    st.plotly_chart(fig_i, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
with st.spinner(f"Training {model_choice} on {len(df):,} samples…"):
    model, X_te_s, y_te, scaler = train_model(df, train_split, n_estimators, model_choice)

probas      = model.predict_proba(X_te_s)[:, 1]
final_preds = (probas > conf_thresh / 100).astype(int)

acc      = accuracy_score(y_te, final_preds)
prec     = precision_score(y_te, final_preds, zero_division=0)
rec      = recall_score(y_te, final_preds, zero_division=0)
n_trades = int(sum(final_preds))

test_df = df.iloc[-len(y_te):].copy()
test_df['Strat_Ret'] = final_preds * test_df['Log_Returns']
cum_strat  = np.exp(test_df['Strat_Ret'].cumsum())
cum_market = np.exp(test_df['Log_Returns'].cumsum())
max_dd     = ((cum_strat - cum_strat.cummax()) / cum_strat.cummax()).min()
dr         = test_df['Strat_Ret']
sharpe     = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
ret_s      = cum_strat.iloc[-1]  - 1
ret_m      = cum_market.iloc[-1] - 1

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("🎯 Precision",   f"{prec:.2%}", help="Correct % when model says UP")
m2.metric("✅ Accuracy",    f"{acc:.2%}",  help="Overall correct predictions")
m3.metric("🔁 Recall",      f"{rec:.2%}",  help="% of actual UP days caught")
m4.metric("📋 Trades",      f"{n_trades}", help=f"Days above {conf_thresh}% confidence")
st.markdown("<br>", unsafe_allow_html=True)
m5, m6, m7, m8 = st.columns(4)
m5.metric("📈 Strat Return",   f"{ret_s:.2%}",  delta=f"Mkt {ret_m:.2%}")
m6.metric("⚡ Sharpe Ratio",   f"{sharpe:.2f}", help="Risk-adj return (>1 good, >2 great)")
m7.metric("📉 Max Drawdown",   f"{max_dd:.2%}", help="Worst peak-to-trough loss")
m8.metric("🏆 Outperformance", f"{ret_s - ret_m:.2%}", help="Strategy minus Buy & Hold")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — EQUITY CURVE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">💹 Backtest: Growth of $1</div>', unsafe_allow_html=True)
sc_col = '#3fb950' if ret_s >= ret_m else '#f85149'
fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=test_df.index, y=cum_market,
    name='Buy & Hold', line=dict(color='#4a5568', width=2, dash='dash')))
fig_bt.add_trace(go.Scatter(x=test_df.index, y=cum_strat,
    name=f'AI Strategy', line=dict(color=sc_col, width=2.5),
    fill='tonexty',
    fillcolor=f'rgba(63,185,80,0.07)' if ret_s >= ret_m else 'rgba(248,81,73,0.06)'))
for val, col, xa in [(cum_strat.iloc[-1], sc_col, 8),
                      (cum_market.iloc[-1], '#8b949e', 8)]:
    fig_bt.add_annotation(x=test_df.index[-1], y=val,
        text=f'  ${val:.2f}', font=dict(color=col, size=12, family='JetBrains Mono'),
        showarrow=False, xanchor='left')
fig_bt.update_layout(**PL, height=400,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                bgcolor='rgba(8,12,20,0.85)', bordercolor='#1e2d40', borderwidth=1),
    yaxis=dict(title='Portfolio Value ($)', tickprefix='$',
               gridcolor='#121c2b', tickfont=dict(family='JetBrains Mono')),
    xaxis=dict(gridcolor='#121c2b'))
st.plotly_chart(fig_bt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — FEATURE IMPORTANCE (horizontal bars, gradient, readable labels)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-hdr">🔍 Signal Logic: Feature Importance</div>',
            unsafe_allow_html=True)

imp_df = pd.DataFrame({
    'Feature':    [FEATURE_LABELS[f] for f in FEATURES],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True).reset_index(drop=True)

# Smooth gradient: grey → blue → green based on rank
def rank_color(i, n):
    t = i / max(n - 1, 1)
    if t < 0.5:
        tt = t * 2
        r = int(80  + (88  - 80)  * tt)
        g = int(90  + (166 - 90)  * tt)
        b = int(110 + (255 - 110) * tt)
    else:
        tt = (t - 0.5) * 2
        r = int(88  + (63  - 88)  * tt)
        g = int(166 + (185 - 166) * tt)
        b = int(255 + (80  - 255) * tt)
    return f'rgb({r},{g},{b})'

n = len(imp_df)
bar_colors = [rank_color(i, n) for i in range(n)]

fig_fi = go.Figure(go.Bar(
    x=imp_df['Importance'],
    y=imp_df['Feature'],           # already readable human labels, horizontal
    orientation='h',
    marker=dict(color=bar_colors,
                line=dict(color='rgba(255,255,255,0.06)', width=0.8)),
    text=[f' {v:.1%}' for v in imp_df['Importance']],
    textposition='outside',
    textfont=dict(color='#c9d1d9', family='JetBrains Mono', size=11.5),
    hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>'
))
fig_fi.update_layout(**{
    **PL,
    'height': 320,
    'margin': dict(l=10, r=75, t=10, b=10),   # overrides PL's margin — no duplicate
    'xaxis': dict(title='Importance Score', tickformat='.0%', gridcolor='#121c2b',
                  range=[0, imp_df['Importance'].max() * 1.28]),
    'yaxis': dict(gridcolor='#121c2b',
                  tickfont=dict(family='Inter', size=12.5, color='#c9d1d9')),
})
st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 & 6 — CONFIDENCE HISTOGRAM + MONTHLY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="sec-hdr" style="margin-top:8px;">📊 Confidence Distribution</div>',
                unsafe_allow_html=True)
    fig_c = go.Figure()
    for mask, name, col in [
        (y_te == 1, 'Actual UP',   'rgba(63,185,80,0.65)'),
        (y_te == 0, 'Actual DOWN', 'rgba(248,81,73,0.55)')
    ]:
        fig_c.add_trace(go.Histogram(
            x=probas[mask.values], name=name,
            marker_color=col, nbinsx=25,
            xbins=dict(start=0, end=1, size=0.04)
        ))
    fig_c.add_vline(x=conf_thresh / 100,
        line=dict(color='#ffa657', dash='dash', width=2),
        annotation=dict(text=f' Threshold {conf_thresh}%',
                        font=dict(color='#ffa657', size=11, family='Inter')))
    fig_c.update_layout(**PL, height=300, barmode='overlay',
        legend=dict(orientation='h', y=1.08, bgcolor='rgba(8,12,20,0.8)',
                    bordercolor='#1e2d40', borderwidth=1),
        xaxis=dict(title='Model Confidence', tickformat='.0%', gridcolor='#121c2b'),
        yaxis=dict(title='# of Days', gridcolor='#121c2b'))
    st.plotly_chart(fig_c, use_container_width=True)

with col_b:
    st.markdown('<div class="sec-hdr" style="margin-top:8px;">📅 Monthly P&L Heatmap</div>',
                unsafe_allow_html=True)
    test_df['Month'] = test_df.index.month
    test_df['Year']  = test_df.index.year
    monthly = test_df.groupby(['Year', 'Month'])['Strat_Ret'].sum().reset_index()
    monthly['Ret'] = np.exp(monthly['Strat_Ret']) - 1
    pivot = monthly.pivot(index='Year', columns='Month', values='Ret').fillna(0)
    mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pivot.columns = [mnames[m - 1] for m in pivot.columns]

    fig_h = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale=[[0,'#4d0f0f'],[0.3,'#f85149'],[0.5,'#131c2e'],
                    [0.7,'#3fb950'],[1,'#0f3d1f']],
        zmid=0,
        text=[[f'{v:.1f}%' for v in row] for row in pivot.values * 100],
        texttemplate='%{text}',
        textfont=dict(size=9, family='JetBrains Mono'),
        colorbar=dict(title='%', tickformat='.0f', ticksuffix='%',
                      thickness=12, len=0.85,
                      tickfont=dict(family='JetBrains Mono'))
    ))
    fig_h.update_layout(**{
        **PL,
        'height': 300,
        'margin': dict(l=0, r=50, t=30, b=0),
        'xaxis': dict(title='', side='top', tickfont=dict(family='Inter', size=11)),
        'yaxis': dict(title='', autorange='reversed',
                      tickfont=dict(family='JetBrains Mono', size=10)),
    })
    st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MATH EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
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
<code>MACD = EMA(12) – EMA(26)</code> &nbsp;|&nbsp; Signal = EMA(9) of MACD<br>
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

<div class="info-card"><b>⑥ Ensemble Learning</b><br>
N trees each trained on a bootstrap sample (Bagging).<br>
Averaging across trees reduces variance and suppresses noise.</div>

<div class="info-card"><b>⑦ Confidence Threshold → Precision Filter</b><br>
Trade only when <code>P(UP|X) > threshold</code><br>
Fewer trades, higher win rate. Precision = TP / (TP + FP).</div>

<div class="info-card"><b>⑧ Risk Metrics</b><br>
<code>Sharpe = (μ_daily / σ_daily) × √252</code><br>
Max drawdown = worst peak-to-trough loss on the equity curve.</div>
""", unsafe_allow_html=True)