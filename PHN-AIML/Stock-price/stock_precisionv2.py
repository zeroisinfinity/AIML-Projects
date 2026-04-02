import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(
    page_title="QuantML: Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem; font-weight: 700; color: #58a6ff;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* Headers */
    h1 { background: linear-gradient(90deg, #58a6ff, #a371f7); -webkit-background-clip: text;
         -webkit-text-fill-color: transparent; font-weight: 800 !important; }
    h2, h3 { color: #e6edf3 !important; }

    /* Expander */
    .streamlit-expanderHeader { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }

    /* Divider */
    hr { border-color: #30363d; }

    /* Info box */
    .info-card {
        background: linear-gradient(135deg, #0d2137 0%, #0d1117 100%);
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────────
st.title("📈 QuantML: Stock Direction Predictor")
st.markdown(
    "<p style='color:#8b949e; font-size:1rem;'>"
    "A quantitative ML pipeline that predicts next-day stock direction using "
    "<b style='color:#58a6ff'>Log Returns</b>, <b style='color:#a371f7'>RSI</b>, "
    "<b style='color:#3fb950'>MACD</b>, and <b style='color:#f78166'>Bollinger Bands</b>.</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 📊 Data Settings")
    ticker = st.text_input("Ticker Symbol", "SPY", placeholder="e.g. AAPL, TSLA, BTC-USD")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🤖 Model Settings")

    model_choice = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting"],
        help="Gradient Boosting is slower but often more accurate."
    )
    train_split = st.slider("Training Data %", 60, 90, 80)
    confidence_threshold = st.slider(
        "Confidence Threshold %",
        50, 75, 55,
        help="Only 'trade' when model confidence exceeds this value."
    )
    n_estimators = st.slider("Number of Trees", 100, 500, 200, 50)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#8b949e; font-size:0.75rem;'>⚠️ For educational purposes only. "
        "Not financial advice.</div>",
        unsafe_allow_html=True
    )


# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data
def load_and_process_data(symbol, start):
    """
    Fetch OHLCV data and engineer 8 features for the ML model.
    FIX: yfinance ≥0.2 returns MultiIndex columns — we flatten them.
    """
    df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
    if df.empty:
        return None

    # ── FIX: Flatten MultiIndex columns (yfinance ≥0.2 bug) ────────────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)  # ensure DatetimeIndex, not timestamps

    # ── FEATURE 1 & 2: Log Returns + Rolling Volatility ────────────────────
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Returns'].rolling(21).std()

    # ── FEATURE 3: RSI (14-day) ─────────────────────────────────────────────
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # ── FEATURE 4 & 5: MACD Line + Signal Line ──────────────────────────────
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ── FEATURE 6 & 7: Bollinger Band Width + %B ───────────────────────────
    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    df['BB_Width'] = (upper_band - lower_band) / sma20  # normalized width
    df['BB_pct'] = (df['Close'] - lower_band) / (upper_band - lower_band)

    # ── FEATURE 8: 5-day Momentum ───────────────────────────────────────────
    df['Momentum_5d'] = df['Close'].pct_change(5)

    # ── TARGET: 1 = Tomorrow closes higher ─────────────────────────────────
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Store SMA for chart
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    return df.dropna()


# ==========================================
# 3. ML ENGINE
# ==========================================
def train_model(df, split_pct, n_est, algo):
    features = ['Log_Returns', 'RSI', 'Volatility', 'MACD', 'MACD_Signal',
                'BB_Width', 'BB_pct', 'Momentum_5d']
    X = df[features]
    y = df['Target']

    split_idx = int(len(df) * (split_pct / 100))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if algo == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_est, min_samples_split=50,
            max_features='sqrt', random_state=42, n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=n_est, learning_rate=0.05,
            max_depth=4, random_state=42
        )
    model.fit(X_train_sc, y_train)
    return model, X_test_sc, y_test, scaler, features


# ==========================================
# 4. EXECUTION
# ==========================================
df = load_and_process_data(ticker, start_date)

if df is None:
    st.error("⚠️ Invalid ticker or no data returned. Try symbols like AAPL, TSLA, SPY.")
    st.stop()

# ── CHART 1: PRICE HISTORY WITH MAs ─────────────────────────────────────────
st.markdown("## 📊 Price History")

fig_price = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.03
)

fig_price.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="OHLC",
    increasing_line_color='#3fb950', decreasing_line_color='#f85149'
), row=1, col=1)

fig_price.add_trace(go.Scatter(
    x=df.index, y=df['SMA_50'], name="SMA 50",
    line=dict(color='#ffa657', width=1.5, dash='dot')
), row=1, col=1)

fig_price.add_trace(go.Scatter(
    x=df.index, y=df['SMA_200'], name="SMA 200",
    line=dict(color='#a371f7', width=1.5, dash='dash')
), row=1, col=1)

# Volume
colors = ['#3fb950' if c >= o else '#f85149'
          for c, o in zip(df['Close'], df['Open'])]
fig_price.add_trace(go.Bar(
    x=df.index, y=df['Volume'], name="Volume",
    marker_color=colors, opacity=0.6
), row=2, col=1)

fig_price.update_layout(
    template="plotly_dark",
    paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
    height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=10, b=0),
    font=dict(family="monospace", color="#c9d1d9")
)
fig_price.update_yaxes(gridcolor='#21262d', zerolinecolor='#30363d')
fig_price.update_xaxes(gridcolor='#21262d')
st.plotly_chart(fig_price, use_container_width=True)

# ── RSI CHART ────────────────────────────────────────────────────────────────
with st.expander("📉 RSI & MACD Indicators", expanded=False):
    fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.5], vertical_spacing=0.05)

    # RSI
    fig_ind.add_trace(go.Scatter(
        x=df.index, y=df['RSI'], name="RSI",
        line=dict(color='#58a6ff', width=1.5)
    ), row=1, col=1)
    fig_ind.add_hline(y=70, line=dict(color='#f85149', dash='dash', width=1), row=1, col=1)
    fig_ind.add_hline(y=30, line=dict(color='#3fb950', dash='dash', width=1), row=1, col=1)

    # MACD
    macd_colors = ['#3fb950' if v >= 0 else '#f85149' for v in (df['MACD'] - df['MACD_Signal'])]
    fig_ind.add_trace(go.Bar(
        x=df.index, y=df['MACD'] - df['MACD_Signal'],
        name="MACD Histogram", marker_color=macd_colors, opacity=0.7
    ), row=2, col=1)
    fig_ind.add_trace(go.Scatter(
        x=df.index, y=df['MACD'], name="MACD Line",
        line=dict(color='#58a6ff', width=1.2)
    ), row=2, col=1)
    fig_ind.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'], name="Signal Line",
        line=dict(color='#ffa657', width=1.2)
    ), row=2, col=1)

    fig_ind.update_layout(
        template="plotly_dark", paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="monospace", color="#c9d1d9")
    )
    fig_ind.update_yaxes(gridcolor='#21262d')
    fig_ind.update_xaxes(gridcolor='#21262d')
    st.plotly_chart(fig_ind, use_container_width=True)

# ── MODEL TRAINING ────────────────────────────────────────────────────────────
st.markdown("## 🤖 Model Training & Evaluation")
st.markdown("<hr>", unsafe_allow_html=True)

with st.spinner(f"Training {model_choice} on {len(df):,} data points…"):
    model, X_test_sc, y_test, scaler, feature_cols = train_model(
        df, train_split, n_estimators, model_choice
    )

probas = model.predict_proba(X_test_sc)[:, 1]
final_preds = (probas > (confidence_threshold / 100)).astype(int)

acc = accuracy_score(y_test, final_preds)
prec = precision_score(y_test, final_preds, zero_division=0)
rec = recall_score(y_test, final_preds, zero_division=0)
n_trades = int(sum(final_preds))

# Sharpe & Max Drawdown
test_df = df.iloc[-len(y_test):].copy()
test_df['Strat_Ret'] = final_preds * test_df['Log_Returns']
cum_strat = np.exp(test_df['Strat_Ret'].cumsum())
cum_market = np.exp(test_df['Log_Returns'].cumsum())

roll_max = cum_strat.cummax()
drawdown = (cum_strat - roll_max) / roll_max
max_dd = drawdown.min()

daily_ret = test_df['Strat_Ret']
sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0

final_return_strat = cum_strat.iloc[-1] - 1
final_return_market = cum_market.iloc[-1] - 1

# ── METRICS ROW 1 ─────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("🎯 Precision", f"{prec:.2%}", help="% correct when predicting UP")
m2.metric("✅ Overall Accuracy", f"{acc:.2%}", help="Total correct predictions")
m3.metric("📋 Trades Taken", f"{n_trades}", help=f"Days model exceeded {confidence_threshold}% confidence")
m4.metric("🔁 Recall", f"{rec:.2%}", help="% of actual UP days that were caught")

st.markdown("<br>", unsafe_allow_html=True)
m5, m6, m7, m8 = st.columns(4)
delta_color = "normal" if final_return_strat >= 0 else "inverse"
m5.metric("📈 Strategy Return", f"{final_return_strat:.2%}", delta=f"vs Market {final_return_market:.2%}")
m6.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}", help="Risk-adjusted return (>1 = good, >2 = great)")
m7.metric("📉 Max Drawdown", f"{max_dd:.2%}", help="Largest peak-to-trough loss")
m8.metric("🏆 Outperformance", f"{(final_return_strat - final_return_market):.2%}",
          help="Strategy return minus Buy & Hold return")

# ── CHART 2: EQUITY CURVE ─────────────────────────────────────────────────
st.markdown("### 💹 Backtest: Growth of $1")

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(
    x=test_df.index, y=cum_market,
    name="Buy & Hold (Market)",
    line=dict(color='#8b949e', width=2, dash='dash'),
    fill=None
))
fig_bt.add_trace(go.Scatter(
    x=test_df.index, y=cum_strat,
    name=f"AI Strategy ({model_choice})",
    line=dict(color='#3fb950', width=2.5),
    fill='tonexty',
    fillcolor='rgba(63,185,80,0.07)'
))

# Annotate final values
fig_bt.add_annotation(
    x=test_df.index[-1], y=cum_strat.iloc[-1],
    text=f"${cum_strat.iloc[-1]:.2f}",
    font=dict(color='#3fb950', size=12, family='monospace'),
    showarrow=True, arrowcolor='#3fb950', ax=40, ay=-20
)
fig_bt.add_annotation(
    x=test_df.index[-1], y=cum_market.iloc[-1],
    text=f"${cum_market.iloc[-1]:.2f}",
    font=dict(color='#8b949e', size=12, family='monospace'),
    showarrow=True, arrowcolor='#8b949e', ax=40, ay=20
)

fig_bt.update_layout(
    template="plotly_dark", paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
    height=430,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Portfolio Value ($)",
    margin=dict(l=0, r=0, t=20, b=0),
    font=dict(family="monospace", color="#c9d1d9"),
    hovermode='x unified'
)
fig_bt.update_yaxes(gridcolor='#21262d', tickprefix='$')
fig_bt.update_xaxes(gridcolor='#21262d')
st.plotly_chart(fig_bt, use_container_width=True)

# ── CHART 3: FEATURE IMPORTANCE ───────────────────────────────────────────
st.markdown("### 🔍 Signal Logic: Feature Importance")

importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

# Colour-code by importance quartile
colors_fi = []
for v in importances['Importance']:
    if v >= importances['Importance'].quantile(0.75):
        colors_fi.append('#3fb950')  # top 25% → green
    elif v >= importances['Importance'].quantile(0.50):
        colors_fi.append('#58a6ff')  # mid 50% → blue
    else:
        colors_fi.append('#8b949e')  # bottom 25% → grey

fig_fi = go.Figure(go.Bar(
    x=importances['Importance'],
    y=importances['Feature'],
    orientation='h',
    marker=dict(color=colors_fi, line=dict(color='#30363d', width=0.5)),
    text=[f"{v:.1%}" for v in importances['Importance']],
    textposition='outside',
    textfont=dict(color='#c9d1d9', family='monospace')
))
fig_fi.update_layout(
    template="plotly_dark", paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
    height=340,
    xaxis=dict(title="Importance Score", gridcolor='#21262d', tickformat='.0%'),
    yaxis=dict(gridcolor='#21262d'),
    margin=dict(l=0, r=60, t=10, b=0),
    font=dict(family="monospace", color="#c9d1d9")
)
st.plotly_chart(fig_fi, use_container_width=True)

# ── CHART 4: CONFIDENCE DISTRIBUTION ─────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📊 Prediction Confidence Distribution")
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(
        x=probas[y_test == 1], name="Actual UP",
        marker_color='rgba(63,185,80,0.7)', nbinsx=30,
        xbins=dict(start=0, end=1, size=0.033)
    ))
    fig_conf.add_trace(go.Histogram(
        x=probas[y_test == 0], name="Actual DOWN",
        marker_color='rgba(248,81,73,0.7)', nbinsx=30,
        xbins=dict(start=0, end=1, size=0.033)
    ))
    fig_conf.add_vline(
        x=confidence_threshold / 100,
        line=dict(color='#ffa657', dash='dash', width=2),
        annotation_text=f"Threshold {confidence_threshold}%",
        annotation_font_color='#ffa657'
    )
    fig_conf.update_layout(
        barmode='overlay', template="plotly_dark",
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1),
        xaxis=dict(title="Model Confidence", tickformat='.0%', gridcolor='#21262d'),
        yaxis=dict(title="Frequency", gridcolor='#21262d'),
        font=dict(family="monospace", color="#c9d1d9")
    )
    st.plotly_chart(fig_conf, use_container_width=True)

with col_b:
    st.markdown("### 📅 Monthly P&L Heatmap")
    test_df['Month'] = test_df.index.month
    test_df['Year'] = test_df.index.year
    monthly = test_df.groupby(['Year', 'Month'])['Strat_Ret'].sum().reset_index()
    monthly['Return'] = np.exp(monthly['Strat_Ret']) - 1
    pivot = monthly.pivot(index='Year', columns='Month', values='Return').fillna(0)
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][
        :len(pivot.columns)]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, '#f85149'], [0.5, '#21262d'], [1, '#3fb950']],
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values * 100],
        texttemplate="%{text}",
        textfont=dict(size=9, family='monospace'),
        colorbar=dict(title="Return %", tickformat='.1f', ticksuffix='%')
    ))
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Year", dtick=1),
        font=dict(family="monospace", color="#c9d1d9")
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── MATH EXPLAINER ────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("🎓 Math-Proof Logic Behind the Model", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class='info-card'>
<b>① Stationarity via Log Returns</b><br>
Raw prices trend and grow exponentially. We convert them to<br>
<code>r_t = ln(P_t / P_{t-1})</code><br>
which is mean-reverting and suitable for ML.
</div>

<div class='info-card'>
<b>② RSI — Bounded Momentum Oscillator</b><br>
<code>RSI = 100 – 100/(1 + AvgGain/AvgLoss)</code><br>
Bounded [0–100]. Overbought >70, Oversold <30.
</div>

<div class='info-card'>
<b>③ MACD — Trend & Momentum</b><br>
<code>MACD = EMA(12) – EMA(26)</code><br>
Signal = EMA(9) of MACD. Crossovers signal regime shifts.
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class='info-card'>
<b>④ Bollinger Bands — Volatility Context</b><br>
<code>BB_Width = (Upper – Lower) / SMA20</code><br>
High width = high volatility regime. %B shows price position within bands.
</div>

<div class='info-card'>
<b>⑤ No Look-Ahead Bias</b><br>
Target = <code>(Close[t+1] > Close[t])</code> using <code>.shift(-1)</code><br>
Train/test split is strictly sequential — no shuffling.
</div>

<div class='info-card'>
<b>⑥ Confidence Threshold → Precision Filter</b><br>
Only trade when <code>P(UP) > threshold</code>.<br>
This maximises <code>Precision = TP/(TP+FP)</code> at cost of fewer trades.
</div>
""", unsafe_allow_html=True)