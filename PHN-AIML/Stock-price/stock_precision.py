import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
# Sets the browser tab title and forces a wide layout for charts
st.set_page_config(page_title="QuantML: Stock Predictor", layout="wide")

st.title("📈 QuantML: Stock Direction Predictor")
st.markdown("""
This dashboard uses **Machine Learning (Random Forest)** to predict if a stock will close **Higher** tomorrow.
By converting prices into **Log Returns** and **RSI**, we ensure the math is 'Stationary' (statistically stable).
""")

# SIDEBAR: User interaction area
st.sidebar.header("Step 1: Data Settings")
ticker = st.sidebar.text_input("Enter Ticker (e.g., AAPL, TSLA, BTC-USD)", "SPY")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))

st.sidebar.header("Step 2: Model Settings")
train_split = st.sidebar.slider("Training Data %", 50, 95, 80)
# 'Confidence' is key. We only 'Trade' if the model is very sure.
confidence_threshold = st.sidebar.slider("Confidence Threshold (Trade only if > X%)", 50, 75, 55)


# ==========================================
# 2. DATA ENGINE (PHASE A & B)
# ==========================================
@st.cache_data
def load_and_process_data(symbol, start):
    """
    Fetches raw OHLCV data and applies mathematical transformations.
    """
    # yfinance mimics a Kaggle dataset experience but with live data
    df = yf.download(symbol, start=start)
    if df.empty: return None

    # --- MATH BLOCK: STATIONARITY ---
    # We use Log Returns: ln(Price_Today / Price_Yesterday)
    # This centers the data around 0, making it 'predictable' for ML.
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- MATH BLOCK: RSI (MOMENTUM) ---
    # Bounded 0-100 indicator. Math: 100 - (100 / (1 + AvgGain/AvgLoss))
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- MATH BLOCK: VOLATILITY ---
    # Measures the 'Fear' in the market (Rolling 21-day Standard Deviation)
    df['Volatility'] = df['Log_Returns'].rolling(window=21).std()

    # --- TARGET LABELING (THE 'ANSWER KEY') ---
    # .shift(-1) moves TOMORROW'S price back to TODAY'S row.
    # Logic: If Close[Tomorrow] > Close[Today], Target = 1 (UP)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()


# ==========================================
# 3. ML ENGINE (PHASE C, D, E)
# ==========================================
def train_model(df, split_percent):
    """
    Splits data sequentially, scales it, and trains the Random Forest.
    """
    # Features (X) are our clues; Target (y) is the answer
    features = ['Log_Returns', 'RSI', 'Volatility']
    X = df[features]
    y = df['Target']

    # TIME-SERIES SPLIT: Crucial for finance. No random shuffling!
    split_idx = int(len(df) * (split_percent / 100))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # SCALING: Standardizes features to have mean=0 and std=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # THE RANDOM FOREST: 200 decision trees voting on the outcome
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, X_test_scaled, y_test, scaler, features


# ==========================================
# 4. EXECUTION & VISUALIZATION
# ==========================================
df = load_and_process_data(ticker, start_date)

if df is not None:
    # --- VISUAL 1: INTERACTIVE PRICE CHART ---
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#00d1ff')))
    fig_price.update_layout(title=f"{ticker} Price History", template="plotly_dark", height=400)
    st.plotly_chart(fig_price, use_container_width=True)

    # --- MODEL TRAINING ---
    with st.spinner("Training Mathematical Engine..."):
        model, X_test_scaled, y_test, scaler, feature_cols = train_model(df, train_split)

    # --- PROBABILITY FILTERING ---
    # We only take the trade if the model is confident
    probas = model.predict_proba(X_test_scaled)[:, 1]  # Probability of 'UP'
    final_preds = (probas > (confidence_threshold / 100)).astype(int)

    # --- METRICS CALCULATIONS ---
    acc = accuracy_score(y_test, final_preds)
    # Precision: When we predict 'UP', how often are we right?
    prec = precision_score(y_test, final_preds, zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Strategy Precision", f"{prec:.2%}", help="Accuracy when the model predicts an UP move.")
    m2.metric("Overall Accuracy", f"{acc:.2%}", help="Total correct guesses vs total days.")
    m3.metric("Trades Found", f"{sum(final_preds)}",
              help=f"Number of days model was > {confidence_threshold}% confident.")

    # --- VISUAL 2: EQUITY CURVE (THE BACKTEST) ---
    st.subheader("Backtest Performance (The Money Proof)")
    test_df = df.iloc[-len(y_test):].copy()

    # Strategy Return: If we 'Trade' (1), we get the return. If we 'Wait' (0), we get 0.
    test_df['Strategy_Returns'] = final_preds * test_df['Log_Returns']

    # Compounding Returns: Cumulative growth of $1
    cum_market = np.exp(test_df['Log_Returns'].cumsum())
    cum_strategy = np.exp(test_df['Strategy_Returns'].cumsum())

    fig_backtest = go.Figure()
    fig_backtest.add_trace(
        go.Scatter(x=test_df.index, y=cum_market, name="Buy & Hold (Market)", line=dict(color='gray', dash='dash')))
    fig_backtest.add_trace(
        go.Scatter(x=test_df.index, y=cum_strategy, name="AI Strategy", line=dict(color='#00ff88', width=3)))
    fig_backtest.update_layout(title="Growth of $1: AI vs Market", template="plotly_dark", height=500)
    st.plotly_chart(fig_backtest, use_container_width=True)

    # --- FEATURE IMPORTANCE VIZ ---
    st.subheader("Signal Logic: What is the Model looking at?")
    importances = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
    st.bar_chart(importances.set_index('Feature'))

else:
    st.error("Invalid Ticker. Please use symbols like AAPL, TSLA, or SPY.")

# FOOTER EXPLAINER
with st.expander("🎓 Learn the Math-Proof Logic"):
    st.write("""
    1. **Stationarity**: Prices trend, which makes them hard to model. We use **Log Returns**, which center the data around 0.
    2. **Look-ahead Bias**: We use `.shift(-1)` to create our Target. This means we use *Today's* data to predict *Tomorrow's* result.
    3. **Random Forest**: Instead of one decision tree, we use 200. This averages out 'noise' and focuses on 'signal'.
    4. **Confidence Threshold**: We don't trade every day. By only trading when the model is > 55% sure, we mathematically increase our odds of winning.
    """)