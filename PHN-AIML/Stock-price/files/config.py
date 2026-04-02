# =============================================================================
# config.py  —  QuantML global constants
# All magic numbers, stock universe, and feature registry live here.
# Nothing else imports from app-level; everything imports from here.
# =============================================================================

# ── Feature registry ──────────────────────────────────────────────────────────
FEATURES = [
    'Log_Returns', 'RSI', 'Volatility',
    'MACD', 'MACD_Signal',
    'BB_Width', 'BB_pct',
    'Momentum_5d',
]

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

# ── Indicator windows ─────────────────────────────────────────────────────────
RSI_WINDOW       = 14
VOLATILITY_WIN   = 21
BB_WINDOW        = 20
MOMENTUM_WINDOW  = 5
EMA_FAST         = 12
EMA_SLOW         = 26
EMA_SIGNAL       = 9
SMA_SHORT        = 50
SMA_LONG         = 200

# ── Cache TTL (seconds) ───────────────────────────────────────────────────────
# 60 s = near-real-time refresh for intraday prices.
CACHE_TTL = 60

# ── Stock universe ────────────────────────────────────────────────────────────
STOCKS = {
    "US Large-Cap": {
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
    "Growth & Tech": {
        "AMD – AMD":         "AMD",
        "NFLX – Netflix":    "NFLX",
        "CRM – Salesforce":  "CRM",
        "PLTR – Palantir":   "PLTR",
        "SNOW – Snowflake":  "SNOW",
        "UBER – Uber":       "UBER",
        "SHOP – Shopify":    "SHOP",
    },
    "Finance & Value": {
        "JPM – JPMorgan":        "JPM",
        "BAC – Bank of America": "BAC",
        "GS – Goldman Sachs":    "GS",
        "V – Visa":              "V",
        "MA – Mastercard":       "MA",
        "WMT – Walmart":         "WMT",
    },
    "Global ETFs": {
        "EEM – Emerging Mkts":   "EEM",
        "EFA – Developed Mkts":  "EFA",
        "FXI – China Large-Cap": "FXI",
        "INDA – India ETF":      "INDA",
        "EWJ – Japan ETF":       "EWJ",
    },
    "Crypto": {
        "BTC-USD – Bitcoin":  "BTC-USD",
        "ETH-USD – Ethereum": "ETH-USD",
        "SOL-USD – Solana":   "SOL-USD",
        "BNB-USD – BNB":      "BNB-USD",
    },
    "Commodities": {
        "GLD – Gold ETF":     "GLD",
        "USO – Oil ETF":      "USO",
        "SLV – Silver ETF":   "SLV",
        "PDBC – Commodities": "PDBC",
    },
}

# ── Algorithm choices ─────────────────────────────────────────────────────────
ALGORITHMS = ["XGBoost", "Random Forest", "Gradient Boosting"]

# ── Plotly shared layout base ─────────────────────────────────────────────────
# Charts import this and merge their own overrides via {**PLOTLY_BASE, ...}
PLOTLY_BASE = dict(
    paper_bgcolor='#080c14',
    plot_bgcolor='#0c1220',
    font=dict(family='Inter', color='#c9d1d9', size=12),
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=40, b=0),
    hovermode='x unified',
)
