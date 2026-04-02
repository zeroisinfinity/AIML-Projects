# =============================================================================
# ui/charts.py  —  Plotly figure builders
# Every function returns a go.Figure — no st.plotly_chart() calls here.
# app.py decides where/when to render each figure.
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import PLOTLY_BASE, FEATURE_LABELS, FEATURES


# ── 1. Candlestick + Volume ───────────────────────────────────────────────────

def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        name='OHLC',
        increasing=dict(line=dict(color='#3fb950'), fillcolor='rgba(63,185,80,0.75)'),
        decreasing=dict(line=dict(color='#f85149'), fillcolor='rgba(248,81,73,0.75)'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
        name='SMA 50', line=dict(color='#ffa657', width=1.4, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'],
        name='SMA 200', line=dict(color='#a371f7', width=1.4, dash='dash')), row=1, col=1)

    vc = ['rgba(63,185,80,0.5)' if c >= o else 'rgba(248,81,73,0.5)'
          for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
        marker_color=vc, showlegend=False), row=2, col=1)

    fig.update_layout(**PLOTLY_BASE, height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(8,12,20,0.85)', bordercolor='#1e2d40', borderwidth=1))
    fig.update_yaxes(gridcolor='#121c2b', zerolinecolor='#1e2d40',
                     tickfont=dict(family='JetBrains Mono'))
    fig.update_xaxes(gridcolor='#121c2b')
    return fig


# ── 2. RSI + MACD panel ───────────────────────────────────────────────────────

def rsi_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.48, 0.52], vertical_spacing=0.10,
                        subplot_titles=('RSI (14)', 'MACD'))

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#58a6ff', width=1.6)), row=1, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(248,81,73,0.06)', line_width=0, row=1, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(63,185,80,0.06)',  line_width=0, row=1, col=1)
    fig.add_hline(y=70, line=dict(color='#f85149', dash='dash', width=1), row=1, col=1)
    fig.add_hline(y=30, line=dict(color='#3fb950', dash='dash', width=1), row=1, col=1)

    # MACD histogram
    hv = df['MACD'] - df['MACD_Signal']
    hc = ['rgba(63,185,80,0.6)' if v >= 0 else 'rgba(248,81,73,0.6)' for v in hv]
    fig.add_trace(go.Bar(x=df.index, y=hv, marker_color=hc,
        name='Histogram', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
        name='MACD',   line=dict(color='#58a6ff', width=1.4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'],
        name='Signal', line=dict(color='#ffa657', width=1.4)), row=2, col=1)

    fig.update_layout(**{**PLOTLY_BASE,
        'height': 420,
        'margin': dict(l=0, r=0, t=60, b=0),
    }, legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='right', x=1,
                   bgcolor='rgba(8,12,20,0.85)', bordercolor='#1e2d40', borderwidth=1))

    # Style subplot title annotations to sit neatly above each panel
    for ann in fig.layout.annotations:
        ann.update(font=dict(family='Inter', size=12, color='#8b949e'),
                   x=0.01, xanchor='left')

    fig.update_yaxes(gridcolor='#121c2b')
    fig.update_xaxes(gridcolor='#121c2b')
    return fig


# ── 3. Equity curve ───────────────────────────────────────────────────────────

def equity_curve(test_df, cum_strat, cum_market, ret_s, ret_m) -> go.Figure:
    sc_col = '#3fb950' if ret_s >= ret_m else '#f85149'
    fill_c = 'rgba(63,185,80,0.07)' if ret_s >= ret_m else 'rgba(248,81,73,0.06)'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df.index, y=cum_market,
        name='Buy & Hold', line=dict(color='#4a5568', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=test_df.index, y=cum_strat,
        name='AI Strategy', line=dict(color=sc_col, width=2.5),
        fill='tonexty', fillcolor=fill_c))

    for val, col in [(cum_strat.iloc[-1], sc_col), (cum_market.iloc[-1], '#8b949e')]:
        fig.add_annotation(x=test_df.index[-1], y=val,
            text=f'  ${val:.2f}',
            font=dict(color=col, size=12, family='JetBrains Mono'),
            showarrow=False, xanchor='left')

    fig.update_layout(**PLOTLY_BASE, height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    bgcolor='rgba(8,12,20,0.85)', bordercolor='#1e2d40', borderwidth=1),
        yaxis=dict(title='Portfolio Value ($)', tickprefix='$',
                   gridcolor='#121c2b', tickfont=dict(family='JetBrains Mono')),
        xaxis=dict(gridcolor='#121c2b'))
    return fig


# ── 4. Feature importance (horizontal gradient bars) ─────────────────────────

def feature_importance(model) -> go.Figure:
    imp_df = pd.DataFrame({
        'Feature':    [FEATURE_LABELS[f] for f in FEATURES],
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=True).reset_index(drop=True)

    n = len(imp_df)
    colors = [_rank_color(i, n) for i in range(n)]

    fig = go.Figure(go.Bar(
        x=imp_df['Importance'],
        y=imp_df['Feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.06)', width=0.8)),
        text=[f' {v:.1%}' for v in imp_df['Importance']],
        textposition='outside',
        textfont=dict(color='#c9d1d9', family='JetBrains Mono', size=11.5),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>',
    ))
    fig.update_layout(**{**PLOTLY_BASE,
        'height': 320,
        'margin': dict(l=10, r=75, t=10, b=10),
        'xaxis': dict(title='Importance Score', tickformat='.0%',
                      gridcolor='#121c2b',
                      range=[0, imp_df['Importance'].max() * 1.28]),
        'yaxis': dict(gridcolor='#121c2b',
                      tickfont=dict(family='Inter', size=12.5, color='#c9d1d9')),
    })
    return fig


# ── 5. Confidence histogram ───────────────────────────────────────────────────

def confidence_histogram(probas, y_te, conf_thresh: int) -> go.Figure:
    fig = go.Figure()
    for mask, name, col in [
        (y_te == 1, 'Actual UP',   'rgba(63,185,80,0.65)'),
        (y_te == 0, 'Actual DOWN', 'rgba(248,81,73,0.55)'),
    ]:
        fig.add_trace(go.Histogram(
            x=probas[mask.values], name=name,
            marker_color=col, nbinsx=25,
            xbins=dict(start=0, end=1, size=0.04),
        ))
    fig.add_vline(x=conf_thresh / 100,
        line=dict(color='#ffa657', dash='dash', width=2),
        annotation=dict(text=f' Threshold {conf_thresh}%',
                        font=dict(color='#ffa657', size=11, family='Inter')))
    fig.update_layout(**PLOTLY_BASE, height=300, barmode='overlay',
        legend=dict(orientation='h', y=1.08,
                    bgcolor='rgba(8,12,20,0.8)', bordercolor='#1e2d40', borderwidth=1),
        xaxis=dict(title='Model Confidence', tickformat='.0%', gridcolor='#121c2b'),
        yaxis=dict(title='# of Days', gridcolor='#121c2b'))
    return fig


# ── 6. Monthly P&L heatmap ────────────────────────────────────────────────────

def monthly_heatmap(test_df: pd.DataFrame) -> go.Figure:
    monthly = test_df.groupby(['Year', 'Month'])['Strat_Ret'].sum().reset_index()
    monthly['Ret'] = np.exp(monthly['Strat_Ret']) - 1
    pivot = monthly.pivot(index='Year', columns='Month', values='Ret').fillna(0)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pivot.columns = [months[m - 1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
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
                      tickfont=dict(family='JetBrains Mono')),
    ))
    fig.update_layout(**{**PLOTLY_BASE,
        'height': 300,
        'margin': dict(l=0, r=50, t=30, b=0),
        'xaxis': dict(title='', side='top', tickfont=dict(family='Inter', size=11)),
        'yaxis': dict(title='', autorange='reversed',
                      tickfont=dict(family='JetBrains Mono', size=10)),
    })
    return fig


# ── Colour helper ─────────────────────────────────────────────────────────────

def _rank_color(i: int, n: int) -> str:
    """Smooth grey → blue → green gradient by rank."""
    t = i / max(n - 1, 1)
    if t < 0.5:
        tt = t * 2
        r = int(80  + (88  - 80)  * tt)
        g = int(90  + (166 - 90)  * tt)
        b = int(110 + (255 - 110) * tt)
    else:
        tt = (t - 0.5) * 2
        r  = int(88  + (63  - 88)  * tt)
        g  = int(166 + (185 - 166) * tt)
        b  = int(255 + (80  - 255) * tt)
    return f'rgb({r},{g},{b})'
