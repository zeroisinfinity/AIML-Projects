# =============================================================================
# models/evaluator.py  —  Backtest & performance metrics
# Pure functions — no Streamlit, no charts, no I/O.
# Takes TrainResult + raw df → returns EvalResult dataclass.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score

from models.trainer import TrainResult


@dataclass
class EvalResult:
    """All scalar metrics + time-series for charts."""
    # Classification metrics
    precision:    float
    accuracy:     float
    recall:       float
    n_trades:     int

    # Backtest time-series
    test_df:      pd.DataFrame   # contains Strat_Ret, Log_Returns, Month, Year
    cum_strat:    pd.Series
    cum_market:   pd.Series

    # Risk metrics
    sharpe:       float
    max_drawdown: float
    ret_strategy: float          # total return over test period
    ret_market:   float          # buy-and-hold total return

    # Raw predictions
    probas:       np.ndarray     # P(UP) for each test day
    final_preds:  np.ndarray     # 0/1 after confidence threshold


def evaluate(
    result:     TrainResult,
    df:         pd.DataFrame,
    conf_thresh: int,
) -> EvalResult:
    """
    Apply confidence threshold, compute all metrics, and build backtest series.

    Confidence threshold logic
    --------------------------
    model.predict_proba returns P(class=1) = P(UP tomorrow).
    We only 'trade' (hold overnight) when this probability exceeds the
    threshold.  This is a precision filter: fewer signals, higher hit rate.

    Formally:
        final_pred_t = 1  if  P(UP | X_t) > θ  else  0

    Strategy return:
        R_strat_t = final_pred_t × r_t

    where r_t is the log return on day t.  If we don't trade, we earn 0.

    Sharpe Ratio (annualised)
    -------------------------
        Sharpe = (μ_daily / σ_daily) × √252

    where μ = mean daily strategy return, σ = std dev.
    A Sharpe > 1.0 is considered good; > 2.0 is excellent.

    Max Drawdown
    ------------
        DD_t = (CumMax_t − CumValue_t) / CumMax_t

    where CumMax is the running maximum of the equity curve.
    Max Drawdown = min(DD_t) — the worst peak-to-trough loss percentage.
    """
    probas      = result.model.predict_proba(result.X_test_scaled)[:, 1]
    final_preds = (probas > conf_thresh / 100).astype(int)

    prec = precision_score(result.y_test, final_preds, zero_division=0)
    acc  = accuracy_score(result.y_test,  final_preds)
    rec  = recall_score(result.y_test,    final_preds, zero_division=0)

    # Align test_df rows with y_test (last n rows of df)
    test_df = df.iloc[-len(result.y_test):].copy()
    test_df['Strat_Ret'] = final_preds * test_df['Log_Returns']

    cum_strat  = np.exp(test_df['Strat_Ret'].cumsum())
    cum_market = np.exp(test_df['Log_Returns'].cumsum())

    # Drawdown on equity curve
    running_max = cum_strat.cummax()
    drawdown    = (cum_strat - running_max) / running_max
    max_dd      = float(drawdown.min())

    # Sharpe
    dr     = test_df['Strat_Ret']
    sharpe = float((dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0)

    test_df['Month'] = test_df.index.month
    test_df['Year']  = test_df.index.year

    return EvalResult(
        precision=float(prec),
        accuracy=float(acc),
        recall=float(rec),
        n_trades=int(final_preds.sum()),
        test_df=test_df,
        cum_strat=cum_strat,
        cum_market=cum_market,
        sharpe=sharpe,
        max_drawdown=max_dd,
        ret_strategy=float(cum_strat.iloc[-1] - 1),
        ret_market=float(cum_market.iloc[-1] - 1),
        probas=probas,
        final_preds=final_preds,
    )
