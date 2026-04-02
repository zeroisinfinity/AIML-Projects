# =============================================================================
# models/trainer.py  —  Model training pipeline
# Responsibilities:
#   • Time-series split (NO random shuffle — strict chronological order)
#   • StandardScaler fit on train, transform on test
#   • Support XGBoost, Random Forest, Gradient Boosting
#   • Return a lightweight TrainResult dataclass
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config import FEATURES


@dataclass
class TrainResult:
    """All outputs from a single training run — passed around the UI layer."""
    model:        object           # fitted classifier
    X_test_scaled: np.ndarray     # scaled test features
    y_test:       pd.Series       # true labels for test period
    scaler:       StandardScaler  # fitted scaler (for future inference)
    feature_names: list[str]      # = FEATURES (for importance lookup)
    algo_name:    str             # human-readable algorithm name
    n_train:      int             # rows used for training
    n_test:       int             # rows used for testing


def train(
    df:          pd.DataFrame,
    split_pct:   int,
    n_estimators: int,
    algo:        str,
) -> TrainResult:
    """
    Train a classifier on df using a strict time-series split.

    Parameters
    ----------
    df           : feature-engineered DataFrame (output of build_features)
    split_pct    : % of rows to use for training  (e.g. 80)
    n_estimators : number of trees / boosting rounds
    algo         : one of "XGBoost", "Random Forest", "Gradient Boosting"

    Why no shuffle?
    ---------------
    Using sklearn's train_test_split(shuffle=True) on time-series data
    leaks future information into training — the model 'sees' market
    conditions that hadn't happened yet.  We always split at index idx
    so train = [0 … idx-1], test = [idx … end].
    """
    X = df[FEATURES]
    y = df['Target']

    idx       = int(len(df) * split_pct / 100)
    X_tr, X_te = X.iloc[:idx], X.iloc[idx:]
    y_tr, y_te = y.iloc[:idx], y.iloc[idx:]

    # Scale: fit ONLY on train to avoid data leakage from test statistics
    scaler     = StandardScaler()
    X_tr_s     = scaler.fit_transform(X_tr)
    X_te_s     = scaler.transform(X_te)

    model = _build_model(algo, n_estimators)
    model.fit(X_tr_s, y_tr)

    return TrainResult(
        model=model,
        X_test_scaled=X_te_s,
        y_test=y_te,
        scaler=scaler,
        feature_names=FEATURES,
        algo_name=algo,
        n_train=len(X_tr),
        n_test=len(X_te),
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_model(algo: str, n_estimators: int):
    """
    Instantiate the chosen classifier.

    XGBoost hyper-params rationale
    --------------------------------
    max_depth=4      : shallow trees prevent overfit on financial noise
    learning_rate    : 0.05  (shrinkage — each tree contributes little;
                              requires more rounds but generalises better)
    subsample=0.8    : row-level bagging per round (like RF but on residuals)
    colsample_bytree : random feature subset per tree (reduces correlation)
    use_label_encoder=False, eval_metric='logloss': suppress deprecation noise
    tree_method='hist': fast histogram-based split finding (GPU-optional)

    Random Forest rationale
    -----------------------
    min_samples_split=50 : prevents micro-splits on noise
    max_features='sqrt'  : Breiman's original recommendation for classification
    n_jobs=-1            : parallelise across all CPU cores

    Gradient Boosting rationale
    ---------------------------
    max_depth=4, learning_rate=0.05 mirrors XGBoost but uses sklearn's
    pure-Python CART implementation (no GPU, slower, but no extra dependency).
    """
    if algo == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Run:  pip install xgboost"
            )
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
        )
    elif algo == "Random Forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=50,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )
    else:  # Gradient Boosting
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )
