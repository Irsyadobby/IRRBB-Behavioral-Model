"""
Behavioral rate modeling library for Interest Rate Risk in the Banking Book
(IRRBB) — early redemption (term deposits) and prepayment (loan products)
rate forecasting.

This module implements two independent modeling tracks on the same target
series (a rate bounded in (0,1)):

1. **ARIMAX (SARIMAX with exogenous regressors)** — the primary statistical
   model, selected via a combinatorial grid search over macroeconomic
   predictor subsets and (p,d,q) orders, validated through standard ARIMA
   residual diagnostics (Ljung-Box, Heteroskedasticity, Jarque-Bera) and
   coefficient significance testing.
2. **HistGradientBoostingRegressor (HGBR)** — a non-linear machine-learning
   benchmark, tuned via time-series cross-validation, with permutation-based
   feature importance for interpretability.

Supporting utilities cover target/predictor transformation (logit/log/log1p,
z-score standardization, lagging, EMA smoothing) and pre-modeling feature
screening (lag-correlation scan, VIF multicollinearity check, PCA).

Full statistical derivations (formulas for every transformation and every
diagnostic test used below) are documented in ``docs/methodology.md`` of the
parent repository. Docstrings in this module reference the corresponding
section (e.g. "Methodology §4.1") for traceability between code and theory.

Author: Market & Liquidity Risk Quant Analyst — IRRBB / FRTB / VaR / Treasury
        Product Pricing & Valuation.
"""

from __future__ import annotations

import time
import warnings
import itertools
import re
from typing import Dict, List, Tuple, Optional, Sequence, Any

# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import io
from contextlib import redirect_stdout

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
# Optional: apply a custom matplotlib style file if available locally.
# Left unset by default for portability across environments — set your own
# style file path here if desired, e.g.:
#     plt.style.use("path/to/your/style.mplstyle")
import seaborn as sns

# --------------------------------------------------------------------------
# Statistics (ARIMAX estimation & diagnostics — Methodology §1, §4)
# --------------------------------------------------------------------------
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import scipy.stats as stats
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from scipy.stats import ttest_1samp, t
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# --------------------------------------------------------------------------
# Machine Learning (HGBR benchmark — Methodology §6)
# --------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.inspection import permutation_importance

__all__ = [
    # Target/predictor transformation utilities
    "run_arima",
    "evaluate_model",
    "evaluate_selected_models",
    # Feature engineering & screening
    "add_ema",
    "cek_vif",
    "find_correl",
    "run_pca",
    # ML benchmark
    "run_hgbr",
    "evaluate_hgbr",
    "feature_importance_permutation",
]


# ============================================================================
# Target Transformation Utilities  (Methodology §2.1)
# ============================================================================

def _transform_y(y: pd.Series, method: Optional[str], eps: float) -> pd.Series:
    """
    Apply a target (Y) transformation prior to ARIMAX / HGBR estimation.

    Behavioral rate series are bounded in (0,1); fitting a Gaussian-error
    model directly on a bounded series risks out-of-range forecasts and
    violates the homoskedasticity assumption near the boundaries. This
    function implements the transformations described in Methodology §2.1.

    Parameters
    ----------
    y : pd.Series
        Raw target series (e.g. a monthly early redemption / prepayment
        rate).
    method : {'none', 'log', 'log1p', 'logit'}, optional
        - 'none'  : no transformation.
        - 'log'   : y' = ln(y). Requires y > 0.
        - 'log1p' : y' = ln(1 + y). Requires y >= 0 (handles zero values).
        - 'logit' : y' = ln(y / (1-y)). Requires y in (0,1) — the primary
          transform used for behavioral rate targets, since its inverse
          (the logistic/sigmoid function) guarantees back-transformed
          forecasts remain valid rates in (0,1).
    eps : float
        Numerical floor/ceiling used to clip y away from 0 and 1 (or 0
        alone, for 'log') before taking the transform, avoiding -inf/+inf.

    Returns
    -------
    pd.Series
        Transformed series, on which the ARIMAX/HGBR model is actually
        fitted.

    See Also
    --------
    _inverse_y : back-transforms model output to the original rate scale.
    """
    method = (method or 'none').lower()
    y = y.astype(float)
    if method == 'none':
        return y
    elif method == 'log': # Y > 0
        return np.log(np.clip(y, eps, None))
    elif method == 'log1p': # Y >= 0
        return np.log1p(np.clip(y, 0, None))
    elif method == 'logit': # Y in (0,1)
        y_c = np.clip(y, eps, 1 - eps)
        return np.log(y_c / (1.0 - y_c))
    else:
        raise ValueError(f"y_transform tidak dikenal: {method}")


def _inverse_y(y_t: pd.Series | np.ndarray, method: Optional[str], eps: float) -> pd.Series | np.ndarray:
    """
    Back-transform model output (fitted values / forecasts) to the original
    rate scale. Exact inverse of :func:`_transform_y` — see Methodology §2.1
    for the closed-form derivation of each inverse mapping (in particular,
    the logistic/sigmoid inverse of the logit transform).

    Parameters
    ----------
    y_t : pd.Series or np.ndarray
        Transformed-scale values (model fitted values or forecasts).
    method : {'none', 'log', 'log1p', 'logit'}, optional
        Must match the ``method`` used in the corresponding
        :func:`_transform_y` call.
    eps : float
        Unused here (kept for a symmetric function signature with
        :func:`_transform_y`); retained for interface consistency.

    Returns
    -------
    pd.Series or np.ndarray
        Values back-transformed to the original scale (e.g. a rate in
        (0,1) for the 'logit' method).
    """
    method = (method or 'none').lower()
    if method == 'none':
        return y_t
    elif method == 'log':
        return np.exp(y_t)
    elif method == 'log1p':
        return np.expm1(y_t)
    elif method == 'logit':
        return 1.0 / (1.0 + np.exp(-y_t))
    else:
        raise ValueError(f"y_transform tidak dikenal: {method}")


# ============================================================================
# Primary Model — ARIMAX Grid Search  (Methodology §1)
# ============================================================================

def run_arima(
    df: pd.DataFrame,
    y_col: str = 'IDR',
    x_num: int = 2,
    x_list: Optional[List[str]] = None,
    test_size: int = 12,
    min_obs: int = 36,
    scale_exog: bool = True,
    dummy_variable: bool = False,
    intercept: bool = True,
    order_grid: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = True,
    suppress_warnings: bool = True,
    optimizers: Tuple[str, ...] = ('lbfgs', 'bfgs', 'powell', 'nm'),
    y_transform: Optional[str] = 'none',
    log_eps: float = 1e-6,
    maxiter: int = 500,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Combinatorial ARIMAX (SARIMAX with exogenous regressors) grid search.

    For every combination of ``x_num`` macroeconomic predictors drawn from
    the candidate pool (columns prefixed ``'X'`` in ``df``, or ``x_list`` if
    provided), this function fits a SARIMAX model across an ARIMA order
    grid, selects the best-fitting ``(p,d,q)`` per predictor combination by
    **out-of-sample MAPE** on a held-out test window, and returns a ranked
    leaderboard across all predictor combinations tried.

    See Methodology §1.4 for the full rationale of this two-stage
    (variable-subset × order) search strategy, and §1.3 for the
    Maximum-Likelihood / optimizer fallback estimation procedure.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel containing the target column ``y_col``, a ``'Date'``
        column (optional, used for chronological sorting), and candidate
        exogenous predictor columns prefixed with ``'X'`` (e.g. ``'X_rate'``,
        ``'X_cpi'``). A dedicated anomaly/outlier dummy column named
        ``'XAO'`` is treated specially when ``dummy_variable=True``.
    y_col : str, default 'IDR'
        Name of the target (behavioral rate) column.
    x_num : int, default 2
        Number of macro predictors to combine per ARIMAX specification
        (excluding the anomaly dummy, which is always appended when
        ``dummy_variable=True``).
    x_list : list of str, optional
        Restrict the candidate predictor pool to these columns instead of
        all columns prefixed ``'X'``.
    test_size : int, default 12
        Number of most-recent observations held out for out-of-sample MAPE
        evaluation (forecast horizon).
    min_obs : int, default 36
        Minimum number of non-missing observations required for a given
        predictor combination to be attempted; combinations below this
        threshold are skipped and logged as 'Insufficient Observation'.
    scale_exog : bool, default True
        If True, standardize (z-score) continuous exogenous regressors
        using train-window statistics only (Methodology §2.2). Binary/dummy
        columns are automatically detected and excluded from scaling.
    dummy_variable : bool, default False
        If True, expects an anomaly/outlier dummy column ``'XAO'`` in
        ``df`` (e.g. a COVID-19 shock indicator) and appends it to every
        predictor combination, excluded from the ``x_num`` combinatorics.
    intercept : bool, default True
        If True, includes a constant/trend term in the SARIMAX
        specification (``trend='c'``); otherwise no intercept
        (``trend='n'``).
    order_grid : list of (int,int,int), optional
        Candidate ``(p,d,q)`` orders to search. Defaults to the full
        ``{0,1,2}³`` grid (27 combinations).
    verbose : bool, default True
        Print progress per predictor combination.
    suppress_warnings : bool, default True
        Suppress statsmodels ``ValueWarning`` / ``ConvergenceWarning``
        during the search (recommended — grid search intentionally attempts
        many mis-specified models, which is expected to raise warnings).
    optimizers : tuple of str, default ('lbfgs','bfgs','powell','nm')
        MLE optimizers attempted in order; the first to converge is used
        (Methodology §1.3 — ARIMAX likelihood surfaces with several
        exogenous regressors are not always well-behaved).
    y_transform : {'none','log','log1p','logit'}, default 'none'
        Target transformation applied before fitting; see
        :func:`_transform_y` and Methodology §2.1. Use ``'logit'`` for
        behavioral rate targets bounded in (0,1).
    log_eps : float, default 1e-6
        Numerical floor used by the target transform (see
        :func:`_transform_y`).
    maxiter : int, default 500
        Maximum MLE iterations per optimizer attempt.

    Returns
    -------
    results_df : pd.DataFrame
        Leaderboard of the best model per predictor combination, columns:
        ``['X Set','Order','AIC Train','MAPE Test','N Train','N Test',
        'N Total','Note']``, sorted ascending by ``'MAPE Test'``.
    fitted_models : dict[str, SARIMAXResults]
        Fitted model objects keyed by the same ``'X Set'`` label used in
        ``results_df``, for downstream use in :func:`evaluate_model` /
        :func:`evaluate_selected_models`.
    """
    t0_all = time.time()

    if suppress_warnings:
        warnings.filterwarnings('ignore', category=ValueWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    if order_grid is None:
        order_grid = [(p, d, q) for p in (0, 1, 2) for d in (0, 1, 2) for q in (0, 1, 2)]

    x_all = [c for c in df.columns if c.startswith('X')]

    if dummy_variable:
        has_ao = ('XAO' in x_all)
        x_macro = [c for c in x_all if c != 'XAO']
    else:
        x_macro = x_all

    if x_list is None:
        x_list = x_macro

    x_sets = list(itertools.combinations(x_list, x_num))
    total_models = len(x_sets)

    results: List[dict] = []
    fitted_models: Dict[str, object] = {}

    for i, x_set in enumerate(x_sets, start=1):
        t0_pair = time.time()
        x_names = list(x_set)

        if dummy_variable:
            x_names = x_names + ['XAO']

        x_label = " + ".join(x_names)

        cols = [y_col] + x_names
        sub = df[cols].copy().dropna(subset=cols)
        n_total = len(sub)

        if verbose:
            print(f"[{i:3d}/{total_models}] Mencoba Kombinasi: {x_label}\n observasi: {n_total}")

        if n_total < min_obs:
            results.append({
                'X Set': x_label,
                'Order': None,
                'AIC Train': np.nan,
                'MAPE Test': np.nan,
                'N Train': 0,
                'N Test': 0,
                'N Total': n_total,
                'Note': 'Insufficient Observation',
            })
            if verbose:
                print(f"  -> dilewati (observasi < {min_obs})\n")
            continue

        n_test = test_size
        n_train = n_total - n_test

        y_train = sub[y_col].iloc[:n_train]
        y_test = sub[y_col].iloc[n_train:]
        x_train = sub[x_names].iloc[:n_train].copy()
        x_test = sub[x_names].iloc[n_train:].copy()

        if scale_exog:
            binary_cols = []

            for c in x_names:
                vals = sub[c].dropna().unique()

                if len(vals) <= 3 and set(np.round(vals, 6)).issubset({0.0, 1.0}):
                    binary_cols.append(c)

            cont_cols = [c for c in x_names if c not in binary_cols]

            mu = x_train[cont_cols].mean()
            std = x_train[cont_cols].std(ddof=0).replace(0, 1)
            x_train.loc[:, cont_cols] = (x_train[cont_cols] - mu) / std
            x_test.loc[:, cont_cols] = (x_test[cont_cols] - mu) / std

        y_train_t = _transform_y(y_train, y_transform, log_eps)

        trend_flag = 'c' if intercept else 'n'

        best_model = None
        best_order = None
        best_aic = np.inf
        best_mape = np.inf

        for (p, d, q) in order_grid:
            model = SARIMAX(
                endog=y_train_t,
                exog=x_train,
                order=(p, d, q),
                trend=trend_flag,
                enforce_invertibility=False,
                enforce_stationarity=False,
            )
            fitted = None
            for opt in optimizers:
                try:
                    fitted = model.fit(method=opt, maxiter=maxiter, disp=False)
                    break
                except Exception:
                    fitted = None
                    continue

            if fitted is None:
                continue

            try:
                y_pred_t = fitted.get_forecast(steps=n_test, exog=x_test).predicted_mean
                y_pred = _inverse_y(y_pred_t, y_transform, log_eps)
                mape_pct = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)

                if np.isfinite(mape_pct) and (mape_pct < best_mape):
                    best_mape = mape_pct
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted

            except Exception:
                continue

        if best_model is None:
            results.append({
                'X Set': x_label,
                'Order': None,
                'AIC Train': np.nan,
                'MAPE Test': np.nan,
                'N Train': 0,
                'N Test': 0,
                'N Total': n_total,
                'Note': 'No Convergent Model',
            })
            if verbose:
                print("  -> gagal konvergen untuk seluruh grid\n")
            continue

        results.append({
            'X Set': x_label,
            'Order': best_order,
            'AIC Train': best_aic,
            'MAPE Test': best_mape,
            'N Train': n_train,
            'N Test': n_test,
            'N Total': n_total,
            'Note': '',
        })
        fitted_models[x_label] = best_model

        if verbose:
            dt = time.time() - t0_pair
            print(f"  -> Model : {x_label}, order : {best_order}, MAPE : {best_mape:.2f}% ({dt:.2f}s)\n")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['MAPE Test'], na_position='last').reset_index(drop=True)

    if verbose:
        dt_all = time.time() - t0_all
        print(f"Selesai: {len(x_sets)} pasangan diuji dalam {dt_all:.2f}s.\n")

    return results_df, fitted_models


# ============================================================================
# Model Diagnostics & Selection  (Methodology §4)
# ============================================================================

def evaluate_model(
    res,
    df: pd.DataFrame,
    y_col: str = 'IDR',
    x_names: Optional[List[str]] = None,
    test_size: int = 12,
    y_transform: str = 'none',
    log_eps: float = 1e-6,
    alpha: float = 0.05,
    scale_exog: bool = True,
    figsize: Tuple[int, int] = (16, 6),
    show: bool = False,
    verbose: bool = False,
    confidence_interval: bool = True
):
    """
    Full diagnostic suite for a single fitted ARIMAX model.

    Produces an out-of-sample forecast with confidence interval, and runs
    the three standard ARIMA residual diagnostic tests plus a coefficient
    significance table, as specified in Methodology §4:

    - **Ljung-Box test** (§4.1) — H0: residuals are serially uncorrelated.
    - **Heteroskedasticity (H) test** (§4.2) — H0: residual variance is
      constant over time (Goldfeld-Quandt-type ratio of squared residuals
      across the first/second half of the sample).
    - **Jarque-Bera test** (§4.3) — H0: residuals are normally distributed.
    - **Coefficient significance / Wald test** (§4.4) — per-parameter
      t-test against H0: coefficient = 0.

    Parameters
    ----------
    res : SARIMAXResults
        A fitted model object, typically taken from the
        ``fitted_models`` dict returned by :func:`run_arima`.
    df : pd.DataFrame
        The same (or compatible) panel used to fit ``res``, used here to
        reconstruct the train/test split for forecasting and plotting.
    y_col : str, default 'IDR'
        Target column name.
    x_names : list of str, optional
        Exogenous predictor columns used by ``res``. Defaults to all
        columns prefixed ``'X'`` in ``df``.
    test_size : int, default 12
        Held-out forecast horizon, must match what was used when ``res``
        was originally fitted.
    y_transform : {'none','log','log1p','logit'}, default 'none'
        Must match the transform used to fit ``res`` (Methodology §2.1).
    log_eps : float, default 1e-6
        Numerical floor for the target transform.
    alpha : float, default 0.05
        Significance level used for every diagnostic test's decision rule
        (reject H0 if p-value < alpha).
    scale_exog : bool, default True
        Must match the scaling choice used to fit ``res`` (Methodology
        §2.2).
    figsize : tuple, default (16, 6)
        Actual-vs-predicted plot size.
    show : bool, default False
        If False, the figure is closed after creation (useful when batch-
        processing many models via :func:`evaluate_selected_models`).
    verbose : bool, default False
        If True, print the full SARIMAX summary table.
    confidence_interval : bool, default True
        If True, shade the forecast confidence interval on the plot.

    Returns
    -------
    dict
        ``{'metrics': {...}, 'forecast': {...}, 'residuals': {...},
        'tests': {...}, 'figure': matplotlib.Figure}`` — see inline keys for
        the out-of-sample MAPE, forecast series with CI bounds, in-sample
        and out-of-sample residuals, the three diagnostic test results
        (statistic, p-value, decision), the coefficient significance table,
        and the actual-vs-predicted figure.
    """
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    if x_names is None:
        x_names = [c for c in df.columns if c.startswith('X')]
    x_names = [x for x in x_names if x in df.columns]

    cols = [y_col] + x_names
    sub = df[cols].copy().dropna(subset=cols)
    n_total = len(sub)
    if n_total < (test_size + 1):
        raise ValueError("Observasi tidak cukup untuk melakukan train-test split.")

    n_test = test_size
    n_train = n_total - n_test

    y_train = sub[y_col].iloc[:n_train]
    y_test = sub[y_col].iloc[n_train:]
    x_train = sub[x_names].iloc[:n_train].copy()
    x_test = sub[x_names].iloc[n_train:].copy()

    if scale_exog:
        mu = x_train.mean()
        std = x_train.std(ddof=0).replace(0, 1)
        x_train = (x_train - mu) / std
        x_test = (x_test - mu) / std

    try:
        k_exog = 0 if res.model.exog is None else res.model.exog.shape[1]
    except Exception:
        k_exog = 0

    if k_exog > 0:
        if x_test is None:
            raise ValueError("Model menggunakan exog; harap berikan x_test untuk forecasting.")
        if x_test.shape[1] != k_exog:
            raise ValueError(f"Jumlah kolom exog tidak sesuai: model k_exog={k_exog}, x_test.shape[1]={x_test.shape[1]}")
        if len(x_test) != n_test:
            raise ValueError(f"Jumlah baris exog harus sama dengan steps (n_test). steps={n_test}, x_test rows={len(x_test)}")

    fc = res.get_forecast(steps=n_test, exog=x_test if k_exog > 0 else None)
    y_pred_t = fc.predicted_mean
    conf_t = fc.conf_int(alpha=alpha)

    y_pred_train_t = res.get_prediction(exog=x_train if k_exog > 0 else None).predicted_mean

    y_pred = _inverse_y(y_pred_t, y_transform, log_eps)
    y_pred_train = _inverse_y(y_pred_train_t, y_transform, log_eps)

    lower = upper = None
    if isinstance(conf_t, pd.DataFrame) and conf_t.shape[1] >= 2:
        lower = _inverse_y(conf_t.iloc[:, 0].values, y_transform, log_eps)
        upper = _inverse_y(conf_t.iloc[:, 1].values, y_transform, log_eps)

    y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
    y_pred_all = pd.concat([y_pred_train, y_pred]).reset_index(drop=True)

    mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)

    resid_train = pd.Series(res.resid, name="In Sample Residual")
    resid_oos = pd.Series(y_test.values - y_pred.values, index=y_test.index, name='resid_oos')

    summ = res.summary()
    summ_text = summ.as_text()
    summ_text = (summ_text.replace('\u2013', '-').replace('\u2014', '-'))

    def _grab_float(pattern: str, text: str) -> float:
        """Parse a single numeric value out of the SARIMAX text summary via regex."""
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return np.nan
        return np.nan

    # Diagnostic statistics parsed directly from the statsmodels summary table
    # (Ljung-Box Q & p-value, Heteroskedasticity H & p-value, Jarque-Bera JB,
    # p-value, skewness, kurtosis) — see Methodology §4.1-§4.3 for the
    # underlying formulas of each statistic.
    ljung_q = _grab_float(r"Ljung-Box.*?\(Q\).*?:\s*([-0-9.eE]+)", summ_text)
    ljung_p = _grab_float(r"Prob(?:\s*\(Q\)|\s*Q)\s*:\s*([-0-9.eE]+)", summ_text)
    het_h   = _grab_float(r"Heteroskedasticity\s*\(H\)\s*:\s*([-0-9.eE]+)", summ_text)
    het_p   = _grab_float(r"Prob\s*\(H\)(?:\s*\(two-sided\))?\s*:\s*([-0-9.eE]+)", summ_text)
    jb_stat = _grab_float(r"Jarque-Bera\s*\(JB\)\s*:\s*([-0-9.eE]+)", summ_text)
    jb_p    = _grab_float(r"Prob(?:\s*\(JB\)|\s*JB)\s*:\s*([-0-9.eE]+)", summ_text)
    skew    = _grab_float(r"Skew\s*:\s*([-0-9.eE]+)", summ_text)
    kurt    = _grab_float(r"Kurtosis\s*:\s*([-0-9.eE]+)", summ_text)

    def _decision(pval: float, alpha: float) -> str:
        """Standard reject/fail-to-reject decision rule against alpha."""
        if not (np.isfinite(pval)):
            return "Tidak dapat diputuskan (p-value NA)"
        return "Tolak H0" if pval < alpha else "Tak Tolak H0"

    decisions = {
        'ljung_box': {
            'H0': 'Tidak ada autokorelasi residual',
            'alpha': alpha,
            'stat': ljung_q,
            'pvalue': ljung_p,
            'decision': _decision(ljung_p, alpha),
        },
        'heteroskedasticity_H': {
            'H0': 'Homoskedastisitas (varian konstan)',
            'alpha': alpha,
            'stat': het_h,
            'pvalue': het_p,
            'decision': _decision(het_p, alpha),
        },
        'jarque_bera': {
            'H0': 'Residual berdistribusi normal',
            'alpha': alpha,
            'stat': jb_stat,
            'pvalue': jb_p,
            'decision': _decision(jb_p, alpha),
            'skew': skew,
            'kurtosis': kurt,
        }
    }

    coef_table_df = None
    try:
        for tbl in summ.tables:
            header = [h.strip() for h in tbl.header]
            if any('coef' in h.lower() for h in header) and any('p>' in h.lower() for h in header):
                rows = tbl.data
                colnames = ['param'] + [c.strip() for c in header if c.strip() != '']
                records = []
                for r in rows:
                    if len(r) < 2:
                        continue
                    param = r[0].strip()
                    vals = [v.strip() for v in r[1:]]
                    while len(vals) < len(colnames) - 1:
                        vals.append('')
                    records.append([param] + vals[:len(colnames)-1])
                coef_table_df = pd.DataFrame(records, columns=colnames)
                for c in coef_table_df.columns:
                    if c.lower() == 'param':
                        continue
                    coef_table_df[c] = pd.to_numeric(coef_table_df[c], errors='coerce')
                break
    except Exception:
        coef_table_df = None

    if coef_table_df is not None:
        pcol = None
        for c in coef_table_df.columns:
            if 'p>|' in c.lower():
                pcol = c
                break
        if pcol is not None:
            coef_table_df['signif'] = np.where(coef_table_df[pcol] < alpha, 'Signif', 'Not signif')
        else:
            pv = pd.Series(res.pvalues, name='p_value')
            pr = pd.Series(res.params, name='coef')
            tv = pd.Series(res.tvalues if hasattr(res, 'tvalues') else res.zvalues, name='t_or_z')
            coef_table_df = pd.concat([pr, tv, pv], axis=1)
            coef_table_df.columns = ['coef', 't_or_z', 'p_value']
            coef_table_df['signif'] = np.where(coef_table_df['p_value'] < alpha, 'Signif', 'Not signif')
            coef_table_df.insert(0, 'param', coef_table_df.index)
    else:
        pv = pd.Series(res.pvalues, name='p_value')
        pr = pd.Series(res.params, name='coef')
        tv = pd.Series(res.tvalues if hasattr(res, 'tvalues') else res.zvalues, name='t_or_z')
        coef_table_df = pd.concat([pr, tv, pv], axis=1)
        coef_table_df.columns = ['coef', 't_or_z', 'p_value']
        coef_table_df['signif'] = np.where(coef_table_df['p_value'] < alpha, 'Signif', 'Not signif')
        coef_table_df.insert(0, 'param', coef_table_df.index)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_all.loc[:n_train], label='Actual (Train)', color='red', lw=1.5)
    ax.plot(y_all.loc[n_train:], label='Actual (Test)', color='blue', lw=1.5)
    ax.plot(y_pred_all.loc[:n_train], label='Predicted (Train)', color='tab:blue', lw=1.5)
    ax.plot(y_pred_all.loc[n_train:], label='Predicted (Test)', color='green', lw=1.5)
    ax.axvline(x=n_train, color='gray', linestyle='--', lw=1)

    if confidence_interval:
        if lower is not None and upper is not None:
            ax.fill_between(y_all.index[n_train:], lower, upper, color='tab:blue', alpha=0.15,
                            label=f'CI ({(1-alpha):.0%})')

    ax.set_title('Actual vs Predicted')
    ax.set_xlabel('Posisi Waktu')
    ax.set_ylabel('Prepayment Rate')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    results = {
        'metrics': {'MAPE_%': mape},
        'forecast': {
            'y_pred': pd.Series(y_pred, index=y_test.index, name='y_pred'),
            'conf_int_original_scale': (
                pd.Series(lower, index=y_test.index, name='lower') if lower is not None else None,
                pd.Series(upper, index=y_test.index, name='upper') if upper is not None else None,
            ),
        },
        'residuals': {
            'train': resid_train,
            'oos': resid_oos,
        },
        'tests': {
            'ljung_box': {
                'stat': ljung_q, 'pvalue': ljung_p, 'alpha': alpha,
                'H0': 'Tidak ada autokorelasi residual',
                'decision': decisions['ljung_box']['decision'],
            },
            'heteroskedasticity_H': {
                'stat': het_h, 'pvalue': het_p, 'alpha': alpha,
                'H0': 'Homoskedastisitas (varian konstan)',
                'decision': decisions['heteroskedasticity_H']['decision'],
            },
            'jarque_bera': {
                'stat': jb_stat, 'pvalue': jb_p, 'alpha': alpha,
                'H0': 'Residual berdistribusi normal',
                'skew': skew, 'kurtosis': kurt,
                'decision': decisions['jarque_bera']['decision'],
            },
            'coef_significance': coef_table_df,
        },
        'figure': fig,
    }

    if verbose:
        print(res.summary())

    if show is False:
        plt.close(fig)

    return results


def evaluate_selected_models(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    models: Dict[str, object],
    y_col: str = 'IDR',
    mape_threshold: float = 25.0,
    test_size: int = 12,
    y_transform: str = 'none',
    log_eps: float = 1e-6,
    alpha: float = 0.05,
    scale_exog: bool = True,
    figsize: Tuple[int, int] = (18, 6),
    show: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Batch-run :func:`evaluate_model` diagnostics across every ARIMAX
    candidate from :func:`run_arima`'s leaderboard that passes an
    out-of-sample MAPE threshold.

    This produces a single side-by-side comparison table combining forecast
    accuracy *and* residual diagnostic validity (Methodology §4) — final
    model selection should not rely on MAPE alone, since a model with
    excellent accuracy but a failing Ljung-Box test (residual
    autocorrelation) or insignificant macro coefficients is not a sound
    *behavioral* model (it would not be economically interpretable, and its
    forecast confidence intervals would not be reliable).

    Parameters
    ----------
    df : pd.DataFrame
        Same panel used in :func:`run_arima`.
    results_df : pd.DataFrame
        Leaderboard returned by :func:`run_arima`.
    models : dict[str, SARIMAXResults]
        Fitted models dict returned by :func:`run_arima`.
    y_col : str, default 'IDR'
        Target column name.
    mape_threshold : float, default 25.0
        Only candidates with ``'MAPE Test' < mape_threshold`` (in %) are
        evaluated.
    test_size, y_transform, log_eps, alpha, scale_exog, figsize, show, verbose
        Passed through to :func:`evaluate_model` for each candidate.

    Returns
    -------
    pd.DataFrame
        One row per evaluated candidate: MAPE, Ljung-Box / Heteroskedasticity
        / Jarque-Bera decisions & p-values, residual skew/kurtosis, count of
        statistically significant exogenous coefficients, and the maximum
        p-value among them — sorted ascending by MAPE.
    """
    mape_col = 'MAPE Test'
    xkey_col = 'X Set'
    sel = results_df[results_df[mape_col] < mape_threshold].copy().reset_index(drop=True)

    rows_out: List[dict] = []

    for _, row in sel.iterrows():
        key = row[xkey_col]
        ord_tuple = row.get('Order', None)
        exogs = [s.strip() for s in key.split('+')]

        cols = [y_col] + exogs
        sub = df[cols].copy().dropna(subset=cols)
        n_total = len(sub)
        n_test = test_size
        if n_total < (n_test + 1):
            rows_out.append({
                'Model Key': key,
                'Order': ord_tuple,
                'MAPE (%)': np.nan,
                'LB Decision': 'Insufficient obs',
                'LB P-Value': np.nan,
                'H Decision': 'Insufficient obs',
                'H P-Value': np.nan,
                'JB Decision': 'Insufficient obs',
                'JB P-Value': np.nan,
                'Skew': np.nan,
                'Kurtosis': np.nan,
                'Significant X': np.nan,
                'Max p-value X': np.nan,
                'N train': 0,
                'N test': 0,
                'N total': n_total,
            })
            continue

        n_train = n_total - n_test

        try:
            eval_res = evaluate_model(
                res=models[key],
                df=df,
                y_col=y_col,
                x_names=exogs,
                test_size=test_size,
                y_transform=y_transform,
                log_eps=log_eps,
                alpha=alpha,
                scale_exog=scale_exog,
                figsize=figsize,
                show=show,
                verbose=verbose,
            )
        except Exception as e:
            rows_out.append({
                'Model Key': key,
                'Order': ord_tuple,
                'MAPE (%)': np.nan,
                'LB Decision': f'Error: {str(e)}',
                'LB P-Value': np.nan,
                'H Decision': f'Error: {str(e)}',
                'H P-Value': np.nan,
                'JB Decision': f'Error: {str(e)}',
                'JB P-Value': np.nan,
                'Skew': np.nan,
                'Kurtosis': np.nan,
                'Significant X': np.nan,
                'Max p-value X': np.nan,
                'N train': n_train,
                'N test': n_test,
                'N total': n_total,
            })
            continue

        metrics = eval_res['metrics']
        tests = eval_res['tests']

        lb = tests.get('ljung_box', {})
        het = tests.get('heteroskedasticity_H', {})
        jb = tests.get('jarque_bera', {})
        coef_tbl = tests.get('coef_significance', None)

        exog_signif_cnt = np.nan
        exog_pval_max = np.nan
        if isinstance(coef_tbl, pd.DataFrame):
            pcol = None
            for c in coef_tbl.columns:
                if 'p>|' in c.lower() or c.lower() == 'p_value':
                    pcol = c
                    break

            if pcol is not None and 'param' in coef_tbl.columns:
                exog_mask = coef_tbl['param'].astype(str).str.strip().isin(exogs)
                exog_pvals = coef_tbl.loc[exog_mask, pcol]
                exog_signif_cnt = int((exog_pvals < alpha).sum()) if exog_pvals.size > 0 else np.nan
                exog_pval_max = float(exog_pvals.max()) if exog_pvals.size > 0 else np.nan

        rows_out.append({
            'Model Key': key,
            'Order': ord_tuple,
            'MAPE (%)': metrics.get('MAPE_%', np.nan),
            'LB Decision': lb.get('decision', None),
            'LB P-Value': lb.get('pvalue', np.nan),
            'H Decision': het.get('decision', None),
            'H P-Value': het.get('pvalue', np.nan),
            'JB Decision': jb.get('decision', None),
            'JB P-Value': jb.get('pvalue', np.nan),
            'Skew': jb.get('skew', np.nan),
            'Kurtosis': jb.get('kurtosis', np.nan),
            'Significant X': exog_signif_cnt,
            'Max p-value X': exog_pval_max,
            'N train': n_train,
            'N test': n_test,
            'N total': n_total,
        })

    diagnostics_df = pd.DataFrame(rows_out)
    diagnostics_df = diagnostics_df.sort_values(by=['MAPE (%)'], na_position='last').reset_index(drop=True)
    return diagnostics_df


# ============================================================================
# Feature Engineering & Screening  (Methodology §2.3, §2.4, §3)
# ============================================================================

def add_ema(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    spans: Sequence[int] = (3, 6, 12),
    lags: Sequence[int] = (0,),
    min_periods: Optional[int] = None,
    adjust: bool = False,
    skip_binary: bool = True,
    keep_original: bool = True,
    prefix: Optional[str] = None,
    in_place: bool = False,
) -> pd.DataFrame:
    """
    Add Exponential Moving Average (EMA) smoothed — and optionally lagged —
    versions of macroeconomic predictor columns.

    Implements EMA_t = alpha * x_t + (1-alpha) * EMA_{t-1}, with
    alpha = 2 / (span + 1); see Methodology §2.4 for the closed-form
    (unrolled) derivation showing this is an exponentially-decaying weighted
    average of all past observations. Multiple spans (default 3/6/12
    months) capture short/medium/long-term macro trend persistence, used to
    reduce the influence of point-in-time noise in raw macro releases
    relative to the underlying trend.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel.
    cols : sequence of str, optional
        Columns to smooth. Defaults to all columns prefixed ``'X'``.
    spans : sequence of int, default (3, 6, 12)
        EMA spans (in periods) to compute; one output column per span.
    lags : sequence of int, default (0,)
        Additional lag(s) to apply on top of each EMA series (0 = no extra
        lag).
    min_periods : int, optional
        Minimum periods required to produce an EMA value; defaults to the
        span itself if not provided.
    adjust : bool, default False
        Passed to ``pandas.Series.ewm`` — whether to use the (non-recursive)
        weighted-average form or the exact recursive form described above.
    skip_binary : bool, default True
        Skip columns detected as binary/dummy (e.g. an anomaly indicator),
        since smoothing a 0/1 flag is not meaningful.
    keep_original : bool, default True
        If False, drop the raw (unsmoothed) source columns after adding the
        EMA-derived columns.
    prefix : str, optional
        Override the naming prefix used for generated columns (defaults to
        the source column name).
    in_place : bool, default False
        If True, mutate ``df`` directly instead of operating on a copy.

    Returns
    -------
    pd.DataFrame
        ``df`` augmented with new columns named ``"{col}_EMA{span}"``
        (and ``"..._L{lag}"`` suffix for lag > 0).
    """
    if not in_place:
        out = df.copy()
    else:
        out = df

    # Tentukan daftar kolom target
    if cols is None:
        cols = [c for c in out.columns if c.startswith('X')]
    else:
        cols = [c for c in cols if c in out.columns]

    if min_periods is None:
        pass

    # Utility deteksi binary
    def _is_binary(series: pd.Series) -> bool:
        """Detect whether a series is effectively a 0/1 binary/dummy indicator."""
        vals = series.dropna().unique()
        if len(vals) == 0:
            return False
        # izinkan floating 0/1 (mis. float64 setelah scale)
        return len(vals) <= 3 and set(np.round(vals, 6)).issubset({0.0, 1.0})

    new_cols: List[str] = []
    for c in cols:
        s = out[c].astype(float)

        if skip_binary and _is_binary(s):
            continue

        for span in spans:
            mp = span if min_periods is None else int(min_periods)
            ema = s.ewm(span=span, adjust=adjust, min_periods=mp).mean()

            base_name = f"{prefix or c}_EMA{span}"
            # buat lag-lag
            for L in sorted(set(lags)):
                col_name = base_name if L == 0 else f"{base_name}_L{L}"
                out[col_name] = ema.shift(L) if L > 0 else ema
                new_cols.append(col_name)

    if not keep_original:
        out = out.drop(columns=list(cols), errors='ignore')

    return out


def cek_vif(
    df: pd.DataFrame,
    x_list: list):
    """
    Variance Inflation Factor (VIF) multicollinearity check.

    For each variable in ``x_list``, regresses it on all remaining
    variables and computes VIF_j = 1 / (1 - R_j^2), where R_j^2 is the
    coefficient of determination of that auxiliary regression (Methodology
    §3.2). VIF_j = 1 indicates no collinearity; conventional rules of thumb
    flag VIF_j > 5 (moderate) or > 10 (severe) concern — relevant here
    since several macroeconomic series are naturally correlated (e.g.
    policy rate and inflation), which can otherwise destabilize the ARIMAX
    coefficient estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel containing the candidate predictor columns.
    x_list : list of str
        Columns to check for pairwise/multi-way multicollinearity.

    Returns
    -------
    pd.DataFrame
        Columns ``['Variabel', 'VIF']``, one row per variable in
        ``x_list``.
    """
    sub = df[x_list]
    sub.dropna(inplace = True)

    vif_data = pd.DataFrame()
    vif_data['Variabel'] = sub.columns

    vif_data['VIF'] = [variance_inflation_factor(sub.values, i) for i in range(len(sub.columns))]

    return vif_data


def find_correl(
    df,
    target_col = 'IDR',
    lag_grid = [0,1,2,3],
    correl_threshold = 0.5
):
    """
    Lag-optimal Pearson correlation screening between the target and every
    candidate macro predictor.

    For each macro column (prefixed ``'X'``), scans ``lag_grid`` and
    selects the lag maximizing absolute correlation with the target
    (Methodology §3.1). This is a fast, model-free screening step used to
    shortlist candidate predictors *before* committing to the full ARIMAX
    combinatorial search in :func:`run_arima`, reducing the search space.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel containing the target column and candidate ``'X*'``
        predictor columns.
    target_col : str, default 'IDR'
        Target (behavioral rate) column name.
    lag_grid : list of int, default [0,1,2,3]
        Candidate lags (in periods) to test for each predictor.
    correl_threshold : float, default 0.5
        Minimum absolute correlation (at the optimal lag) required for a
        predictor to be retained in the output.

    Returns
    -------
    pd.DataFrame
        Columns ``['Variabel', 'Lag Optimum', 'Correl Optimum']``, filtered
        to predictors exceeding ``correl_threshold``.
    """
    result_df = []
    summary = []

    for c in df.columns:
        if c.startswith('X'):

            best_corr = 0

            for l in lag_grid:
                c_diff = df[c].shift(l)
                corr = df[target_col].corr(c_diff, method = 'pearson')

                if np.abs(corr) > np.abs(best_corr):
                    best_corr = corr
                    best_lag = l

            summary.append({
                'Variabel' : c,
                'Lag Optimum' : best_lag,
                'Correl Optimum' : best_corr
            })

        else:
            continue

    result_df = pd.DataFrame(summary)

    result_df = result_df[np.abs(result_df['Correl Optimum']) > correl_threshold].reset_index(drop = True)

    return result_df


def run_pca(
    df,
    standardize = True,
    var_threshold = 0.80,
    n_components = None,
    show_plot = False
):
    """
    Principal Component Analysis (PCA) dimensionality reduction of the
    macroeconomic predictor set.

    An optional alternative to variable-subset selection when
    multicollinearity among macro predictors (see :func:`cek_vif`) cannot
    be resolved through screening alone. Standardizes the ``'X*'`` columns
    (optional), fits PCA, and retains either a user-specified number of
    components or the minimum number required to explain
    ``var_threshold`` of total variance (Methodology §3.3).

    Parameters
    ----------
    df : pd.DataFrame
        Input panel containing candidate ``'X*'`` predictor columns (and
        optionally a ``'Date'`` column, carried through to the output).
    standardize : bool, default True
        Whether to z-score standardize predictors prior to PCA (recommended
        — PCA is scale-sensitive).
    var_threshold : float, default 0.80
        Minimum cumulative explained variance ratio required when
        ``n_components`` is not explicitly specified.
    n_components : int, optional
        If provided, overrides ``var_threshold`` and retains exactly this
        many components.
    show_plot : bool, default False
        If True, additionally render a bar chart of the full (unretained)
        explained variance ratio spectrum.

    Returns
    -------
    scores_df : pd.DataFrame
        Component scores (one row per observation), columns ``PC1...PCk``.
    loadings_df : pd.DataFrame
        Component loadings (eigenvectors), rows = components, columns =
        original predictor names.
    evr_df : pd.DataFrame
        Explained variance ratio (and cumulative EVR) per retained
        component.
    """
    # Ambil kolom X
    x_cols = df[[c for c in df.columns if c.startswith('X')]]

    # Standardize
    if standardize:
        scaler = StandardScaler()
        x_std = scaler.fit_transform(x_cols)
    else:
        scaler = None
        x_std = x_cols.values

    # PCA
    p_full = min(x_cols.shape[1], x_cols.shape[0])
    pca_full = PCA(n_components = p_full)
    pca_full.fit(x_std)

    evr_full = pca_full.explained_variance_ratio_
    evr_full_cum = np.cumsum(evr_full)

    # Select Component
    if n_components is None:
        k = int(np.searchsorted(evr_full_cum, var_threshold) + 1)
    else:
        k = int(n_components)
    k = max(1, min(k, p_full))

    # Fit PCA Final
    pca = PCA(n_components = k)
    scores = pca.fit_transform(x_std)
    loadings = pca.components_

    # Assign DataFrame
    pc_names = [f'PC{i}' for i in range(1, k + 1)]
    scores_df = pd.DataFrame(scores, index = df.index, columns = pc_names)
    if 'Date' in df.columns:
        scores_df.insert(0, 'Date', df['Date'].values)

    loadings_df = pd.DataFrame(loadings, index = pc_names, columns = x_cols.columns)

    evr_k = pca.explained_variance_ratio_
    evr_k_cum = np.cumsum(evr_k)

    evr_df = pd.DataFrame({
        'Komponen' : pc_names,
        'EVR' : evr_k,
        'EVR Cum.' : evr_k_cum
    })

    # Plot
    if show_plot:
        plt.figure(figsize=(9, 4.5))
        plt.title('Cumulative Variance Explained (EVR)')
        plt.bar(range(1, len(evr_full)+1), evr_full, color='blue')

    return scores_df, loadings_df, evr_df


def _make_lag(df, cols, lags=(1,2), prefix=None):
    """
    Add lagged copies of the given columns (Methodology §2.3).

    Used internally by :func:`run_hgbr` to construct lagged macro-predictor
    and autoregressive target features, since — unlike ARIMAX, where the
    AR/MA polynomial implicitly handles serial dependence — a tree-based ML
    model has no native notion of a time index and must be given explicit
    lag features.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel.
    cols : sequence of str
        Columns to lag.
    lags : sequence of int, default (1, 2)
        Lag orders to generate.
    prefix : str, optional
        Override naming prefix for generated columns (defaults to the
        source column name).

    Returns
    -------
    pd.DataFrame
        ``df`` (copy) augmented with ``"{col}_L{lag}"`` columns.
    """
    out = df.copy()
    for c in cols:
        for L in lags:
            out[f"{prefix or c}_L{L}"] = df[c].shift(L)
    return out


# ============================================================================
# Benchmark Model — HistGradientBoostingRegressor  (Methodology §6)
# ============================================================================

def run_hgbr(
    df: pd.DataFrame,
    y_col: str = 'IDR',
    x_list: Optional[List[str]] = None,
    # Feature Lagging
    y_lags: Sequence[int] = (1, 2),
    x_lags: Sequence[int] = (0, 1, 2),
    # Train-Test split
    test_size: int = 12,
    # CV (rolling window)
    cv_type: str = "rolling",
    cv_folds: int = 5,
    cv_fold_len: int = 12,
    rolling_train_window: int = 60,
    min_train_for_cv: int = 36,
    # ML Parameter
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    scale_x: bool = True,
    random_state: int = 42,
    # Transform & Threshold
    y_transform: str = 'logit',
    log_eps: float = 1e-6,
    mape_eps: float = 1e-8,
    # Logging
    verbose: bool = True,
    # Run Single Model
    run_mode: str = 'cv',
    best_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    HistGradientBoostingRegressor (HGBR) benchmark model, tuned via
    time-series cross-validation, as an independent non-linear check
    against the ARIMAX pipeline (Methodology §6).

    Features are constructed from lagged macro predictors (``x_lags``) and
    autoregressive lags of the (transformed) target itself (``y_lags``).
    Hyperparameters are tuned via **rolling** or **expanding** window
    time-series CV (never standard random k-fold, which would leak future
    information into training folds for a time-dependent process).

    Parameters
    ----------
    df : pd.DataFrame
        Input panel with the target column and candidate ``'X*'``
        predictor columns.
    y_col : str, default 'IDR'
        Target (behavioral rate) column name.
    x_list : list of str, optional
        Candidate macro predictor columns; defaults to all ``'X*'``
        columns.
    y_lags : sequence of int, default (1, 2)
        Autoregressive lags of the (transformed) target to include as
        features.
    x_lags : sequence of int, default (0, 1, 2)
        Lags of each macro predictor to include as features (0 = 
        contemporaneous).
    test_size : int, default 12
        Held-out forecast horizon (final evaluation, not used in CV).
    cv_type : {'rolling', 'expanding'}, default 'rolling'
        Time-series CV scheme. 'rolling' uses a fixed-size sliding training
        window (``rolling_train_window``); 'expanding' uses all data prior
        to each validation fold.
    cv_folds : int, default 5
        Number of validation folds.
    cv_fold_len : int, default 12
        Length (in periods) of each validation fold.
    rolling_train_window : int, default 60
        Training window length (in periods) for 'rolling' CV.
    min_train_for_cv : int, default 36
        Minimum training-window length required for a CV fold to be valid;
        if the available training data cannot satisfy
        ``min_train_for_cv + cv_folds * cv_fold_len``, the function
        automatically falls back to a single hold-out validation split
        (see ``use_cv`` fallback logic).
    param_grid : dict, optional
        Hyperparameter grid for ``HistGradientBoostingRegressor``
        (searched via a manual rolling-CV loop, not
        ``sklearn.GridSearchCV``, since the latter does not natively
        support rolling-window time-series CV with a fixed window length).
        Defaults to a grid over ``max_depth``, ``learning_rate``,
        ``max_iter``, ``loss``, ``l2_regularization``, ``min_samples_leaf``,
        etc.
    scale_x : bool, default True
        Standardize features (z-score) inside the pipeline.
    random_state : int, default 42
        Reproducibility seed.
    y_transform : {'none','log','log1p','logit'}, default 'logit'
        Target transform (Methodology §2.1); defaults to 'logit' since the
        target is a bounded rate.
    log_eps : float, default 1e-6
        Numerical floor for the target transform.
    mape_eps : float, default 1e-8
        Numerical floor applied to the true value denominator when
        computing MAPE, avoiding division by (near-)zero.
    verbose : bool, default True
        Print progress per hyperparameter combination / CV fold.
    run_mode : {'cv', 'single', 'all'}, default 'cv'
        - 'cv'     : full hyperparameter search via time-series CV (as
          described above), then refit on the full training window with
          the best-found parameters, evaluated on the held-out test set.
        - 'single' : skip the search; fit once on the training window
          using ``best_params`` (typically taken from a prior 'cv' run),
          evaluated on the held-out test set. Requires ``best_params``.
        - 'all'    : fit once on the *entire* available dataset (train +
          test) using ``best_params`` — used to produce the final
          production model once hyperparameters have been finalized via
          'cv'/'single' on historical hold-out data. Requires
          ``best_params``.

    Returns
    -------
    dict
        Fitted pipeline, best hyperparameters, CV/train/test MAPE, feature
        list, the engineered feature matrix itself (``'X_features_model'``
        — required by :func:`feature_importance_permutation` so that
        permutation importance is computed on the exact same lag/EMA
        feature set the model was trained on), and train/test true &
        predicted series (original rate scale) — structure depends on
        ``run_mode`` (see inline keys).
    """

    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    # Target & transform (pakai util Mas yang sudah ada di file)
    y = df[y_col].astype(float)
    y_t = _transform_y(y, y_transform, log_eps)  # logit

    if x_list is None:
        x_list = [c for c in df.columns if c.startswith('X')]
    X_base = df[x_list].copy() if len(x_list) > 0 else pd.DataFrame(index=df.index)

    X_all = X_base.copy()
    if x_lags and any(L != 0 for L in x_lags):
        for L in sorted({L for L in x_lags if L > 0}):
            X_all = _make_lag(X_all, X_base.columns.tolist(), lags=(L,), prefix=None)

    if y_lags:
        for L in sorted(set(y_lags)):
            X_all[f'Yt_L{L}'] = y_t.shift(L)

    data = pd.concat([y, y_t.rename('Yt'), X_all], axis=1).dropna()
    data_index_model = data.index.copy()
    n_total_model = len(data)
    if n_total_model <= test_size:
        raise ValueError("Observasi tidak cukup untuk split train-test pada HGBR.")

    # Split hold-out pada DATA MODEL
    n_train_model = n_total_model - test_size
    y_all = data[y_col].copy()
    y_all_t = data['Yt'].copy()
    X_all_fea = data.drop(columns=['Yt', y_col])

    X_tr, X_te = X_all_fea.iloc[:n_train_model], X_all_fea.iloc[n_train_model:]
    y_tr_t, y_te_t = y_all_t.iloc[:n_train_model], y_all_t.iloc[n_train_model:]
    y_tr, y_te = y_all.iloc[:n_train_model], y_all.iloc[n_train_model:]

    if run_mode.lower() in ("single", "all"):
        if best_params is None or len(best_params) == 0:
            raise ValueError("best_params kosong. Berikan hyper-parameter terbaik untuk run_mode != 'cv'.")

        steps = []
        if scale_x:
            steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))
        steps.append(('hgbr', HistGradientBoostingRegressor(random_state=random_state)))
        pipe = Pipeline(steps)
        pipe.set_params(**best_params)

        if run_mode.lower() == "single":
            pipe.fit(X_tr, y_tr_t)
            y_pred_tr = _inverse_y(pipe.predict(X_tr), y_transform, log_eps)
            y_pred_te = _inverse_y(pipe.predict(X_te), y_transform, log_eps)

            m_tr = float(mean_absolute_percentage_error(
                np.clip(y_tr.values, mape_eps, None), np.asarray(y_pred_tr)) * 100.0)
            m_te = float(mean_absolute_percentage_error(
                np.clip(y_te.values, mape_eps, None), np.asarray(y_pred_te)) * 100.0)

            if verbose:
                print(f"[HGBR SINGLE] MODEL obs={n_total_model} | Train={n_train_model}, Test={test_size}")
                print(f"[HGBR SINGLE] Train MAPE={m_tr:.2f}% | Test MAPE={m_te:.2f}%")

            return {
                'mode'            : 'single',
                'best_estimator'  : pipe,
                'best_params'     : best_params,
                'cv_best_mape_%'  : np.nan,
                'features'        : X_all_fea.columns.tolist(),
                'X_features_model': X_all_fea,  # engineered feature matrix (lags/EMA/Yt), needed by feature_importance_permutation
                'data_index_model': data_index_model,
                'n_total_model'   : int(n_total_model),
                'n_train_model'   : int(n_train_model),
                'y_true_train'    : pd.Series(y_tr.values, index=y_tr.index, name='y_train'),
                'y_true_test'     : pd.Series(y_te.values, index=y_te.index, name='y_test'),
                'y_pred_train'    : pd.Series(np.asarray(y_pred_tr), index=y_tr.index, name='y_pred_train'),
                'y_pred_test'     : pd.Series(np.asarray(y_pred_te), index=y_te.index, name='y_pred_test'),
                'metrics_train'   : {'MAPE_%': float(m_tr)},
                'metrics_test'    : {'MAPE_%': float(m_te)},
            }

        else:
            X_all = X_all_fea
            y_all_t = data['Yt']
            pipe.fit(X_all, y_all_t)

            y_fit = _inverse_y(pipe.predict(X_all), y_transform, log_eps)
            m_fit = float(mean_absolute_percentage_error(
                np.clip(data[y_col].values, mape_eps, None), np.asarray(y_fit)) * 100.0)

            return {
                'mode'            : 'all',
                'best_estimator'  : pipe,
                'best_params'     : best_params,
                'cv_best_mape_%'  : np.nan,
                'features'        : X_all_fea.columns.tolist(),
                'X_features_model': X_all_fea,  # engineered feature matrix (lags/EMA/Yt), needed by feature_importance_permutation
                'data_index_model': data_index_model,
                'n_total_model'   : int(n_total_model),
                'n_train_model'   : int(n_total_model), # seluruh data dipakai untuk fit
                'y_true_train'    : pd.Series(data[y_col].values, index=data.index, name='y_all'),
                'y_true_test'     : pd.Series(dtype=float), # kosong
                'y_pred_train'    : pd.Series(np.asarray(y_fit), index=data.index, name='y_fit'),
                'y_pred_test'     : pd.Series(dtype=float), # kosong
                'metrics_train'   : {'MAPE_%': float(m_fit)},
                'metrics_test'    : {'MAPE_%': np.nan},
            }

    steps = []
    if scale_x:
        steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))
    steps.append(('hgbr', HistGradientBoostingRegressor(random_state=random_state)))
    base_pipe = Pipeline(steps)

    # Parameter grid default (ringkas)
    if param_grid is None:
        param_grid = {
            'hgbr__max_depth': [2, 3, 4],
            'hgbr__learning_rate': [0.03, 0.05],
            'hgbr__max_iter': [500, 800],
            'hgbr__loss': ['absolute_error', 'squared_error'],
            'hgbr__early_stopping': [True],
            'hgbr__l2_regularization': [0.5, 1.0, 2.0],
            'hgbr__min_samples_leaf': [20, 40, 80],
            'hgbr__validation_fraction': [0.15, 0.2],
            'hgbr__n_iter_no_change': [20, 30],
            'hgbr__max_bins': [32]
        }
    grid = list(ParameterGrid(param_grid))

    use_cv = (
        (cv_type is not None) and (cv_folds > 1) and (cv_fold_len > 0) and
        (n_train_model >= (min_train_for_cv + cv_folds * cv_fold_len))
    )

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    if use_cv:
        total_val = cv_folds * cv_fold_len
        first_val_start = max(min_train_for_cv, n_train_model - total_val)
        for k in range(cv_folds):
            va_start = first_val_start + k * cv_fold_len
            va_end = va_start + cv_fold_len
            if va_end > n_train_model:
                break

            if cv_type == 'rolling':
                tr_end = va_start
                tr_start = max(0, tr_end - rolling_train_window)
            else:
                tr_start = 0

            tr_idx = np.arange(tr_start, va_start)
            va_idx = np.arange(va_start, va_end)

            if len(tr_idx) >= min_train_for_cv and len(va_idx) == cv_fold_len:
                folds.append((tr_idx, va_idx))
        if len(folds) == 0:
            use_cv = False

    need = min_train_for_cv + cv_folds * cv_fold_len

    if verbose:
        print(f"[CV Check] n_train={n_train_model}, requirement={need} "
              f"-> {'OK' if use_cv else 'NOT OK (fallback to simple validation)'}")

    if verbose:
        print(f"Total observasi (MODEL): {n_total_model} | Train={n_train_model}, Test={test_size}")
        print(f"Fitur total: {X_all_fea.shape[1]} (X={len(x_list)}, x_lags={x_lags}, y_lags={y_lags})")
        print(f"Parameter combinations: {len(grid)}; CV folds: {len(folds) if use_cv else 0}\n")

    best_params: Optional[Dict[str, Any]] = None
    best_cv_mape: float = np.inf

    for idx, params in enumerate(grid, start=1):
        t0 = time.time()
        pipe = clone(base_pipe)
        pipe.set_params(**params)

        if verbose:
            print(f"[{idx}/{len(grid)}] Param: {params}")
            print(f"observasi (train, MODEL): {len(y_tr_t)}")

        scores: List[float] = []

        if use_cv and len(folds) > 0:
            for k, (tr_idx, va_idx) in enumerate(folds, start=1):
                fold_m = np.nan
                try:
                    X_tr_k, X_va_k = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
                    y_tr_t_k       = y_tr_t.iloc[tr_idx]    # logit
                    y_va_k         = y_tr.iloc[va_idx]      # skala asli

                    pipe.fit(X_tr_k, y_tr_t_k)
                    y_pred_va_t = pipe.predict(X_va_k)
                    y_pred_va = _inverse_y(y_pred_va_t, y_transform, log_eps) # back to original

                    # MAPE sklearn di skala asli (clip y_true)
                    fold_m = float(mean_absolute_percentage_error(np.clip(y_va_k.values, mape_eps, None),
                                                                  np.asarray(y_pred_va)) * 100.0)
                    if np.isfinite(fold_m):
                        scores.append(fold_m)
                except Exception as e:
                    if verbose:
                        print(f"      -> Fold {k}/{len(folds)} ERROR: {e}")

                if verbose:
                    msg = f"{fold_m:.2f}%" if np.isfinite(fold_m) else "NA"
                    print(f"      -> Fold {k}/{len(folds)} "
                          f"[{cv_type}, train={len(tr_idx)}]: MAPE={msg}")

            cv_mape = float(np.mean(scores)) if len(scores) > 0 else np.inf
        else:
            # Fallback tanpa CV: pakai MAPE train sebagai proxy
            va_len = int(min(max(8, cv_fold_len), max(8, n_train_model // 5)))
            va_start = n_train_model - va_len
            tr_start = max(0, va_start - rolling_train_window)

            X_tr_k, X_va_k = X_tr.iloc[tr_start:va_start], X_tr.iloc[va_start:n_train_model]
            y_tr_t_k       = y_tr_t.iloc[tr_start:va_start]

            y_va_k         = y_tr.iloc[va_start:n_train_model]

            pipe.fit(X_tr_k, y_tr_t_k)
            y_pred_va_t = pipe.predict(X_va_k)
            y_pred_va = _inverse_y(y_pred_va_t, y_transform, log_eps)

            cv_mape = float(mean_absolute_percentage_error(
                np.clip(y_va_k.values, mape_eps, None),
                np.asarray(y_pred_va)
            ) * 100.0)

        dt = time.time() - t0
        if verbose:
            cv_msg = f"{cv_mape:.2f}%" if np.isfinite(cv_mape) else "NA"
            print(f"   -> CV-MAPE: {cv_msg}    ({dt:.2f}s)\n")

        # Seleksi param terbaik berdasar CV-MAPE
        if np.isfinite(cv_mape) and (cv_mape < best_cv_mape):
            best_cv_mape = cv_mape
            best_params = params

    if best_params is None:
        raise RuntimeError("Tidak ada kombinasi parameter yang berhasil pada CV.")

    if verbose:
        print("--- Param terbaik (CV-MAPE) ---")
        print(best_params, f"\nCV-MAPE terbaik: {best_cv_mape:.2f}%\n")

    best_pipe = clone(base_pipe)
    best_pipe.set_params(**best_params)
    best_pipe.fit(X_tr, y_tr_t)

    # Prediksi TRAIN & TEST (skala asli)
    y_pred_tr_t = best_pipe.predict(X_tr)
    y_pred_te_t = best_pipe.predict(X_te)
    y_pred_tr   = _inverse_y(y_pred_tr_t, y_transform, log_eps)
    y_pred_te   = _inverse_y(y_pred_te_t, y_transform, log_eps)

    mape_train = float(mean_absolute_percentage_error(np.clip(y_tr.values, mape_eps, None),
                                                      np.asarray(y_pred_tr)) * 100.0)
    mape_test  = float(mean_absolute_percentage_error(np.clip(y_te.values, mape_eps, None),
                                                      np.asarray(y_pred_te)) * 100.0)

    if verbose:
        print(f"HGBR - Train MAPE: {mape_train:.2f}%")
        print(f"HGBR - Test MAPE: {mape_test:.2f}%")

    out = {
        'best_estimator': best_pipe,
        'best_params': best_params,
        'cv_best_mape_%': float(best_cv_mape),
        'features': X_all_fea.columns.tolist(),
        'X_features_model': X_all_fea,  # engineered feature matrix (lags/EMA/Yt), needed by feature_importance_permutation
        # Info indeks MODEL (untuk evaluator/plot)
        'data_index_model': data_index_model,
        'n_total_model': int(n_total_model),
        'n_train_model': int(n_train_model),
        # Series utk metrik/plot (index = index MODEL yang sudah terpotong)
        'y_true_train': pd.Series(y_tr.values, index=y_tr.index, name='y_train'),
        'y_true_test': pd.Series(y_te.values, index=y_te.index, name='y_test'),
        'y_pred_train': pd.Series(np.asarray(y_pred_tr), index=y_tr.index, name='y_pred_train'),
        'y_pred_test': pd.Series(np.asarray(y_pred_te), index=y_te.index, name='y_pred_test'),
        'metrics_train': {'MAPE_%': float(mape_train)},
        'metrics_test': {'MAPE_%': float(mape_test)},
    }
    return out


def evaluate_hgbr(out: Dict[str, Any], df: pd.DataFrame, y_col: str = 'IDR',
                  figsize: Tuple[int, int] = (16, 6), show: bool = True):
    """
    Actual-vs-predicted visualization for a fitted HGBR model (output of
    :func:`run_hgbr`).

    Parameters
    ----------
    out : dict
        Output dict from :func:`run_hgbr`.
    df : pd.DataFrame
        Same panel used to fit the HGBR model (used to recover the original
        target scale / index alignment).
    y_col : str, default 'IDR'
        Target column name.
    figsize : tuple, default (16, 6)
        Plot size.
    show : bool, default True
        If False, the figure is closed after creation.

    Returns
    -------
    matplotlib.figure.Figure
        Actual vs. predicted (train/test) plot.
    """
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    idx_model = out['data_index_model']
    n_total_mod = out['n_total_model']
    n_train_mod = out['n_train_model']

    y_all_model = df.loc[idx_model, y_col].astype(float).reset_index(drop=True)

    pred_all = pd.Series([np.nan]*n_total_mod, name='y_pred_all', dtype=float)
    pred_all.iloc[:n_train_mod] = out['y_pred_train'].values
    pred_all.iloc[n_train_mod:] = out['y_pred_test'].values

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_all_model.iloc[:n_train_mod], label='Actual (Train)', color='red', lw=1.5)
    ax.plot(y_all_model.iloc[n_train_mod:], label='Actual (Test)', color='blue', lw=1.5)
    ax.plot(pred_all.iloc[:n_train_mod], label='Predicted (Train)', color='tab:blue', lw=1.5)
    ax.plot(pred_all.iloc[n_train_mod:], label='Predicted (Test)', color='green', lw=1.5)
    ax.axvline(x=n_train_mod, color='gray', linestyle='--', lw=1)
    ax.set_title('Actual vs Predicted (HGBR)')
    ax.set_xlabel('Posisi Waktu')
    ax.set_ylabel('Prepayment Rate')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if not show:
        plt.close(fig)
    return fig


def feature_importance_permutation(
    out: Dict[str, Any],
    df: pd.DataFrame,
    y_col: str,
    n_repeats: int = 30,
    random_state: int = 42,
    mape_eps: float = 1e-8,
    y_transform: str = 'logit',
    log_eps: float = 1e-6,
    use_test_only: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Permutation feature importance for a fitted HGBR model (Methodology §6).

    For each feature, its values are randomly shuffled in the evaluation
    set and the resulting increase in MAPE is recorded:
    ``Importance_j = MAPE(y, f(X_perm(j))) - MAPE(y, f(X))``, averaged over
    ``n_repeats`` repetitions to reduce noise from a single random
    permutation. A large positive value indicates the model relies heavily
    on that feature. Used as an interpretability cross-check against the
    ARIMAX coefficient significance table (Methodology §4.4) — i.e. do the
    two independent modeling tracks agree on which macro drivers matter?

    Parameters
    ----------
    out : dict
        Output dict from :func:`run_hgbr`.
    df : pd.DataFrame
        Same panel used to fit the HGBR model.
    y_col : str
        Target column name.
    n_repeats : int, default 30
        Number of random shuffles per feature.
    random_state : int, default 42
        Reproducibility seed.
    mape_eps : float, default 1e-8
        Numerical floor for the MAPE denominator.
    y_transform : {'none','log','log1p','logit'}, default 'logit'
        Must match the transform used to fit the model in ``out``.
    log_eps : float, default 1e-6
        Numerical floor for the target transform.
    use_test_only : bool, default True
        If True (recommended), compute importance on the held-out test set
        only, avoiding an over-optimistic in-sample importance estimate.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns ``['feature', 'importance_mean', 'importance_std',
        'importance_norm']``, sorted descending by ``importance_mean``.
        ``importance_norm`` is a [0,1]-rescaled version for convenient
        ranking/plotting.
    """
    est = out['best_estimator']
    feat_names = out['features']
    idx_model = out['data_index_model']
    n_train = int(out['n_train_model'])

    if 'X_features_model' in out:
        # Preferred path: reuse the exact engineered feature matrix (lags/EMA/Yt)
        # produced inside run_hgbr, guaranteeing feature alignment regardless of
        # x_lags/y_lags settings.
        X_all = out['X_features_model'][feat_names].copy()
        y_all = pd.concat([out['y_true_train'], out['y_true_test']]).rename(y_col)
    else:
        # Backward-compatible fallback (only valid when feat_names are raw,
        # un-lagged columns already present in df — e.g. x_lags=(0,), y_lags=()).
        df_use = df.sort_values('Date').reset_index(drop=True).loc[idx_model, :]
        X_all = df_use[feat_names].copy()
        y_all = df_use[y_col].astype(float).copy()

    if use_test_only:
        X_eval = X_all.iloc[n_train:].copy()
        y_eval = y_all.iloc[n_train:].copy()
    else:
        # fallback: pakai seluruh data MODEL (lebih banyak observasi, tapi hati-hati bias)
        X_eval = X_all.copy()
        y_eval = y_all.copy()

    def _score_neg_mape(estimator, X, y_true):
        """Negative MAPE scorer (higher = better), on the back-transformed rate scale."""
        y_pred_t = estimator.predict(X)
        y_pred = _inverse_y(np.asarray(y_pred_t), y_transform, log_eps)
        return -mean_absolute_percentage_error(
            np.clip(np.asarray(y_true), mape_eps, None),
            np.clip(np.asarray(y_pred), mape_eps, None)
        )

    # scorer = make_scorer(_score_neg_mape, greater_is_better=True)

    if verbose:
        print(f"[Permutation] Eval rows={len(y_eval)}, features={len(feat_names)}, repeats={n_repeats}")

    perm = permutation_importance(
        est, X_eval, y_eval,
        scoring = _score_neg_mape,
        n_repeats = n_repeats,
        random_state = random_state
    )

    imp_df = (pd.DataFrame({
        'feature': feat_names,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std
    })
    .sort_values('importance_mean', ascending=False)
    .reset_index(drop=True))

    # normalisasi ke [0,1] (mudah untuk ranking & plot)
    if imp_df['importance_mean'].abs().max() > 0:
        imp_df['importance_norm'] = (
            imp_df['importance_mean'] - imp_df['importance_mean'].min()
        ) / (imp_df['importance_mean'].max() - imp_df['importance_mean'].min() + 1e-12)
    else:
        imp_df['importance_norm'] = 0.0

    return imp_df
