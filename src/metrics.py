"""
metrics.py
----------
Evaluation metrics and forecast-comparison significance testing for the
ELinear-vs-benchmarks potato price forecasting study.

References
----------
Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.
Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
    prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291.
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.
"""

import numpy as np
from scipy import stats


def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps=1e-6):
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true, y_pred, eps=1e-6):
    """Symmetric MAPE (%) -- more stable than MAPE when prices approach zero."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def directional_accuracy(y_true, y_pred, y_prev):
    """
    Percentage of times the predicted price movement direction (up/down/flat
    relative to the last known observed price) matches the actual direction.
    Useful proxy for trading-relevant skill, mirrors the PTS logic in Guo et al. (2025).
    """
    y_true, y_pred, y_prev = (np.asarray(a, dtype=float) for a in (y_true, y_pred, y_prev))
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return float(np.mean(true_dir == pred_dir) * 100)


def practical_application_efficiency(y_pred, low, high):
    """
    PAE metric from Guo et al. (2025) 'ELinear': fraction of predictions that
    fall within the actual day's [low, high] trading range. Only meaningful
    if you have daily high/low alongside modal/close price.
    """
    y_pred, low, high = (np.asarray(a, dtype=float) for a in (y_pred, low, high))
    return float(np.mean((y_pred > low) & (y_pred < high)) * 100)


def diebold_mariano(errors_1, errors_2, h=1, power=2):
    """
    Diebold-Mariano test comparing forecast accuracy of two models.

    Parameters
    ----------
    errors_1, errors_2 : array-like
        Forecast errors (y_true - y_pred) for model 1 and model 2, same length,
        aligned in time.
    h : int
        Forecast horizon (used for HAC-style variance correction; number of
        MA lags in the loss-differential series is h-1).
    power : int
        1 -> loss differential based on |error| (MAE-style)
        2 -> loss differential based on error^2 (MSE-style)

    Returns
    -------
    dm_stat : float
        HLN small-sample corrected DM statistic.
    p_value : float
        Two-sided p-value from a Student-t(T-1) reference distribution
        (Harvey, Leybourne & Newbold 1997 recommend t over normal for finite T).
    """
    e1 = np.asarray(errors_1, dtype=float)
    e2 = np.asarray(errors_2, dtype=float)
    assert len(e1) == len(e2), "Error series must be aligned and equal length"

    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    d_bar = np.mean(d)

    # Newey-West style long-run variance with (h-1) MA lags
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        if T - lag <= 1:
            break
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * (1 - lag / h) * cov
    var_d = max(var_d, 1e-12) / T

    dm_stat = d_bar / np.sqrt(var_d)

    # Harvey, Leybourne & Newbold (1997) small-sample correction
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_corrected = dm_stat * hln

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=T - 1))
    return float(dm_stat_corrected), float(p_value)


def holm_bonferroni(p_values, alpha=0.05):
    """
    Holm-Bonferroni step-down correction for multiple pairwise DM tests
    (e.g., ELinear vs. every benchmark, per horizon).

    Parameters
    ----------
    p_values : list[float]
        Raw p-values from individual DM tests.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    reject : list[bool]
        Whether each hypothesis is rejected after correction, in the
        ORIGINAL order of p_values.
    adjusted_p : list[float]
        Holm-adjusted p-values, in the ORIGINAL order of p_values.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)
    reject_sorted = np.empty(m, dtype=bool)
    running_max = 0.0
    for i in range(m):
        adj = (m - i) * sorted_p[i]
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(running_max, 1.0)
        reject_sorted[i] = adjusted_sorted[i] < alpha

    adjusted_p = np.empty(m)
    reject = np.empty(m, dtype=bool)
    adjusted_p[order] = adjusted_sorted
    reject[order] = reject_sorted
    return reject.tolist(), adjusted_p.tolist()


def summarize_metrics(y_true, y_pred, y_prev=None, low=None, high=None):
    """Convenience wrapper returning a dict of all point-forecast metrics."""
    out = {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }
    if y_prev is not None:
        out["DirAcc"] = directional_accuracy(y_true, y_pred, y_prev)
    if low is not None and high is not None:
        out["PAE"] = practical_application_efficiency(y_pred, low, high)
    return out
