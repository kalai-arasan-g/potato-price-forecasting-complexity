"""
cv_utils.py
-----------
Rolling-origin (walk-forward) cross-validation splitting for time series.

Rationale
---------
Standard k-fold CV shuffles observations and leaks future information into
training folds, which is invalid for autocorrelated/time-ordered data.
Rolling-origin evaluation trains only on the past and validates on a
strictly future block, repeated across multiple origins.

References
----------
Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: an
    analysis and review. International Journal of Forecasting, 16(4), 437-450.
Bergmeir, C. & Benitez, J.M. (2012). On the use of cross-validation for time
    series predictor evaluation. Information Sciences, 191, 192-213.
Hyndman, R.J. & Athanasopoulos, G. Forecasting: Principles and Practice,
    3rd ed., Chapter 5.10 "Time series cross-validation".
"""

import numpy as np


def rolling_origin_splits(n_samples, seq_len, pred_len, n_folds=5, val_size=None, min_train_size=None):
    """
    Generate rolling-origin train/validation index splits for hyperparameter
    tuning. The FINAL test block (most recent data) should be reserved
    separately and NEVER passed into this function -- see
    `chronological_holdout` below.

    Parameters
    ----------
    n_samples : int
        Number of usable samples in the tuning pool (after holding out the
        final test block).
    seq_len : int
        Lookback window length (lag) required by the models.
    pred_len : int
        Forecast horizon.
    n_folds : int
        Number of rolling validation folds.
    val_size : int, optional
        Size of each validation block. Defaults to n_samples // (n_folds + 4),
        kept reasonably small so training pools stay large across folds.
    min_train_size : int, optional
        Minimum samples required before the first fold's validation block.
        Defaults to max(4 * seq_len, n_samples // 3).

    Returns
    -------
    list of (train_idx, val_idx) : list[tuple[np.ndarray, np.ndarray]]
        Index arrays into the tuning-pool array, ordered from earliest to
        latest fold (expanding-window training set each time).
    """
    if val_size is None:
        val_size = max(pred_len * 4, n_samples // (n_folds + 4))
    if min_train_size is None:
        min_train_size = max(4 * seq_len, n_samples // 3)

    splits = []
    total_needed = min_train_size + n_folds * val_size
    if total_needed > n_samples:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {n_folds} folds with "
            f"min_train_size={min_train_size} and val_size={val_size}. "
            f"Reduce n_folds/val_size or supply more data."
        )

    for k in range(n_folds):
        train_end = min_train_size + k * val_size
        val_start = train_end
        val_end = val_start + val_size
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, min(val_end, n_samples))
        splits.append((train_idx, val_idx))
    return splits


def chronological_holdout(n_samples, test_frac=0.15):
    """
    Split indices into a tuning pool (earlier data) and a final untouched
    test block (most recent data), preserving chronological order.

    The tuning pool is what `rolling_origin_splits` operates on. The test
    block is only touched once, at final evaluation, to avoid tuning-on-test
    leakage before Diebold-Mariano significance testing.
    """
    n_test = int(np.floor(n_samples * test_frac))
    n_pool = n_samples - n_test
    pool_idx = np.arange(0, n_pool)
    test_idx = np.arange(n_pool, n_samples)
    return pool_idx, test_idx


def make_windows(series, seq_len, pred_len, indices=None):
    """
    Convert a 1D array into sliding (X, y) windows for supervised forecasting.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
    seq_len : int
    pred_len : int
    indices : np.ndarray, optional
        If given, restricts to windows whose *input* starts within these
        indices (used to slice train/val/test windows without leakage).

    Returns
    -------
    X : np.ndarray, shape (n_windows, seq_len)
    y : np.ndarray, shape (n_windows, pred_len)
    start_positions : np.ndarray
        Index (into `series`) of the first element of each input window,
        useful for traceability back to dates.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    max_start = n - seq_len - pred_len + 1
    all_starts = np.arange(0, max_start)
    if indices is not None:
        idx_set = set(indices.tolist())
        all_starts = np.array([s for s in all_starts if s in idx_set])

    X = np.stack([series[s: s + seq_len] for s in all_starts])
    y = np.stack([series[s + seq_len: s + seq_len + pred_len] for s in all_starts])
    return X, y, all_starts


def zscore_stats(series):
    """
    Compute mean and standard deviation for z-score normalization from a
    series (intended to be TRAINING data only -- never touch validation or
    test data when computing these, to avoid leakage).

    Returns (mean, std), with std floored at a small epsilon to avoid
    division by zero on a degenerate (constant) series.
    """
    series = np.asarray(series, dtype=float)
    return float(np.mean(series)), float(np.std(series) + 1e-8)


def zscore_transform(series, mean, std):
    return (np.asarray(series, dtype=float) - mean) / std


def zscore_inverse(series, mean, std):
    return np.asarray(series, dtype=float) * std + mean


def evaluate_walk_forward(model, predict_fn, series_pool, seq_len, pred_len, train_end, val_end, norm_stats=None):
    """
    Given an already-fitted `model`, walk forward across [train_end, val_end)
    generating one pred_len-length forecast per step, using only data
    strictly before that step (single fit, no leakage). This is THE shared
    evaluation mechanism used identically for every model in the study --
    ELinear and every NeuralForecast-based benchmark alike -- which is what
    makes the model comparison fair and reviewer-safe. It is called both
    during CV tuning (Notebook 02) and during final testing (Notebook 04)
    with the exact same windowing logic, so validation-time and test-time
    scores are computed the same way for every model.

    Parameters
    ----------
    model : object
        An already-fitted model (returned by a model's fit_fn). Fit on
        NORMALIZED data if `norm_stats` is given (the model itself is
        unaware of normalization -- this function handles the transform and
        inverse-transform transparently around it).
    predict_fn : callable
        predict_fn(model, context_series, seq_len, pred_len) -> np.ndarray, shape (pred_len,)
    series_pool : np.ndarray
        RAW (unnormalized) series -- `y_true` is always returned on this
        original scale so metrics stay in real units (e.g., Rs/quintal).
    seq_len, pred_len : int
    train_end, val_end : int
        Index range (into series_pool) defining where the evaluation window
        starts/ends. `series_pool[:train_end]` is what the model was fit on.
    norm_stats : tuple(mean, std), optional
        If given, the context passed to `predict_fn` is z-score normalized
        using these statistics (computed once from training data -- see
        `zscore_stats`), and the model's prediction is inverse-transformed
        back to the original scale before being returned. If None, no
        normalization is applied (backward-compatible default).

    Returns
    -------
    preds, trues : np.ndarray, shape (n_windows, pred_len), or (None, None)
        if the window range is empty. Both are on the ORIGINAL data scale.
    """
    preds, trues = [], []
    for t in range(train_end, val_end - pred_len + 1):
        context = series_pool[:t]
        if norm_stats is not None:
            mean, std = norm_stats
            context_n = zscore_transform(context, mean, std)
            y_pred_n = predict_fn(model, context_n, seq_len, pred_len)
            y_pred = zscore_inverse(y_pred_n, mean, std)
        else:
            y_pred = predict_fn(model, context, seq_len, pred_len)
        y_true = series_pool[t: t + pred_len]
        preds.append(y_pred)
        trues.append(y_true)
    if not preds:
        return None, None
    return np.stack(preds), np.stack(trues)


def cv_score_generic(series_pool, seq_len, pred_len, fit_fn, predict_fn, params, seed, n_folds=5, metric_fn=None, normalize=True):
    """
    Rolling-origin CV score for ANY model conforming to the shared
    fit_fn/predict_fn interface -- used identically for ELinear and every
    NeuralForecast-based benchmark. This is the single evaluation function
    every Optuna objective in Notebook 02 calls, guaranteeing every model
    receives the exact same CV treatment.

    Parameters
    ----------
    fit_fn : callable
        fit_fn(train_series, seq_len, pred_len, params, seed) -> model
    predict_fn : callable
        predict_fn(model, context_series, seq_len, pred_len) -> np.ndarray, shape (pred_len,)
    metric_fn : callable, optional
        metric_fn(y_true, y_pred) -> float. Defaults to MSE if not supplied
        (import locally to avoid a circular import with metrics.py).
    normalize : bool
        If True (default), z-score normalize the input series using
        statistics computed from EACH FOLD'S training data only, before
        calling `fit_fn`, and inverse-transform predictions back to the
        original scale before scoring. Applied identically to every model
        passed through this function -- this is what guarantees every
        model sees the same input treatment, regardless of what each
        model/library's own internal defaults would otherwise do. Model
        architecture itself is untouched: ELinear still does its own
        instance normalization on top, NeuralForecast models still do
        whatever their own layers do -- this only controls the SCALE of
        the raw input series before it reaches any model.

    Returns
    -------
    float : mean metric across folds (lower is better).
    """
    if metric_fn is None:
        from metrics import mse as metric_fn

    splits = rolling_origin_splits(len(series_pool), seq_len, pred_len, n_folds=n_folds)
    fold_scores = []
    for train_idx, val_idx in splits:
        train_end = train_idx[-1] + 1
        val_end = min(val_idx[-1] + 1, len(series_pool))
        train_series = series_pool[:train_end]
        if train_end <= seq_len:
            continue

        if normalize:
            mean, std = zscore_stats(train_series)
            train_series_n = zscore_transform(train_series, mean, std)
            model = fit_fn(train_series_n, seq_len, pred_len, params, seed)
            preds, trues = evaluate_walk_forward(
                model, predict_fn, series_pool, seq_len, pred_len, train_end, val_end, norm_stats=(mean, std)
            )
        else:
            model = fit_fn(train_series, seq_len, pred_len, params, seed)
            preds, trues = evaluate_walk_forward(model, predict_fn, series_pool, seq_len, pred_len, train_end, val_end)

        if preds is None:
            continue
        fold_scores.append(metric_fn(trues.flatten(), preds.flatten()))

    return float(np.mean(fold_scores)) if fold_scores else float("inf")
