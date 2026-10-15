"""
imputation.py
--------------
Missing-value imputation methods for the daily Agra potato price series,
plus an empirical validation routine to justify the final method choice.

Methods implemented
--------------------
1. Kalman smoothing on a local-level+trend+seasonal structural time series
   model (statsmodels UnobservedComponents), analogous to imputeTS's
   na_kalman(model="StructTS") in R.
2. Linear interpolation (baseline).
3. Seasonal decomposition (STL) + linear interpolation of the deseasonalized
   residual, then re-seasonalized.

References
----------
Durbin, J. & Koopman, S.J. (2012). Time Series Analysis by State Space
    Methods, 2nd ed. Oxford University Press. (Kalman filter/smoother for
    missing data in structural models.)
Moritz, S. & Bartz-Beielstein, T. (2017). imputeTS: Time Series Missing
    Value Imputation in R. The R Journal, 9(1), 207-218.
Moritz, S., Sarda, A., Bartz-Beielstein, T., Zaefferer, M., & Stork, J.
    (2015). Comparison of different methods for univariate time series
    imputation in R. arXiv:1510.03924.
Cleveland, R.B., Cleveland, W.S., McRae, J.E., & Terpenning, I. (1990).
    STL: A seasonal-trend decomposition procedure based on loess.
    Journal of Official Statistics, 6(1), 3-73.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.seasonal import STL


def kalman_impute_structts(series, seasonal_period=7, trend=True):
    """
    Impute missing values via Kalman smoothing on a structural time series
    model: local linear trend + (optionally) a seasonal component.

    statsmodels' UnobservedComponents natively supports NaNs in `endog` --
    the Kalman filter/smoother marginalizes over missing observations and
    the smoothed state gives the best linear unbiased estimate to fill them.

    Parameters
    ----------
    series : pd.Series
        Indexed by date, may contain NaNs. A DatetimeIndex with an explicit
        (regular) frequency is recommended -- reindex to a full daily
        calendar with `asfreq('D')` BEFORE calling this function so that
        gaps are represented as NaN rows rather than missing index entries.
    seasonal_period : int
        Seasonal periodicity to model explicitly (e.g., 7 for day-of-week
        mandi patterns). Set to None to fit level+trend only.
    trend : bool
        Whether to include a stochastic local linear trend component.

    Returns
    -------
    imputed : pd.Series
        Same index as `series`, with NaNs replaced by smoothed estimates
        and observed values left untouched.
    fitted_model : UnobservedComponents results object
    """
    y = series.astype(float)
    kwargs = dict(level="local linear trend" if trend else "local level")
    if seasonal_period:
        kwargs["freq_seasonal"] = [{"period": seasonal_period, "harmonics": 3}]

    model = UnobservedComponents(y.values, **kwargs)
    res = model.fit(disp=False)
    smoothed = res.smoothed_state[0] if not seasonal_period else res.fittedvalues

    # Use smoothed/fitted values ONLY to fill gaps; keep real observations exact.
    imputed_values = y.values.copy()
    nan_mask = np.isnan(imputed_values)
    imputed_values[nan_mask] = res.fittedvalues[nan_mask]

    imputed = pd.Series(imputed_values, index=series.index, name=series.name)
    return imputed, res


def linear_interpolate(series):
    """
    Baseline: simple time-weighted linear interpolation.
    limit_direction="both" allows edge points (no neighbor on one side) to be
    filled by extrapolating from the nearest available side, rather than
    being left as NaN.
    """
    return series.interpolate(method="time", limit_direction="both")


def seasonal_decompose_interpolate(series, period=365, robust=True):
    """
    STL-decompose the series (after linearly filling for STL's own fitting
    requirement), then linearly interpolate the trend+residual and re-add
    the seasonal component at the missing positions.

    Following the imputeTS approach validated in Moritz et al. (2015):
    seasonal-split imputation is a strong alternative to Kalman for series
    with a dominant seasonal pattern.
    """
    filled_for_fit = series.interpolate(method="time", limit_direction="both")
    stl = STL(filled_for_fit, period=period, robust=robust).fit()
    deseasonalized = filled_for_fit - stl.seasonal

    # Reapply NaNs to deseasonalized series at original gap positions, then interpolate
    deseasonalized_with_gaps = deseasonalized.copy()
    deseasonalized_with_gaps[series.isna()] = np.nan
    deseasonalized_interp = deseasonalized_with_gaps.interpolate(method="time", limit_direction="both")

    imputed = deseasonalized_interp + stl.seasonal
    imputed[~series.isna()] = series[~series.isna()]  # keep true observations exact
    return imputed, stl


def impute_positive_series(series, method_fn, log_transform=True, floor=1e-3, bound_margin=0.5, **kwargs):
    """
    Wrapper that enforces positivity AND plausibility for price-series
    imputation.

    Structural Kalman smoothing and STL/linear interpolation both extrapolate
    using local trend estimates, which can legitimately produce negative or
    zero values across long gaps or near series edges -- physically
    impossible for a price series. Operating in log-space guarantees the
    back-transformed (exp) result is always strictly positive, and is the
    standard approach for economic/financial price series (Box & Cox, 1964;
    Tsay, "Analysis of Financial Time Series").

    Beyond positivity, MLE-based methods (Kalman) can occasionally diverge
    (optimizer failing to converge on a particular masked-point pattern),
    producing an extreme extrapolated value that, once exponentiated, is
    astronomically large and corrupts any downstream error metric. To guard
    against this pathology we additionally clip the log-space imputed values
    to a plausible range derived from the OBSERVED data: [min_obs - margin,
    max_obs + margin] in log-space, where margin is a multiple of the
    observed log-range. This does not affect well-behaved imputations (which
    stay well inside the observed range) -- it only prevents optimizer
    failures from producing nonsensical outliers.

    Parameters
    ----------
    series : pd.Series
        Original (possibly NaN-containing) price series.
    method_fn : callable
        One of `kalman_impute_structts`, `linear_interpolate`,
        `seasonal_decompose_interpolate`. Called on the log-transformed
        series.
    log_transform : bool
        If True (default), impute on log(series) and exponentiate back.
    floor : float
        Small positive floor applied as a final safety net in case of any
        residual numerical edge cases after exponentiation.
    bound_margin : float
        Fraction of the observed log-range used to extend the plausible
        bound beyond the min/max actually observed (default 0.5 = allow
        imputed values up to 50% of the observed log-range beyond either
        extreme -- generous enough for genuine trend/seasonal excursions,
        tight enough to catch optimizer divergence).

    Returns
    -------
    imputed : pd.Series
        Same shape/index as `series`, guaranteed strictly positive and
        bounded to a plausible range.
    aux : object or None
        Any auxiliary fitted-model object the wrapped method returns
        (e.g., the UnobservedComponents results or STL decomposition),
        passed through unchanged.
    """
    observed = series.dropna().clip(lower=floor)

    if log_transform:
        transformed = np.log(observed.reindex(series.index))
        transformed[series.isna()] = np.nan
        log_obs = np.log(observed)
        lo_bound = log_obs.min() - bound_margin * (log_obs.max() - log_obs.min())
        hi_bound = log_obs.max() + bound_margin * (log_obs.max() - log_obs.min())
    else:
        transformed = series
        lo_bound = observed.min() - bound_margin * (observed.max() - observed.min())
        hi_bound = observed.max() + bound_margin * (observed.max() - observed.min())

    out = method_fn(transformed, **kwargs)
    imputed, aux = (out if isinstance(out, tuple) else (out, None))

    # Sanity-bound BEFORE exponentiating: prevents optimizer-divergence blowups
    imputed = imputed.clip(lower=lo_bound, upper=hi_bound)

    if log_transform:
        imputed = np.exp(imputed)

    imputed = imputed.clip(lower=floor)
    return imputed, aux


def validate_imputation_methods(series, methods, mask_frac=0.1, n_trials=20, random_state=42, **kwargs):
    """
    Empirical validation: repeatedly mask a random subset of currently
    OBSERVED points, impute, and score reconstruction error against the
    known true values. This determines which method to report as primary,
    rather than relying on literature alone.

    Parameters
    ----------
    series : pd.Series
        Fully/mostly observed series (ideally already has few real NaNs, or
        run this on a clean sub-segment of the data).
    methods : dict[str, callable]
        Mapping of method name -> function(series_with_gaps) -> imputed_series
        (or (imputed_series, aux) tuple; aux is ignored here).
    mask_frac : float
        Fraction of observed points to mask per trial.
    n_trials : int
        Number of random masking trials (bootstrap-style) for stable estimates.

    Returns
    -------
    results_df : pd.DataFrame
        Rows = methods, columns = RMSE mean/std, MAE mean/std across trials.
    """
    rng = np.random.default_rng(random_state)
    observed_idx = series.dropna().index
    # Exclude the first/last week from candidate masking positions: interpolation-
    # based methods cannot fill true edge points (no neighbor on one side), so
    # including them would penalize those methods for an artifact of validation
    # design rather than a genuine accuracy difference.
    if len(observed_idx) > 21:
        observed_idx = observed_idx[7:-7]
    n_mask = int(len(observed_idx) * mask_frac)

    records = {name: {"rmse": [], "mae": []} for name in methods}

    for trial in range(n_trials):
        mask_positions = rng.choice(observed_idx, size=n_mask, replace=False)
        true_values = series.loc[mask_positions].copy()

        masked_series = series.copy()
        masked_series.loc[mask_positions] = np.nan

        for name, fn in methods.items():
            out = fn(masked_series, **kwargs.get(name, {}))
            imputed = out[0] if isinstance(out, tuple) else out
            pred_values = imputed.loc[mask_positions]

            err = (true_values.values - pred_values.values).astype(float)
            records[name]["rmse"].append(np.sqrt(np.mean(err ** 2)))
            records[name]["mae"].append(np.mean(np.abs(err)))

    rows = []
    for name, vals in records.items():
        rows.append({
            "method": name,
            "RMSE_mean": np.mean(vals["rmse"]),
            "RMSE_std": np.std(vals["rmse"]),
            "MAE_mean": np.mean(vals["mae"]),
            "MAE_std": np.std(vals["mae"]),
        })
    return pd.DataFrame(rows).sort_values("RMSE_mean").reset_index(drop=True)
