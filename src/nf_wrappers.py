"""
nf_wrappers.py
--------------
Unified NeuralForecast-based wrappers for every deep-learning benchmark
(LSTM, TCN, NBEATS, DLinear, NLinear, PatchTST, iTransformer), built to the
SAME fit_fn/predict_fn interface used for ELinear. This is the core of the
fairness/reviewer-safety guarantee for this study: every model -- proposed
(ELinear) and benchmark alike -- passes through the identical rolling-origin
CV harness, identical seed protocol, and identical externally-computed
metrics (src/metrics.py), regardless of which library implements it.

Design principle
-----------------
NeuralForecast's own `cross_validation()` and `Auto*` convenience classes are
NOT used here, even though they exist and would be less code. They each
carry slightly different internal validation-splitting assumptions
(different from our rolling_origin_splits and from ELinear's manually
implemented CV), which would mean the ELinear-vs-benchmark comparison isn't
apples-to-apples anymore. Prioritizing one shared, auditable evaluation
harness over library-native convenience is the correct choice for a
reviewer-facing comparison.

Reference for the model implementations used
----------------------------------------------
Garza, A., Challu, C., & Mergenthaler-Canseco, M. (2022). Neuralforecast:
    User friendly state-of-the-art neural forecasting models.
    https://github.com/Nixtla/neuralforecast
"""

import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, TCN, NBEATS, DLinear, NLinear, PatchTST, iTransformer
from neuralforecast.losses.pytorch import MSE as NF_MSE

_UNIQUE_ID = "agra_potato"
_BASE_DATE = "2000-01-01"  # arbitrary anchor; only relative spacing matters, freq="D" throughout


def series_to_df(series_1d):
    """Convert a raw 1D numpy array into NeuralForecast's required long format."""
    series_1d = np.asarray(series_1d, dtype=float)
    return pd.DataFrame({
        "unique_id": _UNIQUE_ID,
        "ds": pd.date_range(_BASE_DATE, periods=len(series_1d), freq="D"),
        "y": series_1d,
    })


# Common Lightning/trainer kwargs applied to every model: single accelerator
# auto-selection (GPU if available, else CPU -- consistent with the rest of
# the pipeline), silenced logging so notebook output stays readable.
_COMMON_TRAINER_KWARGS = dict(accelerator="auto", enable_progress_bar=False, logger=False)


def _common_kwargs(seq_len, pred_len, params, seed):
    return dict(
        h=pred_len,
        input_size=seq_len,
        max_steps=params["max_steps"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        random_seed=seed,
        loss=NF_MSE(),
        # Every model receives already-normalized input from the shared
        # harness (cv_utils.cv_score_generic / evaluate_walk_forward), which
        # applies the SAME z-score normalization to every model uniformly.
        # scaler_type="identity" disables each model's own internal
        # preprocessing default (NeuralForecast defaults to "robust" for
        # LSTM/TCN and "identity" for the rest -- an inconsistency that
        # meant different models were effectively seeing differently scaled
        # inputs). This does not touch any model's architecture, only which
        # library-level preprocessing step runs before its own layers.
        scaler_type="identity",
        **_COMMON_TRAINER_KWARGS,
    )


def _lstm_kwargs(seq_len, pred_len, params, seed):
    kw = _common_kwargs(seq_len, pred_len, params, seed)
    kw.update(encoder_hidden_size=params["encoder_hidden_size"], encoder_n_layers=params["encoder_n_layers"])
    return LSTM(**kw)


def _tcn_kwargs(seq_len, pred_len, params, seed):
    kw = _common_kwargs(seq_len, pred_len, params, seed)
    kw.update(kernel_size=params["kernel_size"], encoder_hidden_size=params["encoder_hidden_size"])
    return TCN(**kw)


def _nbeats_kwargs(seq_len, pred_len, params, seed):
    kw = _common_kwargs(seq_len, pred_len, params, seed)
    n_blocks = params["n_blocks"]
    mlp_units = params["mlp_units"]
    kw.update(
        n_blocks=[n_blocks] * 3,
        mlp_units=[[mlp_units, mlp_units]] * 3,
        # Generic N-BEATS (identity-only stacks), not the "interpretable"
        # trend+seasonality configuration: the latter is mathematically
        # undefined at h=1 (Oreshkin et al. 2019 offer both configurations;
        # generic is chosen here so the SAME architecture is used at every
        # horizon -- using interpretable stacks at some horizons and generic
        # at others would confound the horizon comparison).
        stack_types=["identity", "identity", "identity"],
    )
    return NBEATS(**kw)


def _dlinear_kwargs(seq_len, pred_len, params, seed):
    kw = _common_kwargs(seq_len, pred_len, params, seed)
    kw.update(moving_avg_window=params["moving_avg_window"])
    return DLinear(**kw)


def _nlinear_kwargs(seq_len, pred_len, params, seed):
    kw = _common_kwargs(seq_len, pred_len, params, seed)
    return NLinear(**kw)


def _patchtst_kwargs(seq_len, pred_len, params, seed):
    kw = _common_kwargs(seq_len, pred_len, params, seed)
    kw.update(
        patch_len=params["patch_len"], stride=params["stride"],
        hidden_size=params["hidden_size"], n_heads=params["n_heads"],
        encoder_layers=params["encoder_layers"],
    )
    return PatchTST(**kw)


def _itransformer_kwargs(seq_len, pred_len, params, seed):
    kw = _common_kwargs(seq_len, pred_len, params, seed)
    kw.update(
        n_series=1,  # univariate potato-price series
        hidden_size=params["hidden_size"], n_heads=params["n_heads"],
        e_layers=params["e_layers"], d_ff=params["d_ff"],
    )
    return iTransformer(**kw)


# Registry: model name -> (constructor function, Optuna search-space function)
# Search-space functions return a dict of params EXCLUDING seq_len (handled
# at the objective level identically to ELinear) and max_steps/learning_rate/
# batch_size (common to all, also suggested at a shared level below).
NF_MODEL_REGISTRY = {
    "LSTM": {
        "constructor": _lstm_kwargs,
        "search_space": lambda trial: {
            "encoder_hidden_size": trial.suggest_categorical("encoder_hidden_size", [32, 64, 128]),
            "encoder_n_layers": trial.suggest_int("encoder_n_layers", 1, 2),
        },
    },
    "TCN": {
        "constructor": _tcn_kwargs,
        "search_space": lambda trial: {
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
            "encoder_hidden_size": trial.suggest_categorical("encoder_hidden_size", [16, 32, 64]),
        },
    },
    "NBEATS": {
        "constructor": _nbeats_kwargs,
        "search_space": lambda trial: {
            "n_blocks": trial.suggest_int("n_blocks", 1, 3),
            "mlp_units": trial.suggest_categorical("mlp_units", [64, 128, 256]),
        },
    },
    "DLinear": {
        "constructor": _dlinear_kwargs,
        "search_space": lambda trial: {
            "moving_avg_window": trial.suggest_categorical("moving_avg_window", [7, 13, 25]),
        },
    },
    "NLinear": {
        "constructor": _nlinear_kwargs,
        "search_space": lambda trial: {},
    },
    "PatchTST": {
        "constructor": _patchtst_kwargs,
        "search_space": lambda trial: {
            "patch_len": trial.suggest_categorical("patch_len", [4, 8, 16]),
            "stride": trial.suggest_categorical("stride", [2, 4, 8]),
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
            "encoder_layers": trial.suggest_int("encoder_layers", 1, 3),
        },
    },
    "iTransformer": {
        "constructor": _itransformer_kwargs,
        "search_space": lambda trial: {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
            "e_layers": trial.suggest_int("e_layers", 1, 3),
            "d_ff": trial.suggest_categorical("d_ff", [32, 64, 128]),
        },
    },
}


def suggest_common_params(trial):
    """Hyperparameters shared across every NeuralForecast model in the registry."""
    return {
        "max_steps": trial.suggest_int("max_steps", 50, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def make_nf_fit_fn(model_name):
    """
    Returns fit_fn(train_series, seq_len, pred_len, params, seed) -> nf_object
    conforming to the shared generic CV interface (cv_utils.cv_score_generic).

    NOTE ON NORMALIZATION: `train_series` arrives ALREADY z-score normalized
    by the calling harness when normalization is enabled -- this function
    does not normalize anything itself (see cv_utils.zscore_stats/
    zscore_transform and the `normalize` parameter of cv_score_generic).
    Each model's own internal library scaler is separately disabled via
    scaler_type="identity" in _common_kwargs, so the ONLY scaling any model
    receives is the single external one applied uniformly by the harness.
    """
    spec = NF_MODEL_REGISTRY[model_name]

    def fit_fn(train_series, seq_len, pred_len, params, seed=0):
        model = spec["constructor"](seq_len, pred_len, params, seed)
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df=series_to_df(train_series))
        return nf

    return fit_fn


def nf_predict_fn(nf_object, context_series, seq_len, pred_len):
    """
    predict_fn(model, context_series, seq_len, pred_len) -> np.ndarray, shape (pred_len,)
    conforming to the shared generic CV interface. `context_series` arrives
    already normalized (if enabled) by the calling harness, which also
    inverse-transforms this function's return value -- this function itself
    stays agnostic to normalization. The model forecasts the next
    `pred_len` steps from the end of the context, using weights fixed at
    fit time (no refitting on context).
    """
    model_col = nf_object.models[0].__class__.__name__
    context_df = series_to_df(context_series)
    fc = nf_object.predict(df=context_df)
    return fc[model_col].values[:pred_len]


def nf_param_count(nf_object):
    return sum(p.numel() for p in nf_object.models[0].parameters())
