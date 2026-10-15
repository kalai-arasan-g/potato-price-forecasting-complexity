"""
elinear_model.py
-----------------
PyTorch implementation of ELinear, following:

Guo, Y., Guo, H., & Gu, D. (2025). ELinear: ensemble linear model for stock
    prediction. Neural Computing and Applications, 37, 15079-15093.
    https://doi.org/10.1007/s00521-025-11280-9

Components:
  - Instance normalization (subtract-then-add-back the last lookback value)
  - Channel-independent Linear mapping Y = A(X + B)
  - Mutual learning ensemble training (Algorithm 1 in the paper)
"""

import copy
import numpy as np
import torch
import torch.nn as nn

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearLearner(nn.Module):
    """
    A single 'weak learner': instance-normalized single-layer Linear model.
    Y = A(X + B) is implemented as a standard nn.Linear applied to the
    tail-subtracted input, with the tail added back to the output
    (equivalent formulation, easier to train stably).
    """

    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [batch, seq_len]
        last = x[:, -1:].detach()          # instance-normalization anchor
        x_norm = x - last
        out = self.linear(x_norm)
        return out + last                  # re-add tail element(s)


class ELinear(nn.Module):
    """
    Ensemble of `n_learners` LinearLearner modules. Channel-independence is
    handled OUTSIDE this module: call it once per univariate channel (e.g.,
    once for the potato closing/modal price series) rather than passing a
    multivariate tensor in -- this matches the CI strategy in the paper.
    """

    def __init__(self, seq_len, pred_len, n_learners=3):
        super().__init__()
        self.learners = nn.ModuleList([LinearLearner(seq_len, pred_len) for _ in range(n_learners)])

    def forward(self, x, return_individual=False):
        preds = [m(x) for m in self.learners]
        ensemble_pred = torch.stack(preds, dim=0).mean(dim=0)
        if return_individual:
            return ensemble_pred, preds
        return ensemble_pred


def mutual_learning_fit(
    model: ELinear,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs_phase1=50,
    epochs_phase2=5,
    epochs_phase3=20,
    batch_size=32,
    lr=1e-3,
    device=None,
    seed=None,
    verbose=False,
):
    """
    Implements Algorithm 1 (Mutual Learning) from Guo et al. (2025):

      Phase 1: each learner trained independently on random batches against
               real targets (standard bagging-style diversification).
      Phase 2: each learner picks another learner at random and does a brief
               round of training against that learner's PREDICTIONS
               (noise-averaging step).
      Phase 3: all learners do a final round of training against the REAL
               targets again, to re-anchor to ground truth.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Shape (n_samples, seq_len) and (n_samples, pred_len).
    device : str or None
        "cuda" / "cpu". If None (default), auto-selects "cuda" when
        available (torch.cuda.is_available()), else falls back to "cpu".
    """
    if device is None:
        device = DEFAULT_DEVICE
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    model = model.to(device)
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = X.shape[0]

    optimizers = [torch.optim.Adam(learner.parameters(), lr=lr) for learner in model.learners]
    loss_fn = nn.MSELoss()

    def batches():
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            yield X[idx], y[idx]

    # ---- Phase 1: independent bagged training on real targets ----
    for epoch in range(epochs_phase1):
        for xb, yb in batches():
            for learner, opt in zip(model.learners, optimizers):
                opt.zero_grad()
                pred = learner(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        if verbose and (epoch + 1) % max(1, epochs_phase1 // 5) == 0:
            print(f"[Phase 1] epoch {epoch+1}/{epochs_phase1}")

    # ---- Phase 2: mutual learning against peers' predictions ----
    n_learners = len(model.learners)
    for epoch in range(epochs_phase2):
        for xb, yb in batches():
            # snapshot current predictions of all learners (detached) as targets
            with torch.no_grad():
                snapshot_preds = [learner(xb) for learner in model.learners]
            for i, (learner, opt) in enumerate(zip(model.learners, optimizers)):
                partner_idx = np.random.choice([j for j in range(n_learners) if j != i])
                target = snapshot_preds[partner_idx]
                opt.zero_grad()
                pred = learner(xb)
                loss = loss_fn(pred, target)
                loss.backward()
                opt.step()
        if verbose:
            print(f"[Phase 2] epoch {epoch+1}/{epochs_phase2}")

    # ---- Phase 3: final re-anchoring on real targets ----
    for epoch in range(epochs_phase3):
        for xb, yb in batches():
            for learner, opt in zip(model.learners, optimizers):
                opt.zero_grad()
                pred = learner(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        if verbose and (epoch + 1) % max(1, epochs_phase3 // 5) == 0:
            print(f"[Phase 3] epoch {epoch+1}/{epochs_phase3}")

    return model


@torch.no_grad()
def predict(model: ELinear, X: np.ndarray, device=None):
    if device is None:
        device = DEFAULT_DEVICE
    model = model.to(device)
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    out = model(Xt)
    return out.cpu().numpy()


def elinear_fit_fn(train_series, seq_len, pred_len, params, seed=0):
    """
    fit_fn(train_series, seq_len, pred_len, params, seed) -> model
    Conforms to the shared generic CV interface in cv_utils.cv_score_generic.

    NOTE ON NORMALIZATION: `train_series` arrives ALREADY z-score normalized
    by the calling harness (cv_utils.cv_score_generic / the Notebook 03
    final-training loop) when normalization is enabled -- this function does
    not normalize anything itself. The harness applies the exact same
    normalization to every model in the study uniformly; duplicating it here
    would double-normalize and corrupt the harness's own inverse-transform.
    ELinear's own internal instance normalization (subtract/re-add the last
    lookback value, per Guo et al. 2025) is unchanged either way -- it
    operates on a per-window basis on top of whatever series it receives.
    """
    from cv_utils import make_windows
    X_train, y_train, _ = make_windows(train_series, seq_len, pred_len)
    model = ELinear(seq_len=seq_len, pred_len=pred_len, n_learners=params["n_learners"])
    model = mutual_learning_fit(
        model, X_train, y_train,
        epochs_phase1=params["epochs_phase1"], epochs_phase2=params["epochs_phase2"],
        epochs_phase3=params["epochs_phase3"], batch_size=params["batch_size"], lr=params["lr"],
        seed=seed,
    )
    return model


def elinear_predict_fn(model, context_series, seq_len, pred_len):
    """
    predict_fn(model, context_series, seq_len, pred_len) -> np.ndarray, shape (pred_len,)
    Conforms to the shared generic CV interface. `context_series` arrives
    already normalized (if normalization is enabled) by the calling
    harness, which also inverse-transforms this function's return value --
    this function itself stays agnostic to normalization.
    """
    context = np.asarray(context_series[-seq_len:], dtype="float32").reshape(1, -1)
    return predict(model, context)[0]
