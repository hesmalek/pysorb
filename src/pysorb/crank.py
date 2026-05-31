"""Crank spherical diffusion uptake solver."""

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ["crank_uptake", "error", "fit_D_R2"]


def crank_uptake(t, D_R2, n_terms=100):
    """Return fractional uptake q/q_inf for spherical Fickian diffusion."""

    t = np.asarray(t, float)

    uptake = sum(
        np.exp(-(n**2) * np.pi**2 * D_R2 * t) / n**2
        for n in range(1, n_terms + 1)
    )

    q = 1 - (6 / np.pi**2) * uptake

    return np.where(t == 0, 0, q)


def error(y_exp, y_pred, mode="MAE"):
    """Calculate MAE or RMSE between experimental and predicted values."""

    y_exp = np.asarray(y_exp, float)
    y_pred = np.asarray(y_pred, float)
    residual = y_exp - y_pred
    mode = mode.upper()

    if mode == "RMSE":
        return np.sqrt(np.mean(residual**2))

    if mode == "MAE":
        return np.mean(np.abs(residual))

    raise ValueError("mode must be 'MAE' or 'RMSE'")


def fit_D_R2(t, q, mode="MAE", n_terms=100, log_bounds=(-7, 3)):
    """Fit D/R^2 by minimizing the selected error metric on a log10 scale."""

    def objective(log_D):
        D_R2 = 10**log_D
        q_pred = crank_uptake(t, D_R2, n_terms)
        return error(q, q_pred, mode)

    result = minimize_scalar(
        objective,
        bounds=log_bounds,
        method="bounded",
    )

    if not result.success:
        raise RuntimeError(result.message)

    return 10**result.x, result.fun
