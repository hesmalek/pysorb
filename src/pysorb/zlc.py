"""Vectorised ZLC linear model solver."""

import numpy as np
from scipy import optimize

__all__ = ["linmodel"]


def linmodel(
    Lsat,
    DR2,
    gamma,
    tsat,
    t_end,
    NN=3000,
    dt=0.5,
    eps=1e-9,
    min_denominator=1e-14,
    bracket_samples=32,
    sigsum_tol=1e-3,
    max_N=None,
    return_N=False,
):
    """
    Calculate ZLC desorption curves using the linear analytical eigenvalue model.

    Parameters
    ----------
    Lsat : float
        ZLC equilibrium/flow parameter used in the eigenvalue equation.
    DR2 : float
        Diffusion-rate parameter. Must be positive.
    gamma : float
        External resistance / accumulation correction parameter.
    tsat : float
        Saturation/loading time before desorption [s].
    t_end : float
        Final desorption time [s].
    NN : int or "auto", optional
        Number of eigenmodes. Default is 3000.
        Set to "auto" to stop once abs(1 - Sigsum) is below sigsum_tol.
    dt : float, optional
        Time step for returned model arrays [s]. Default is 0.5 s.
    eps : float, optional
        Small offset used to avoid singularities at n*pi.
    min_denominator : float, optional
        Lower guard for near-zero normalisation denominators.
    bracket_samples : int, optional
        Number of interior points to scan if the default eigenvalue bracket
        does not show a sign change.
    sigsum_tol : float, optional
        Convergence tolerance for automatic mode count selection. The default
        stops when Sigsum is within 0.1% of 1.
    max_N : int, optional
        Maximum number of modes to try when NN="auto". If omitted, the cap
        is at least 3000 and increases with Lsat so difficult cases have
        enough room to converge.
    return_N : bool, optional
        If True, append the number of modes used to the returned list.

    Returns
    -------
    list
        [t_model, C_model, C_model_PL, q_model, q_model_PL, Sigsum]
    """

    Lsat = float(Lsat)
    DR2 = float(DR2)
    gamma = float(gamma)
    tsat = float(tsat)
    t_end = float(t_end)
    dt = float(dt)
    auto_N = isinstance(NN, str) and NN.lower() == "auto"

    if Lsat <= 0:
        raise ValueError("Lsat must be positive.")
    if DR2 <= 0:
        raise ValueError("DR2 must be positive.")
    if tsat < 0:
        raise ValueError("tsat must be non-negative.")
    if t_end <= 0:
        raise ValueError("t_end must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if min_denominator <= 0:
        raise ValueError("min_denominator must be positive.")
    if bracket_samples < 2:
        raise ValueError("bracket_samples must be at least 2.")
    if sigsum_tol <= 0:
        raise ValueError("sigsum_tol must be positive.")
    if max_N is not None and max_N < 1:
        raise ValueError("max_N must be at least 1.")

    auto_max_N = max_N
    if auto_N and auto_max_N is None:
        auto_max_N = max(3000, int(np.ceil(80.0 * np.sqrt(Lsat))))
    NN = int(auto_max_N if auto_N else NN)
    if NN < 1:
        raise ValueError("NN must be at least 1.")

    pi = np.pi
    t_model = np.arange(0.0, t_end + 0.5 * dt, dt)

    def guarded_denominator(value, name):
        if not np.isfinite(value):
            raise RuntimeError(f"{name} is non-finite.")
        sign = np.sign(value) if value != 0 else 1.0
        return sign * max(abs(value), min_denominator)

    def bisect_obj(beta):
        return beta / np.tan(beta) + Lsat - 1.0 - gamma * beta**2

    def find_eigenvalue(n):
        a = (n - 1) * pi + eps
        b = n * pi - eps

        fa = bisect_obj(a)
        fb = bisect_obj(b)

        if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
            scan = np.linspace(a, b, bracket_samples)
            fscan = np.array([bisect_obj(x) for x in scan])
            finite = np.isfinite(fscan)
            sign_change = np.where(
                finite[:-1] & finite[1:] & (fscan[:-1] * fscan[1:] <= 0)
            )[0]

            if len(sign_change) == 0:
                raise RuntimeError(
                    f"Could not bracket eigenvalue for mode {n}: "
                    f"f(a)={fa:.3e}, f(b)={fb:.3e}. "
                    "Try changing NN, eps, bracket_samples, Lsat, or gamma."
                )

            i = sign_change[0]
            a = scan[i]
            b = scan[i + 1]

        return optimize.brentq(
            bisect_obj,
            a,
            b,
            xtol=1e-14,
            rtol=1e-14,
            maxiter=100,
        )

    def coefficient_ai(beta):
        beta2 = beta**2
        return 2.0 * Lsat / (
            beta2
            + (Lsat - 1.0)
            + (1.0 - Lsat + gamma * beta2) ** 2
            + gamma * beta2
        )

    beta_list = []
    sigsum_running = 0.0

    for n in range(1, NN + 1):
        beta = find_eigenvalue(n)
        beta_list.append(beta)
        sigsum_running += coefficient_ai(beta)

        if auto_N and abs(1.0 - sigsum_running) <= sigsum_tol:
            break

    beta_values = np.array(beta_list, dtype=float)
    beta2 = beta_values**2

    Ai = coefficient_ai(beta_values)
    Bi = beta2 * DR2
    BPL = Bi * tsat

    exp_minus_BPL = np.exp(-BPL)
    preload_den = guarded_denominator(
        1.0 - np.sum(Ai * exp_minus_BPL),
        "preload denominator",
    )

    AiPL = Ai * (1.0 - exp_minus_BPL) / preload_den

    E = np.exp(-np.outer(t_model, Bi))

    C_model = E @ Ai
    C_model_PL = E @ AiPL

    q_weight = Lsat * DR2 * Ai / Bi - gamma * Ai
    q_den = guarded_denominator(
        Lsat * DR2 * np.sum(Ai / Bi) - gamma,
        "q denominator",
    )
    q_model = (E @ q_weight) / q_den

    q_weight_PL = Lsat * DR2 * AiPL / Bi - gamma * AiPL
    q_den_PL = guarded_denominator(
        Lsat * DR2 * np.sum(AiPL / Bi) - gamma,
        "preload q denominator",
    )
    q_model_PL = (E @ q_weight_PL) / q_den_PL

    Sigsum = np.sum(Ai)

    results = [t_model, C_model, C_model_PL, q_model, q_model_PL, Sigsum]
    if return_N:
        results.append(len(beta_values))

    return results
