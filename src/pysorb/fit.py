import ast
from dataclasses import dataclass
import operator as _op
import re

import numpy as np
from scipy.optimize import brentq, least_squares, minimize

# Universal gas constant
R = 8.314462618  # J/mol/K
DEFAULT_TREF = 298.15
USE_MULTI_TREF = False

# Numerical safety
BIG_Q = 1e20
SMALL_POS = 1e-300
SMALL_DEN = 1e-12
CONSTRAINT_TOL = 1e-9


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _positive_from_log10(log10_x):
    return 10.0 ** np.clip(log10_x, -300, 300)


def _safe_exp(x):
    return np.exp(np.clip(x, -700, 700))


def _safe_pow_pos(base, exponent):
    base = np.maximum(base, SMALL_POS)
    exponent = np.clip(exponent, -100.0, 100.0)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        out = np.power(base, exponent)
    return np.where(np.isfinite(out), out, BIG_Q)


def _safe_div(numer, denom, bad_mask=None):
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)

    if bad_mask is None:
        bad_mask = np.zeros_like(numer, dtype=bool)

    safe = np.abs(denom) > SMALL_DEN
    out = np.full_like(numer, BIG_Q, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out[safe] = numer[safe] / denom[safe]

    bad = (~np.isfinite(out)) | bad_mask
    out[bad] = BIG_Q
    return out


def _sanitize_q(q):
    q = np.asarray(q, dtype=float)
    bad = (~np.isfinite(q)) | (q < 0)
    q = np.where(bad, BIG_Q, q)
    return q


def sanitize_q(q):
    """Return finite, non-negative model predictions.

    Parameters
    ----------
    q : array-like
        Raw loading predictions from a model function.

    Returns
    -------
    numpy.ndarray
        Predictions converted to floats, with non-finite or negative values
        replaced by a large sentinel. This mirrors the sanitizing used during
        fitting and is useful for plotting model functions directly.
    """
    return _sanitize_q(q)


def _dh_j(dH):
    return np.asarray(dH, dtype=float)


def set_multi_t_reference_temperature(Tref):
    global DEFAULT_TREF
    Tref = float(Tref)
    if not np.isfinite(Tref) or Tref <= 0.0:
        raise ValueError("Tref must be a finite temperature greater than 0 K.")
    DEFAULT_TREF = Tref


def get_multi_t_reference_temperature():
    return float(DEFAULT_TREF)


def set_use_multi_t_reference_temperature(use_tref):
    global USE_MULTI_TREF
    USE_MULTI_TREF = bool(use_tref)


def get_use_multi_t_reference_temperature():
    return bool(USE_MULTI_TREF)


def _temperature_reference_factor(T):
    T = np.asarray(T, dtype=float)
    if USE_MULTI_TREF:
        return (1.0 / T) - (1.0 / DEFAULT_TREF)
    return 1.0 / T


def _b_from_log(log10_b0, dH, T):
    b0 = _positive_from_log10(log10_b0)
    return b0 * _safe_exp((-_dh_j(dH) / R) * _temperature_reference_factor(T))


# -----------------------------------------------------------------------
# Multi-temperature models
# -----------------------------------------------------------------------

def model_linear(PT, log10_K0, dH):
    P, T = PT
    K0 = _positive_from_log10(log10_K0)
    K = K0 * _safe_exp((-_dh_j(dH) / R) * _temperature_reference_factor(T))
    q = K * np.maximum(P, 0.0)
    return _sanitize_q(q)


def model_langmuir(PT, qs, log10_b0, dH):
    P, T = PT
    Pp = np.maximum(P, 0.0)
    b = _b_from_log(log10_b0, dH, T)
    bP = b * Pp
    q = qs * _safe_div(bP, 1.0 + bP)
    return _sanitize_q(q)


def model_dslangmuir(PT, qs1, log10_b01, dH1, qs2, log10_b02, dH2):
    P, T = PT
    Pp = np.maximum(P, 0.0)

    b1 = _b_from_log(log10_b01, dH1, T)
    b2 = _b_from_log(log10_b02, dH2, T)

    q1 = qs1 * _safe_div(b1 * Pp, 1.0 + b1 * Pp)
    q2 = qs2 * _safe_div(b2 * Pp, 1.0 + b2 * Pp)
    return _sanitize_q(q1 + q2)


def model_tslangmuir(PT, qs1, log10_b01, dH1, qs2, log10_b02, dH2, qs3, log10_b03, dH3):
    P, T = PT
    Pp = np.maximum(P, 0.0)

    b1 = _b_from_log(log10_b01, dH1, T)
    b2 = _b_from_log(log10_b02, dH2, T)
    b3 = _b_from_log(log10_b03, dH3, T)

    q1 = qs1 * _safe_div(b1 * Pp, 1.0 + b1 * Pp)
    q2 = qs2 * _safe_div(b2 * Pp, 1.0 + b2 * Pp)
    q3 = qs3 * _safe_div(b3 * Pp, 1.0 + b3 * Pp)
    return _sanitize_q(q1 + q2 + q3)


def model_toth(PT, qs, log10_b0, dH, t):
    P, T = PT
    Pp = np.maximum(P, 0.0)
    b = _b_from_log(log10_b0, dH, T)
    bp = np.maximum(b * Pp, SMALL_POS)
    bp_t = _safe_pow_pos(bp, t)
    denom = _safe_pow_pos(1.0 + bp_t, 1.0 / max(t, 1e-8))
    q = qs * _safe_div(bp, denom)
    return _sanitize_q(q)


def model_sips(PT, qs, log10_b0, dH, f):
    P, T = PT
    Pp = np.maximum(P, 0.0)
    b = _b_from_log(log10_b0, dH, T)
    bp_f = _safe_pow_pos(np.maximum(b * Pp, SMALL_POS), f)
    q = qs * _safe_div(bp_f, 1.0 + bp_f)
    return _sanitize_q(q)


def model_dssips(PT, qs1, log10_b01, dH1, f1, qs2, log10_b02, dH2, f2):
    P, T = PT
    Pp = np.maximum(P, 0.0)

    b1 = _b_from_log(log10_b01, dH1, T)
    b2 = _b_from_log(log10_b02, dH2, T)

    bp1 = _safe_pow_pos(np.maximum(b1 * Pp, SMALL_POS), f1)
    bp2 = _safe_pow_pos(np.maximum(b2 * Pp, SMALL_POS), f2)

    q1 = qs1 * _safe_div(bp1, 1.0 + bp1)
    q2 = qs2 * _safe_div(bp2, 1.0 + bp2)
    return _sanitize_q(q1 + q2)


def model_freundlich(PT, log10_K0, dH, n):
    P, T = PT
    Pp = np.maximum(P, 0.0)
    K = _b_from_log(log10_K0, dH, T)
    expo = 1.0 / max(n, 1e-8)
    q = K * _safe_pow_pos(np.maximum(Pp, SMALL_POS), expo)
    # Preserve the physical zero-pressure limit after clamping P for safe exponentiation.
    q[Pp == 0.0] = 0.0
    return _sanitize_q(q)


def model_bet(PT, qs, log10_bs0, dHs, log10_bl0, dHl):
    P, T = PT
    Pp = np.maximum(P, 0.0)

    bs = _b_from_log(log10_bs0, dHs, T)
    bl = _b_from_log(log10_bl0, dHl, T)

    term1 = 1.0 - bl * Pp
    term2 = 1.0 + (bs - bl) * Pp
    denom = term1 * term2

    bad = (term1 <= SMALL_DEN) | (term2 <= SMALL_DEN)
    numer = qs * bs * Pp
    q = _safe_div(numer, denom, bad_mask=bad)
    return _sanitize_q(q)


def model_dsbet(PT, qs1, log10_bs01, dHs1, log10_bl01, dHl1,
                qs2, log10_bs02, dHs2, log10_bl02, dHl2):
    P, T = PT
    Pp = np.maximum(P, 0.0)

    bs1 = _b_from_log(log10_bs01, dHs1, T)
    bl1 = _b_from_log(log10_bl01, dHl1, T)
    bs2 = _b_from_log(log10_bs02, dHs2, T)
    bl2 = _b_from_log(log10_bl02, dHl2, T)

    term11 = 1.0 - bl1 * Pp
    term12 = 1.0 + (bs1 - bl1) * Pp
    denom1 = term11 * term12
    bad1 = (term11 <= SMALL_DEN) | (term12 <= SMALL_DEN)
    q1 = _safe_div(qs1 * bs1 * Pp, denom1, bad_mask=bad1)

    term21 = 1.0 - bl2 * Pp
    term22 = 1.0 + (bs2 - bl2) * Pp
    denom2 = term21 * term22
    bad2 = (term21 <= SMALL_DEN) | (term22 <= SMALL_DEN)
    q2 = _safe_div(qs2 * bs2 * Pp, denom2, bad_mask=bad2)

    return _sanitize_q(q1 + q2)


def model_quadratic(PT, qsat, log10_b0, dHb, log10_c0, dHc):
    P, T = PT
    Pp = np.maximum(P, 0.0)

    b = _b_from_log(log10_b0, dHb, T)
    c = _b_from_log(log10_c0, dHc, T)

    numer = qsat * (b * Pp + 2.0 * c * Pp**2)
    denom = 1.0 + b * Pp + c * Pp**2
    bad = denom <= SMALL_DEN
    q = _safe_div(numer, denom, bad_mask=bad)
    return _sanitize_q(q)


def model_redlich_peterson(PT, log10_a0, dHa, log10_b0, dHb, v):
    P, T = PT
    Pp = np.maximum(P, 0.0)

    a = _b_from_log(log10_a0, dHa, T)
    b = _b_from_log(log10_b0, dHb, T)

    Pv = _safe_pow_pos(np.maximum(Pp, SMALL_POS), v)
    Pv[Pp == 0.0] = 0.0
    denom = 1.0 + b * Pv
    q = _safe_div(a * Pp, denom)
    return _sanitize_q(q)


# -----------------------------------------------------------------------
# Single-temperature models
# -----------------------------------------------------------------------

def model_linear_single(PT, log10_K):
    P, _T = PT
    K = _positive_from_log10(log10_K)
    q = K * np.maximum(P, 0.0)
    return _sanitize_q(q)


def model_langmuir_single(PT, qs, log10_b):
    P, _T = PT
    Pp = np.maximum(P, 0.0)
    b = _positive_from_log10(log10_b)
    bP = b * Pp
    q = qs * _safe_div(bP, 1.0 + bP)
    return _sanitize_q(q)


def model_dslangmuir_single(PT, qs1, log10_b1, qs2, log10_b2):
    P, _T = PT
    Pp = np.maximum(P, 0.0)

    b1 = _positive_from_log10(log10_b1)
    b2 = _positive_from_log10(log10_b2)

    q1 = qs1 * _safe_div(b1 * Pp, 1.0 + b1 * Pp)
    q2 = qs2 * _safe_div(b2 * Pp, 1.0 + b2 * Pp)
    return _sanitize_q(q1 + q2)


def model_tslangmuir_single(PT, qs1, log10_b1, qs2, log10_b2, qs3, log10_b3):
    P, _T = PT
    Pp = np.maximum(P, 0.0)

    b1 = _positive_from_log10(log10_b1)
    b2 = _positive_from_log10(log10_b2)
    b3 = _positive_from_log10(log10_b3)

    q1 = qs1 * _safe_div(b1 * Pp, 1.0 + b1 * Pp)
    q2 = qs2 * _safe_div(b2 * Pp, 1.0 + b2 * Pp)
    q3 = qs3 * _safe_div(b3 * Pp, 1.0 + b3 * Pp)
    return _sanitize_q(q1 + q2 + q3)


def model_toth_single(PT, qs, log10_b, t):
    P, _T = PT
    Pp = np.maximum(P, 0.0)
    b = _positive_from_log10(log10_b)
    bp = np.maximum(b * Pp, SMALL_POS)
    bp_t = _safe_pow_pos(bp, t)
    denom = _safe_pow_pos(1.0 + bp_t, 1.0 / max(t, 1e-8))
    q = qs * _safe_div(bp, denom)
    return _sanitize_q(q)


def model_sips_single(PT, qs, log10_b, f):
    P, _T = PT
    Pp = np.maximum(P, 0.0)
    b = _positive_from_log10(log10_b)
    bp_f = _safe_pow_pos(np.maximum(b * Pp, SMALL_POS), f)
    q = qs * _safe_div(bp_f, 1.0 + bp_f)
    return _sanitize_q(q)


def model_dssips_single(PT, qs1, log10_b1, f1, qs2, log10_b2, f2):
    P, _T = PT
    Pp = np.maximum(P, 0.0)

    b1 = _positive_from_log10(log10_b1)
    b2 = _positive_from_log10(log10_b2)

    bp1 = _safe_pow_pos(np.maximum(b1 * Pp, SMALL_POS), f1)
    bp2 = _safe_pow_pos(np.maximum(b2 * Pp, SMALL_POS), f2)

    q1 = qs1 * _safe_div(bp1, 1.0 + bp1)
    q2 = qs2 * _safe_div(bp2, 1.0 + bp2)
    return _sanitize_q(q1 + q2)


def model_freundlich_single(PT, log10_K, n):
    P, _T = PT
    Pp = np.maximum(P, 0.0)
    K = _positive_from_log10(log10_K)
    expo = 1.0 / max(n, 1e-8)
    q = K * _safe_pow_pos(np.maximum(Pp, SMALL_POS), expo)
    # Preserve the physical zero-pressure limit after clamping P for safe exponentiation.
    q[Pp == 0.0] = 0.0
    return _sanitize_q(q)


def model_bet_single(PT, qs, log10_bs, log10_bl):
    P, _T = PT
    Pp = np.maximum(P, 0.0)

    bs = _positive_from_log10(log10_bs)
    bl = _positive_from_log10(log10_bl)

    term1 = 1.0 - bl * Pp
    term2 = 1.0 + (bs - bl) * Pp
    denom = term1 * term2

    bad = (term1 <= SMALL_DEN) | (term2 <= SMALL_DEN)
    numer = qs * bs * Pp
    q = _safe_div(numer, denom, bad_mask=bad)
    return _sanitize_q(q)


def model_dsbet_single(PT, qs1, log10_bs1, log10_bl1, qs2, log10_bs2, log10_bl2):
    P, _T = PT
    Pp = np.maximum(P, 0.0)

    bs1 = _positive_from_log10(log10_bs1)
    bl1 = _positive_from_log10(log10_bl1)
    bs2 = _positive_from_log10(log10_bs2)
    bl2 = _positive_from_log10(log10_bl2)

    term11 = 1.0 - bl1 * Pp
    term12 = 1.0 + (bs1 - bl1) * Pp
    denom1 = term11 * term12
    bad1 = (term11 <= SMALL_DEN) | (term12 <= SMALL_DEN)
    q1 = _safe_div(qs1 * bs1 * Pp, denom1, bad_mask=bad1)

    term21 = 1.0 - bl2 * Pp
    term22 = 1.0 + (bs2 - bl2) * Pp
    denom2 = term21 * term22
    bad2 = (term21 <= SMALL_DEN) | (term22 <= SMALL_DEN)
    q2 = _safe_div(qs2 * bs2 * Pp, denom2, bad_mask=bad2)

    return _sanitize_q(q1 + q2)


def model_quadratic_single(PT, qsat, log10_b, log10_c):
    P, _T = PT
    Pp = np.maximum(P, 0.0)

    b = _positive_from_log10(log10_b)
    c = _positive_from_log10(log10_c)

    numer = qsat * (b * Pp + 2.0 * c * Pp**2)
    denom = 1.0 + b * Pp + c * Pp**2
    bad = denom <= SMALL_DEN
    q = _safe_div(numer, denom, bad_mask=bad)
    return _sanitize_q(q)


def model_redlich_peterson_single(PT, log10_a, log10_b, v):
    P, _T = PT
    Pp = np.maximum(P, 0.0)

    a = _positive_from_log10(log10_a)
    b = _positive_from_log10(log10_b)

    Pv = _safe_pow_pos(np.maximum(Pp, SMALL_POS), v)
    Pv[Pp == 0.0] = 0.0
    denom = 1.0 + b * Pv
    q = _safe_div(a * Pp, denom)
    return _sanitize_q(q)


# -----------------------------------------------------------------------
# Model metadata
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one isotherm model in one temperature mode.

    The object is iterable in the historical tuple order for compatibility with
    existing callers:
    function, label, equations, parameter_names, parameter_units, lower_bounds,
    upper_bounds.
    """

    function: object
    label: str
    equations: list
    parameter_names: list
    parameter_units: list
    lower_bounds: list
    upper_bounds: list

    def __iter__(self):
        yield self.function
        yield self.label
        yield self.equations
        yield self.parameter_names
        yield self.parameter_units
        yield self.lower_bounds
        yield self.upper_bounds


MULTI_T_MODELS = {
    "Linear": (
        model_linear,
        "Linear",
        ["q = K * P", "K = K0 * exp((-dH / R) * (1/T - 1/T_ref))", "fit variables: log10(K0), dH"],
        ["log10(K0)", "dH"],
        ["", "J/mol"],
        [-20.0, -100000.0],
        [5.0, 0.0],
    ),
    "Langmuir": (
        model_langmuir,
        "Langmuir",
        ["q = qs * b*P / (1 + b*P)", "b = b0 * exp((-dH / R) * (1/T - 1/T_ref))", "fit variables: log10(b0), dH"],
        ["qs", "log10(b0)", "dH"],
        ["", "", "J/mol"],
        [0.1, -20.0, -100000.0],
        [10.0, 5.0, 0.0],
    ),
    "Dual-Site Langmuir": (
        model_dslangmuir,
        "Dual-Site Langmuir",
        ["q = qs1*b1*P/(1+b1*P) + qs2*b2*P/(1+b2*P)",
         "b1 = b01*exp((-dH1 / R)*(1/T - 1/T_ref)), b2 = b02*exp((-dH2 / R)*(1/T - 1/T_ref))",
         "fit variables: log10(b01), dH1, log10(b02), dH2"],
        ["qs1", "log10(b01)", "dH1", "qs2", "log10(b02)", "dH2"],
        ["", "", "J/mol", "", "", "J/mol"],
        [0.1, -20.0, -100000.0, 0.1, -20.0, -100000.0],
        [10.0, 5.0, 0.0, 10.0, 5.0, 0.0],
    ),
    "Triple-Site Langmuir": (
        model_tslangmuir,
        "Triple-Site Langmuir",
        ["q = qs1*b1*P/(1+b1*P) + qs2*b2*P/(1+b2*P) + qs3*b3*P/(1+b3*P)",
         "bi = bi0*exp((-dHi / R)*(1/T - 1/T_ref))",
         "fit variables: log10(b01), dH1, log10(b02), dH2, log10(b03), dH3"],
        ["qs1", "log10(b01)", "dH1", "qs2", "log10(b02)", "dH2", "qs3", "log10(b03)", "dH3"],
        ["", "", "J/mol", "", "", "J/mol", "", "", "J/mol"],
        [0.1, -20.0, -100000.0, 0.1, -20.0, -100000.0, 0.1, -20.0, -100000.0],
        [10.0, 5.0, 0.0, 10.0, 5.0, 0.0, 10.0, 5.0, 0.0],
    ),
    "Toth": (
        model_toth,
        "Toth",
        ["q = qs * (b*P) / (1 + (b*P)^t)^(1/t)", "b = b0 * exp((-dH / R) * (1/T - 1/T_ref))", "fit variables: log10(b0), dH, t"],
        ["qs", "log10(b0)", "dH", "t"],
        ["", "", "J/mol", ""],
        [0.1, -20.0, -100000.0, 0.1],
        [10.0, 5.0, 0.0, 5.0],
    ),
    "Sips": (
        model_sips,
        "Sips",
        ["q = qs * (b*P)^f / (1 + (b*P)^f)", "b = b0 * exp((-dH / R) * (1/T - 1/T_ref))", "fit variables: log10(b0), dH, f"],
        ["qs", "log10(b0)", "dH", "f"],
        ["", "", "J/mol", ""],
        [0.1, -20.0, -100000.0, 0.1],
        [10.0, 5.0, 0.0, 5.0],
    ),
    "Dual-Site Sips": (
        model_dssips,
        "Dual-Site Sips",
        ["q = qs1*(b1*P)^f1/(1+(b1*P)^f1) + qs2*(b2*P)^f2/(1+(b2*P)^f2)",
         "bi = bi0*exp((-dHi / R)*(1/T - 1/T_ref))",
         "fit variables: log10(b01), dH1, f1, log10(b02), dH2, f2"],
        ["qs1", "log10(b01)", "dH1", "f1", "qs2", "log10(b02)", "dH2", "f2"],
        ["", "", "J/mol", "", "", "", "J/mol", ""],
        [0.1, -20.0, -100000.0, 0.1, 0.1, -20.0, -100000.0, 0.1],
        [10.0, 5.0, 0.0, 5.0, 10.0, 5.0, 0.0, 5.0],
    ),
    "Freundlich": (
        model_freundlich,
        "Freundlich",
        ["q = K * P^(1/n)", "K = K0 * exp((-dH / R) * (1/T - 1/T_ref))", "fit variables: log10(K0), dH, n"],
        ["log10(K0)", "dH", "n"],
        ["", "J/mol", ""],
        [-20.0, -100000.0, 0.1],
        [5.0, 0.0, 10.0],
    ),
    "BET": (
        model_bet,
        "BET",
        ["q = qs * (bs*P) / ((1 - bl*P)(1 + (bs-bl)*P))",
         "bs = bs0*exp((-dHs / R)*(1/T - 1/T_ref)), bl = bl0*exp((-dHl / R)*(1/T - 1/T_ref))",
         "fit variables: log10(bs0), dHs, log10(bl0), dHl"],
        ["qs", "log10(bs0)", "dHs", "log10(bl0)", "dHl"],
        ["", "", "J/mol", "", "J/mol"],
        [0.1, -20.0, -100000.0, -20.0, -100000.0],
        [20.0, 5.0, 0.0, 5.0, 0.0],
    ),
    "Dual-Site BET": (
        model_dsbet,
        "Dual-Site BET",
        ["q = q1 + q2",
         "qi = qsi * (bsi*P) / ((1 - bli*P)(1 + (bsi-bli)*P))",
         "bsi = bsi0*exp((-dHsi / R)*(1/T - 1/T_ref)), bli = bli0*exp((-dHli / R)*(1/T - 1/T_ref))"],
        ["qs1", "log10(bs01)", "dHs1", "log10(bl01)", "dHl1", "qs2", "log10(bs02)", "dHs2", "log10(bl02)", "dHl2"],
        ["", "", "J/mol", "", "J/mol", "", "", "J/mol", "", "J/mol"],
        [0.1, -20.0, -100000.0, -20.0, -100000.0, 0.1, -20.0, -100000.0, -20.0, -100000.0],
        [20.0, 5.0, 0.0, 5.0, 0.0, 20.0, 5.0, 0.0, 5.0, 0.0],
    ),
    "Quadratic": (
        model_quadratic,
        "Quadratic",
        ["q = qsat * (b*P + 2c*P^2) / (1 + b*P + c*P^2)",
         "b = b0*exp((-dHb / R)*(1/T - 1/T_ref)), c = c0*exp((-dHc / R)*(1/T - 1/T_ref))",
         "fit variables: log10(b0), dHb, log10(c0), dHc"],
        ["qsat", "log10(b0)", "dHb", "log10(c0)", "dHc"],
        ["", "", "J/mol", "", "J/mol"],
        [0.1, -20.0, -100000.0, -20.0, -100000.0],
        [20.0, 5.0, 0.0, 5.0, 0.0],
    ),
    "Redlich-Peterson": (
        model_redlich_peterson,
        "Redlich-Peterson",
        ["q = (a*P) / (1 + b*P^v)",
         "a = a0*exp((-dHa / R)*(1/T - 1/T_ref)), b = b0*exp((-dHb / R)*(1/T - 1/T_ref))",
         "fit variables: log10(a0), dHa, log10(b0), dHb, v"],
        ["log10(a0)", "dHa", "log10(b0)", "dHb", "v"],
        ["", "J/mol", "", "J/mol", ""],
        [-20.0, -100000.0, -20.0, -100000.0, 0.1],
        [10.0, 0.0, 10.0, 0.0, 1.0],
    ),
}

SINGLE_T_MODELS = {
    "Linear": (
        model_linear_single,
        "Linear",
        ["single-T mode", "q = K * P", "fit variable: log10(K)"],
        ["log10(K)"],
        [""],
        [-20.0],
        [5.0],
    ),
    "Langmuir": (
        model_langmuir_single,
        "Langmuir",
        ["single-T mode", "q = qs * b*P / (1 + b*P)", "fit variables: qs, log10(b)"],
        ["qs", "log10(b)"],
        ["", ""],
        [0.1, -20.0],
        [10.0, 5.0],
    ),
    "Dual-Site Langmuir": (
        model_dslangmuir_single,
        "Dual-Site Langmuir",
        ["single-T mode", "q = qs1*b1*P/(1+b1*P) + qs2*b2*P/(1+b2*P)", "fit variables: qs1, log10(b1), qs2, log10(b2)"],
        ["qs1", "log10(b1)", "qs2", "log10(b2)"],
        ["", "", "", ""],
        [0.1, -20.0, 0.1, -20.0],
        [10.0, 5.0, 10.0, 5.0],
    ),
    "Triple-Site Langmuir": (
        model_tslangmuir_single,
        "Triple-Site Langmuir",
        ["single-T mode", "q = qs1*b1*P/(1+b1*P) + qs2*b2*P/(1+b2*P) + qs3*b3*P/(1+b3*P)",
         "fit variables: qs1, log10(b1), qs2, log10(b2), qs3, log10(b3)"],
        ["qs1", "log10(b1)", "qs2", "log10(b2)", "qs3", "log10(b3)"],
        ["", "", "", "", "", ""],
        [0.1, -20.0, 0.1, -20.0, 0.1, -20.0],
        [10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
    ),
    "Toth": (
        model_toth_single,
        "Toth",
        ["single-T mode", "q = qs * (b*P) / (1 + (b*P)^t)^(1/t)", "fit variables: qs, log10(b), t"],
        ["qs", "log10(b)", "t"],
        ["", "", ""],
        [0.1, -20.0, 0.1],
        [10.0, 5.0, 5.0],
    ),
    "Sips": (
        model_sips_single,
        "Sips",
        ["single-T mode", "q = qs * (b*P)^f / (1 + (b*P)^f)", "fit variables: qs, log10(b), f"],
        ["qs", "log10(b)", "f"],
        ["", "", ""],
        [0.1, -20.0, 0.1],
        [10.0, 5.0, 5.0],
    ),
    "Dual-Site Sips": (
        model_dssips_single,
        "Dual-Site Sips",
        ["single-T mode", "q = qs1*(b1*P)^f1/(1+(b1*P)^f1) + qs2*(b2*P)^f2/(1+(b2*P)^f2)",
         "fit variables: qs1, log10(b1), f1, qs2, log10(b2), f2"],
        ["qs1", "log10(b1)", "f1", "qs2", "log10(b2)", "f2"],
        ["", "", "", "", "", ""],
        [0.1, -20.0, 0.1, 0.1, -20.0, 0.1],
        [10.0, 5.0, 5.0, 10.0, 5.0, 5.0],
    ),
    "Freundlich": (
        model_freundlich_single,
        "Freundlich",
        ["single-T mode", "q = K * P^(1/n)", "fit variables: log10(K), n"],
        ["log10(K)", "n"],
        ["", ""],
        [-20.0, 0.1],
        [5.0, 10.0],
    ),
    "BET": (
        model_bet_single,
        "BET",
        ["single-T mode", "q = qs * (bs*P) / ((1 - bl*P)(1 + (bs-bl)*P))", "fit variables: qs, log10(bs), log10(bl)"],
        ["qs", "log10(bs)", "log10(bl)"],
        ["", "", ""],
        [0.1, -20.0, -20.0],
        [20.0, 5.0, 5.0],
    ),
    "Dual-Site BET": (
        model_dsbet_single,
        "Dual-Site BET",
        ["single-T mode", "q = q1 + q2", "qi = qsi * (bsi*P) / ((1 - bli*P)(1 + (bsi-bli)*P))"],
        ["qs1", "log10(bs1)", "log10(bl1)", "qs2", "log10(bs2)", "log10(bl2)"],
        ["", "", "", "", "", ""],
        [0.1, -20.0, -20.0, 0.1, -20.0, -20.0],
        [20.0, 5.0, 5.0, 20.0, 5.0, 5.0],
    ),
    "Quadratic": (
        model_quadratic_single,
        "Quadratic",
        ["single-T mode", "q = qsat * (b*P + 2c*P^2) / (1 + b*P + c*P^2)", "fit variables: qsat, log10(b), log10(c)"],
        ["qsat", "log10(b)", "log10(c)"],
        ["", "", ""],
        [0.1, -20.0, -20.0],
        [20.0, 5.0, 5.0],
    ),
    "Redlich-Peterson": (
        model_redlich_peterson_single,
        "Redlich-Peterson",
        ["single-T mode", "q = (a*P) / (1 + b*P^v)", "fit variables: log10(a), log10(b), v"],
        ["log10(a)", "log10(b)", "v"],
        ["", "", ""],
        [-20.0, -20.0, 0.1],
        [10.0, 10.0, 1.0],
    ),
}

MULTI_T_MODELS = {name: ModelSpec(*spec) for name, spec in MULTI_T_MODELS.items()}
SINGLE_T_MODELS = {name: ModelSpec(*spec) for name, spec in SINGLE_T_MODELS.items()}


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

METRIC_OPTIONS = ["RMSE", "MAE", "ARE"]


@dataclass
class IsothermFitResult:
    """Result returned by fit_isotherm."""

    parameters: np.ndarray
    parameter_names: list
    parameter_units: list
    model_name: str
    model_label: str
    mode: str
    q_fit: np.ndarray
    metrics: dict
    scipy_result: object
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    @property
    def parameters_by_name(self):
        return {name: float(value) for name, value in zip(self.parameter_names, self.parameters)}


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    resid = y_true - y_pred
    n = len(y_true)

    rmse = float(np.sqrt(np.mean(resid ** 2))) if n > 0 else float("nan")
    mae = float(np.mean(np.abs(resid))) if n > 0 else float("nan")

    rel_ok = np.isfinite(y_true) & (y_true > 0.0)
    are = float(np.mean(np.abs(resid[rel_ok]) / y_true[rel_ok])) if np.any(rel_ok) else float("nan")

    ss_res = float(np.sum(resid ** 2)) if n > 0 else float("nan")
    y_mean = float(np.mean(y_true)) if n > 0 else float("nan")
    ss_tot = float(np.sum((y_true - y_mean) ** 2)) if n > 0 else float("nan")
    if n > 0 and np.isfinite(ss_tot) and ss_tot > 0.0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0 if n > 0 and np.isfinite(ss_res) and ss_res == 0.0 else float("nan")

    return {"R2": r2, "RMSE": rmse, "MAE": mae, "ARE": are}


def _metric_objective_value(q_obs, resid, metric):
    resid = np.asarray(resid, dtype=float)
    if not np.all(np.isfinite(resid)):
        return BIG_Q
    if metric == "RMSE":
        return float(np.sqrt(np.mean(resid ** 2))) if len(resid) else BIG_Q
    if metric == "MAE":
        return float(np.mean(np.abs(resid))) if len(resid) else BIG_Q
    if metric == "ARE":
        q_obs = np.asarray(q_obs, dtype=float)
        rel_ok = np.isfinite(q_obs) & (q_obs > 0.0)
        if not np.any(rel_ok):
            return BIG_Q
        return float(np.mean(np.abs(resid[rel_ok]) / q_obs[rel_ok]))
    raise ValueError("metric must be one of: " + ", ".join(METRIC_OPTIONS))


# -----------------------------------------------------------------------
# Model-agnostic isosteric heat
# -----------------------------------------------------------------------

def _as_positive_pressure(p):
    p = float(p)
    if not np.isfinite(p):
        return SMALL_POS
    return max(p, 1e-12)


def solve_pressure_for_loading(q_target, temps, func, popt, p_min=0.0, p_max=1.0, max_expand=20):
    """Invert any fitted q(P,T) model numerically at fixed loading."""
    q_target = float(q_target)
    temps = np.asarray(temps, dtype=float)
    if not np.isfinite(q_target) or q_target <= 0:
        raise ValueError("Loading must be positive for isosteric heat.")

    p_start = _as_positive_pressure(p_min)
    p_cap = _as_positive_pressure(p_max)
    if p_cap <= p_start:
        p_cap = max(1.0, p_start * 10.0)

    solved = []
    for temp in temps:
        def residual(pressure):
            q_val = func((np.asarray([pressure], dtype=float), np.asarray([temp], dtype=float)), *popt)[0]
            return float(q_val) - q_target

        lo = p_start
        hi = p_cap
        f_lo = residual(lo)
        f_hi = residual(hi)

        expands = 0
        while np.isfinite(f_hi) and f_lo * f_hi > 0 and f_hi < 0 and expands < max_expand:
            hi *= 10.0
            f_hi = residual(hi)
            expands += 1

        if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
            solved.append(np.nan)
            continue

        try:
            solved.append(brentq(residual, lo, hi, maxiter=100))
        except ValueError:
            solved.append(np.nan)

    return np.asarray(solved, dtype=float)


def isosteric_heat_for_loadings(loadings, temps, func, popt, p_min=0.0, p_max=1.0):
    """Return q_st in J/mol from ln(P) vs 1/T at fixed loading."""
    temps = np.asarray(temps, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    inv_t = 1.0 / temps

    if len(np.unique(np.round(temps, 10))) < 2:
        raise ValueError("Isosteric heat requires fitted data at two or more temperatures.")

    qst = np.full_like(loadings, np.nan, dtype=float)
    p_solved = np.full((len(loadings), len(temps)), np.nan, dtype=float)

    for i, loading in enumerate(loadings):
        if not np.isfinite(loading) or loading <= 0:
            continue
        pressures = solve_pressure_for_loading(loading, temps, func, popt, p_min=p_min, p_max=p_max)
        p_solved[i, :] = pressures
        ok = np.isfinite(pressures) & (pressures > 0)
        if np.count_nonzero(ok) < 2:
            continue
        slope, _intercept = np.polyfit(inv_t[ok], np.log(pressures[ok]), 1)
        qst[i] = -R * slope

    return qst, p_solved


# -----------------------------------------------------------------------
# Constraint helpers
# -----------------------------------------------------------------------

_COMPARISON_PATTERN = re.compile(r"(<=|>=|==|<|>)")

_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
}

_UNOPS = {
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}

_ALLOWED_FUNCS = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "min": min,
    "max": max,
    "pow": pow,
}

_ALLOWED_NAMES = {"pi": np.pi, "e": np.e}


def _ast_eval(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        key = node.id
        if key in env:
            return env[key]
        if key in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[key]
        raise ValueError(f"Unknown variable in constraint: '{key}'")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINOPS:
            raise ValueError(f"Operator {op_type.__name__} not allowed in constraints.")
        return _BINOPS[op_type](_ast_eval(node.left, env), _ast_eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNOPS:
            raise ValueError(f"Unary operator {op_type.__name__} not allowed in constraints.")
        return _UNOPS[op_type](_ast_eval(node.operand, env))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed in constraints.")
        fname = node.func.id
        if fname not in _ALLOWED_FUNCS:
            raise ValueError(f"Function '{fname}' not allowed in constraints.")
        args = [_ast_eval(a, env) for a in node.args]
        return _ALLOWED_FUNCS[fname](*args)
    raise ValueError(f"Unsupported expression type in constraint: {type(node).__name__}")


def _safe_eval_expr(expr, env):
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in constraint expression '{expr}': {exc}") from exc
    return _ast_eval(tree.body, env)


def _build_constraint_env(param_names, p):
    env = {}
    for name, val in zip(param_names, p):
        if name.startswith("log10(") and name.endswith(")"):
            base = name[6:-1]
            env[base] = 10.0 ** float(val)
            env[f"log10_{base}"] = float(val)
        else:
            env[name] = float(val)
    return env


def _parse_constraint_text(text):
    items = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if part:
                items.append(part)
    return items


def validate_constraints(param_names, constraints):
    """Validate constraint text against model parameter names.

    Parameters
    ----------
    param_names : sequence of str
        Fitted parameter names for the selected model, such as ``["qs",
        "log10(b)"]``.
    constraints : str
        Semicolon- or newline-separated constraint expressions. Lines may
        contain ``#`` comments.

    Returns
    -------
    list of str
        Parsed constraint expressions with blank lines and comments removed.

    Raises
    ------
    ValueError
        If an expression has invalid syntax, references an unknown variable, or
        uses an unsupported operation.

    Notes
    -----
    For parameters stored in log space, both the fitted coordinate
    ``log10_b`` and the direct variable ``b`` are available in constraints.
    """
    parsed = _parse_constraint_text(constraints)
    test_env = _build_constraint_env(param_names, np.ones(len(param_names), dtype=float))
    for expr in parsed:
        _constraint_violation_value(expr, test_env)
    return parsed


def _split_constraint_expr(expr):
    comparisons = list(_COMPARISON_PATTERN.finditer(expr))
    if not comparisons:
        raise ValueError(f"Constraint must contain one comparison operator: {expr}")
    if len(comparisons) > 1:
        raise ValueError("Chained comparisons not supported. Use separate constraints")
    m = comparisons[0]
    op = m.group(1)
    lhs = expr[:m.start()].strip()
    rhs = expr[m.end():].strip()
    if not lhs or not rhs:
        raise ValueError(f"Invalid constraint: {expr}")
    return lhs, op, rhs


def _constraint_difference(expr, env):
    lhs, _op_text, rhs = _split_constraint_expr(expr)
    lhs_val = float(_safe_eval_expr(lhs, env))
    rhs_val = float(_safe_eval_expr(rhs, env))
    return lhs_val - rhs_val


def _constraint_violation_value(expr, env):
    _lhs, op, _rhs = _split_constraint_expr(expr)
    diff = _constraint_difference(expr, env)
    if op in (">", ">="):
        return max(0.0, -diff)
    if op in ("<", "<="):
        return max(0.0, diff)
    if op == "==":
        return abs(diff)
    raise ValueError(f"Unsupported operator in constraint: {expr}")


def _constraint_margin_value(expr, env):
    _lhs, op, _rhs = _split_constraint_expr(expr)
    diff = _constraint_difference(expr, env)
    if op in (">", ">="):
        return diff
    if op in ("<", "<="):
        return -diff
    if op == "==":
        return diff
    raise ValueError(f"Unsupported operator in constraint: {expr}")


def _expr_uses_fit_variables(expr, variable_names):
    tree = ast.parse(expr.strip(), mode="eval")
    used = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if node.id in variable_names:
                used.add(node.id)

    _Visitor().visit(tree)
    return used


def _constant_constraint_value(expr, variable_names):
    if _expr_uses_fit_variables(expr, variable_names):
        return None
    try:
        return float(_safe_eval_expr(expr, {}))
    except Exception:
        return None


def _constraint_variable_maps(param_names):
    maps = {}
    for idx, name in enumerate(param_names):
        maps[name] = (idx, lambda x: float(x))
        if name.startswith("log10(") and name.endswith(")"):
            base = name[6:-1]
            maps[f"log10_{base}"] = (idx, lambda x: float(x))
            maps[base] = (idx, lambda x: float(np.log10(x)) if x > 0.0 else np.nan)
    return maps


def _simple_constraint_var(expr, variable_maps):
    text = expr.strip()
    if text in variable_maps:
        return variable_maps[text]

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return None
    node = tree.body
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "log10"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
    ):
        name = f"log10_{node.args[0].id}"
        return variable_maps.get(name)
    return None


def _apply_simple_constraint_bound(expr, param_names, lo, hi):
    lhs, op, rhs = _split_constraint_expr(expr)
    variable_maps = _constraint_variable_maps(param_names)
    variable_names = set(variable_maps)

    lhs_var = _simple_constraint_var(lhs, variable_maps)
    rhs_val = _constant_constraint_value(rhs, variable_names)
    rhs_var = _simple_constraint_var(rhs, variable_maps)
    lhs_val = _constant_constraint_value(lhs, variable_names)

    if lhs_var is not None and rhs_val is not None:
        idx, to_fit_space = lhs_var
        value = to_fit_space(rhs_val)
        direction = "lower" if op in (">", ">=") else "upper" if op in ("<", "<=") else None
    elif rhs_var is not None and lhs_val is not None:
        idx, to_fit_space = rhs_var
        value = to_fit_space(lhs_val)
        direction = "upper" if op in (">", ">=") else "lower" if op in ("<", "<=") else None
    else:
        return False

    if direction is None or not np.isfinite(value):
        return False
    if direction == "lower":
        lo[idx] = max(lo[idx], value)
    else:
        hi[idx] = min(hi[idx], value)
    return True


def _apply_simple_constraint_bounds(param_names, lo, hi, constraints):
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()
    remaining = []
    for expr in constraints:
        if not _apply_simple_constraint_bound(expr, param_names, lo, hi):
            remaining.append(expr)
    if np.any(lo >= hi):
        raise ValueError("Constraints make parameter bounds empty or inconsistent.")
    return lo, hi, remaining


def build_residual_with_constraints(func, PT, q_obs, param_names, constraint_text, penalty_scale):
    constraints = validate_constraints(param_names, constraint_text)

    def residuals(p):
        q_pred = func(PT, *p)
        q_pred = _sanitize_q(q_pred)
        data_resid = np.asarray(q_pred - q_obs, dtype=float)
        data_resid[~np.isfinite(data_resid)] = BIG_Q

        if not constraints:
            return data_resid

        env = _build_constraint_env(param_names, p)
        penalty_terms = [penalty_scale * _constraint_violation_value(expr, env) for expr in constraints]
        return np.concatenate([data_resid, np.asarray(penalty_terms, dtype=float)])

    return residuals


# -----------------------------------------------------------------------
# Analytic Jacobians for selected multi-T models
# -----------------------------------------------------------------------

LN10 = np.log(10.0)


def _jac_langmuir_site(P, T, qs, log10_b0, dH):
    b0 = _positive_from_log10(log10_b0)
    ref_factor = _temperature_reference_factor(T)
    b = b0 * _safe_exp((-_dh_j(dH) / R) * ref_factor)
    bP = b * P
    denom = 1.0 + bP
    theta = bP / denom

    dq_dqs = theta
    dq_d_log10b0 = qs * theta * (1.0 - theta) * LN10
    dq_d_dH = -(1.0 / R) * qs * theta * (1.0 - theta) * ref_factor

    return dq_dqs, dq_d_log10b0, dq_d_dH


def jac_langmuir(PT, qs, log10_b0, dH):
    P, T = PT
    dq_dqs, dq_d_lb0, dq_d_dH = _jac_langmuir_site(P, T, qs, log10_b0, dH)
    return np.column_stack([dq_dqs, dq_d_lb0, dq_d_dH])


def jac_dslangmuir(PT, qs1, log10_b01, dH1, qs2, log10_b02, dH2):
    P, T = PT
    dq1_dqs1, dq1_dlb01, dq1_ddH1 = _jac_langmuir_site(P, T, qs1, log10_b01, dH1)
    dq2_dqs2, dq2_dlb02, dq2_ddH2 = _jac_langmuir_site(P, T, qs2, log10_b02, dH2)
    return np.column_stack([
        dq1_dqs1, dq1_dlb01, dq1_ddH1,
        dq2_dqs2, dq2_dlb02, dq2_ddH2,
    ])


def jac_tslangmuir(PT, qs1, log10_b01, dH1, qs2, log10_b02, dH2, qs3, log10_b03, dH3):
    P, T = PT
    dq1_dqs1, dq1_dlb01, dq1_ddH1 = _jac_langmuir_site(P, T, qs1, log10_b01, dH1)
    dq2_dqs2, dq2_dlb02, dq2_ddH2 = _jac_langmuir_site(P, T, qs2, log10_b02, dH2)
    dq3_dqs3, dq3_dlb03, dq3_ddH3 = _jac_langmuir_site(P, T, qs3, log10_b03, dH3)
    return np.column_stack([
        dq1_dqs1, dq1_dlb01, dq1_ddH1,
        dq2_dqs2, dq2_dlb02, dq2_ddH2,
        dq3_dqs3, dq3_dlb03, dq3_ddH3,
    ])


def jac_linear(PT, log10_K0, dH):
    P, T = PT
    K0 = _positive_from_log10(log10_K0)
    ref_factor = _temperature_reference_factor(T)
    K = K0 * _safe_exp((-_dh_j(dH) / R) * ref_factor)
    dq_d_lK0 = K * P * LN10
    dq_d_dH = -(1.0 / R) * K * P * ref_factor
    return np.column_stack([dq_d_lK0, dq_d_dH])


MULTI_T_MODEL_JACS = {
    "Linear": jac_linear,
    "Langmuir": jac_langmuir,
    "Dual-Site Langmuir": jac_dslangmuir,
    "Triple-Site Langmuir": jac_tslangmuir,
    "Toth": None,
    "Sips": None,
    "Dual-Site Sips": None,
    "Freundlich": None,
    "BET": None,
    "Dual-Site BET": None,
    "Quadratic": None,
    "Redlich-Peterson": None,
}

SINGLE_T_MODEL_JACS = {k: None for k in SINGLE_T_MODELS.keys()}


def _make_jac_closure(jac_func, PT, n_constraints):
    # Extra residual rows, such as legacy penalty rows, need real derivatives.
    # Let scipy finite-difference the whole residual vector in that case.
    if jac_func is None or n_constraints > 0:
        return "2-point"

    def jac_residuals(p):
        return jac_func(PT, *p)

    return jac_residuals


def fit_model_parameters(
    func,
    P,
    T,
    q,
    param_names,
    lo,
    hi,
    *,
    jac_func=None,
    initial_guess=None,
    constraints="",
    n_trials=20,
    metric="RMSE",
    random_state=None,
    max_nfev=4000,
    progress_callback=None,
    cancel_callback=None,
):
    """Fit a model function with bounds, constraints, and metric selection.

    This is the lower-level public fitting engine used by the GUI. Most scripts
    should call :func:`fit_isotherm`, which handles model lookup and default
    bounds. Use this function when you already have a model function and
    parameter metadata, or when an application needs progress/cancel callbacks.

    Parameters
    ----------
    func : callable
        Model function with signature ``func((P, T), *params)``.
    P, T, q : array-like
        Pressure, absolute temperature in Kelvin, and observed loading arrays.
    param_names : sequence of str
        Names of the fitted parameters in optimizer order.
    lo, hi : array-like
        Lower and upper bounds in optimizer coordinates.
    jac_func : callable, optional
        Analytic Jacobian for least-squares RMSE fits, with signature
        ``jac_func((P, T), *params)``.
    initial_guess : array-like, optional
        First starting point. It must satisfy the final effective bounds after
        simple constraints are converted to bounds.
    constraints : str, optional
        Constraint text. Simple one-parameter inequalities are converted to
        bounds. General inequalities and equalities are passed to SLSQP.
    n_trials : int, default 20
        Number of starting points to try.
    metric : {"RMSE", "MAE", "ARE"}, default "RMSE"
        Metric to optimize and use for selecting the best trial.
    random_state : int or numpy.random.Generator seed, optional
        Seed passed to NumPy's default random generator for reproducible starts.
    max_nfev : int, default 4000
        Maximum function evaluations for least-squares fits, or maximum SLSQP
        iterations for minimize-based fits.
    progress_callback : callable, optional
        Called as ``progress_callback(done, total)`` after each trial.
    cancel_callback : callable, optional
        Called before each trial. If it returns true, fitting stops.

    Returns
    -------
    tuple
        ``(scipy_result, parameters, metrics, effective_lower_bounds,
        effective_upper_bounds)``.

    Raises
    ------
    ValueError
        For invalid data, bounds, metric, initial guess, or constraints.
    RuntimeError
        If no trial finds a successful feasible fit.
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    q = np.asarray(q, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    pnames = list(param_names)

    if P.shape != q.shape or P.shape != T.shape:
        raise ValueError("P, q, and T must have the same shape.")
    if P.size < 3:
        raise ValueError("Need at least 3 data points.")
    if np.any(~np.isfinite(P)) or np.any(~np.isfinite(q)) or np.any(~np.isfinite(T)):
        raise ValueError("P, q, and T must contain only finite values.")
    if np.any(P < 0.0):
        raise ValueError("Pressure values must be non-negative.")
    if np.any(q < 0.0):
        raise ValueError("Loading values must be non-negative.")
    if np.any(T <= 0.0):
        raise ValueError("Temperatures must be greater than 0 K.")

    if lo.shape != hi.shape or len(lo) != len(pnames):
        raise ValueError("Bounds length must match selected model parameters: " + ", ".join(pnames))
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("Bounds must be finite.")
    if np.any(lo >= hi):
        raise ValueError("Every lower bound must be less than its upper bound.")

    parsed_constraints = validate_constraints(pnames, constraints)
    lo, hi, nonlinear_constraints = _apply_simple_constraint_bounds(pnames, lo, hi, parsed_constraints)

    if initial_guess is None:
        first_guess = 0.5 * (lo + hi)
    else:
        first_guess = np.asarray(initial_guess, dtype=float)
        if len(first_guess) != len(pnames):
            raise ValueError("initial_guess length must match selected model parameters: " + ", ".join(pnames))
        if np.any(first_guess < lo) or np.any(first_guess > hi):
            raise ValueError("initial_guess must be inside bounds and simple constraints.")

    try:
        n_trials = int(n_trials)
    except (TypeError, ValueError):
        raise ValueError("n_trials must be a positive integer.")
    if n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if metric not in METRIC_OPTIONS:
        raise ValueError("metric must be one of: " + ", ".join(METRIC_OPTIONS))
    if metric == "ARE" and not np.any(q > 0.0):
        raise ValueError("ARE requires at least one positive observed loading.")

    rng = np.random.default_rng(random_state)
    best_result = None
    best_params = None
    best_metrics = None
    best_score = np.inf
    last_error = None

    use_minimize = bool(nonlinear_constraints) or metric in {"MAE", "ARE"}
    residual_func = build_residual_with_constraints(
        func=func,
        PT=(P, T),
        q_obs=q,
        param_names=pnames,
        constraint_text="",
        penalty_scale=1.0,
    )
    jac_callable = _make_jac_closure(jac_func, (P, T), 0)

    scipy_constraints = []
    for expr in nonlinear_constraints:
        _lhs, op, _rhs = _split_constraint_expr(expr)
        if op == "==":
            scipy_constraints.append({
                "type": "eq",
                "fun": lambda p, e=expr: _constraint_margin_value(e, _build_constraint_env(pnames, p)),
            })
        else:
            scipy_constraints.append({
                "type": "ineq",
                "fun": lambda p, e=expr: _constraint_margin_value(e, _build_constraint_env(pnames, p)),
            })

    def _objective(p):
        resid = residual_func(p)
        return _metric_objective_value(q, resid, metric)

    for trial_idx in range(n_trials):
        if cancel_callback is not None and cancel_callback():
            break

        if trial_idx == 0:
            guess = first_guess
        elif trial_idx == n_trials - 1 and best_params is not None:
            guess = best_params
        else:
            guess = rng.uniform(lo, hi)

        try:
            if use_minimize:
                result = minimize(
                    _objective,
                    x0=guess,
                    method="SLSQP",
                    bounds=list(zip(lo, hi)),
                    constraints=scipy_constraints,
                    options={"maxiter": int(max_nfev), "ftol": 1e-12},
                )
            else:
                result = least_squares(
                    residual_func,
                    x0=guess,
                    bounds=(lo, hi),
                    method="trf",
                    jac=jac_callable,
                    max_nfev=max_nfev,
                )

            if not result.success:
                last_error = str(result.message)
                continue

            params = np.asarray(result.x, dtype=float)
            q_fit = _sanitize_q(func((P, T), *params))
            if not np.all(np.isfinite(q_fit)):
                continue
            if parsed_constraints:
                env = _build_constraint_env(pnames, params)
                violations = [_constraint_violation_value(expr, env) for expr in parsed_constraints]
                if any(v > CONSTRAINT_TOL for v in violations):
                    continue

            metrics = compute_metrics(q, q_fit)
            score = metrics[metric]
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_result = result
                best_params = params
                best_metrics = metrics
        except Exception as exc:
            last_error = str(exc)
        finally:
            if progress_callback is not None:
                progress_callback(trial_idx + 1, n_trials)

    if best_result is None:
        msg = "No successful fits. Adjust bounds/constraints or increase trials."
        if last_error:
            msg += f" Last error: {last_error}"
        raise RuntimeError(msg)

    return best_result, best_params, best_metrics, lo, hi


def detect_temperature_mode(T):
    """Return 'single' for one unique temperature, otherwise 'multi'.

    T must be supplied in Kelvin.
    """
    return "single" if count_unique_temperatures(T) == 1 else "multi"


def count_unique_temperatures(T):
    """Return the number of unique temperatures after tolerance-based rounding."""
    T = np.asarray(T, dtype=float)
    if T.size == 0:
        raise ValueError("Temperature array must not be empty.")
    if np.any(~np.isfinite(T)) or np.any(T <= 0.0):
        raise ValueError("Temperatures must be finite and greater than 0 K.")
    return len(np.unique(np.round(T, 10)))


def get_temperature_dependence_note(T):
    """Return a note when temperature dependence is estimated from only two temperatures."""
    if count_unique_temperatures(T) == 2:
        return "At least three temperatures are considered acceptable to estimate temperature dependence."
    return None


def get_model(model_name, mode):
    """Return the :class:`ModelSpec` for a model name and temperature mode.

    mode must be 'single' or 'multi'.
    """
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be 'single' or 'multi'.")
    models = SINGLE_T_MODELS if mode == "single" else MULTI_T_MODELS
    if model_name not in models:
        available = ", ".join(models)
        raise ValueError(f"Model '{model_name}' is not available in {mode}-temperature mode. Choose: {available}")
    return models[model_name]


def fit_isotherm(
    P,
    q,
    T,
    model_name,
    *,
    mode="auto",
    bounds=None,
    initial_guess=None,
    constraints="",
    n_trials=20,
    metric="RMSE",
    random_state=None,
    max_nfev=4000,
):
    """Fit an adsorption isotherm model and return an :class:`IsothermFitResult`.

    This is the main public API for script and notebook use. It looks up the
    selected model, applies default or user-supplied bounds, chooses the fitting
    mode, applies constraints, and reports standard error metrics.

    Parameters
    ----------
    P, q, T : array-like
        Pressure, observed loading, and absolute temperature in Kelvin. All
        arrays must have the same shape. Pressure and loading must be
        non-negative; temperature must be greater than 0 K.
    model_name : str
        Name of a model present in ``SINGLE_T_MODELS`` or ``MULTI_T_MODELS``.
    mode : {"auto", "single", "multi"}, default "auto"
        Temperature mode. ``"auto"`` uses one unique temperature for
        single-temperature fitting and multiple temperatures for multi-T fitting.
    bounds : None or tuple, optional
        ``(lower_bounds, upper_bounds)`` in fitted parameter order. If omitted,
        model defaults are used.
    initial_guess : array-like, optional
        First starting point. It must satisfy bounds and simple constraints.
    constraints : str, optional
        Semicolon- or newline-separated constraints such as ``"qs < 3.5"`` or
        ``"qs1 + qs2 < 4"``.
    n_trials : int, default 20
        Number of starting points. The first uses ``initial_guess`` or the
        effective bounds midpoint, middle trials are random starts, and the last
        retries the best result when available.
    metric : {"RMSE", "MAE", "ARE"}, default "RMSE"
        Metric to optimize and use for best-trial selection.
    random_state : int, optional
        Seed for reproducible random starts.
    max_nfev : int, default 4000
        Maximum evaluations/iterations per trial.

    Returns
    -------
    IsothermFitResult
        Fitted parameters, parameter metadata, fitted q values, metrics, bounds,
        and the raw SciPy result object.

    Notes
    -----
    Simple one-parameter constraints are converted to optimizer bounds. General
    inequality/equality constraints use SLSQP. RMSE fits without nonlinear
    constraints use SciPy ``least_squares``; MAE and ARE use SLSQP so the
    selected metric is genuinely optimized. Heat parameters in multi-temperature
    models are fitted and reported in J/mol.
    """
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)
    T = np.asarray(T, dtype=float)

    if P.shape != q.shape or P.shape != T.shape:
        raise ValueError("P, q, and T must have the same shape.")
    if P.size < 3:
        raise ValueError("Need at least 3 data points.")
    if np.any(~np.isfinite(P)) or np.any(~np.isfinite(q)) or np.any(~np.isfinite(T)):
        raise ValueError("P, q, and T must contain only finite values.")
    if np.any(P < 0.0):
        raise ValueError("Pressure values must be non-negative.")
    if np.any(q < 0.0):
        raise ValueError("Loading values must be non-negative.")
    if np.any(T <= 0.0):
        raise ValueError("Temperatures must be greater than 0 K.")

    if mode == "auto":
        mode = detect_temperature_mode(T)
    elif mode not in {"single", "multi"}:
        raise ValueError("mode must be 'auto', 'single', or 'multi'.")

    jac_dict = SINGLE_T_MODEL_JACS if mode == "single" else MULTI_T_MODEL_JACS
    func, display, _expr, pnames, punits, default_lo, default_hi = get_model(model_name, mode)
    jac_func = jac_dict.get(model_name)

    if bounds is None:
        lo = np.asarray(default_lo, dtype=float)
        hi = np.asarray(default_hi, dtype=float)
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must be None or a (lower_bounds, upper_bounds) tuple.")
        if bounds[0] is None or bounds[1] is None:
            raise ValueError("Provide both lower_bounds and upper_bounds, or leave bounds as None.")
        lo = np.asarray(bounds[0], dtype=float)
        hi = np.asarray(bounds[1], dtype=float)

    if lo.shape != hi.shape or len(lo) != len(pnames):
        raise ValueError("Bounds length must match selected model parameters: " + ", ".join(pnames))
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("Bounds must be finite.")
    if np.any(lo >= hi):
        raise ValueError("Every lower bound must be less than its upper bound.")

    best_result, best_params, best_metrics, lo, hi = fit_model_parameters(
        func=func,
        P=P,
        T=T,
        q=q,
        param_names=pnames,
        lo=lo,
        hi=hi,
        jac_func=jac_func,
        initial_guess=initial_guess,
        constraints=constraints,
        n_trials=n_trials,
        metric=metric,
        random_state=random_state,
        max_nfev=max_nfev,
    )

    q_fit = _sanitize_q(func((P, T), *best_params))
    return IsothermFitResult(
        parameters=np.asarray(best_params, dtype=float),
        parameter_names=list(pnames),
        parameter_units=list(punits),
        model_name=model_name,
        model_label=display,
        mode=mode,
        q_fit=q_fit,
        metrics=dict(best_metrics),
        scipy_result=best_result,
        lower_bounds=lo,
        upper_bounds=hi,
    )


# -----------------------------------------------------------------------

__all__ = [
    "R",
    "get_multi_t_reference_temperature", "set_multi_t_reference_temperature",
    "get_use_multi_t_reference_temperature", "set_use_multi_t_reference_temperature",
    "METRIC_OPTIONS",
    "MULTI_T_MODELS", "SINGLE_T_MODELS",
    "MULTI_T_MODEL_JACS", "SINGLE_T_MODEL_JACS",
    "IsothermFitResult", "ModelSpec",
    "detect_temperature_mode", "count_unique_temperatures", "get_temperature_dependence_note",
    "get_model", "fit_model_parameters", "fit_isotherm",
    "compute_metrics", "sanitize_q", "validate_constraints",
    "isosteric_heat_for_loadings", "solve_pressure_for_loading",
]
