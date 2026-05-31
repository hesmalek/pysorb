# =========================
# File: pysorb_mix_bt.py
# =========================

"""
pysorb_mix_bt: A Python module for adsorption mixture modelling.

This module provides fast routines for unary isotherm calculations,
IAST mixture predictions, and extended multisite adsorption models.

Supported isotherms
-------------------
- Linear
- Langmuir (SSL)
- Sips
- Freundlich
- Toth
- BET
- Quadratic
- Redlich-Peterson

Notes
-----
`p_part` represents the per-component input value supplied to the
isotherm model, typically partial pressure, but it may also represent
another concentration-like variable if the fitted isotherm form is
consistent with that basis.
"""

import warnings
import numpy as np
from numba import njit
import math


__version__ = "0.2.0"

__all__ = ["parse", "iast", "ext", "unary", "clip_partial_pressures"]


EPS = 1e-20
SMALL_DEN = 1e-15
EXP_CLIP = 50.0
P_PART_MIN = 1e-12

IAST_MAX_ITER = 150
IAST_TOL = 1e-8
IAST_X_UNARY_FLOOR = 1e-8
IAST_P_MIN = P_PART_MIN
IAST_PI_LOG_FLOOR = 1e-300

STEP_NORM_MAX = 2.0

DAMPING_INIT = 1.0
DAMPING_GROW = 1.3
DAMPING_SHRINK = 0.5
DAMPING_MIN = 1e-8
LINESEARCH_MAX_STEPS = 16

PIVOT_TOL = 1e-14

TOTH_SERIES_TOL = 1e-11
TOTH_SERIES_MAX_TERMS = 5000

BET_P_LIMIT = 1e-15
RP_PI_SEGMENTS = 8
RP_PI_POWER = 4.0

NUMBA_CACHE = True  # set False during development to avoid stale cache issues


# ============================================================
# MODEL IDS
# ============================================================

MODEL_LINEAR     = 0
MODEL_SSL        = 1
MODEL_SIPS       = 2
MODEL_FREUNDLICH = 3
MODEL_TOTH       = 4
MODEL_BET        = 5
MODEL_QUADRATIC  = 6
MODEL_RP         = 7

MODEL_MISSING    = -1


# ============================================================
# PYTHON INPUT PARSER
# ============================================================


MODEL_PARAM_COUNTS = {
    MODEL_LINEAR: 1,
    MODEL_SSL: 2,
    MODEL_SIPS: 3,
    MODEL_FREUNDLICH: 2,
    MODEL_TOTH: 3,
    MODEL_BET: 3,
    MODEL_QUADRATIC: 3,
    MODEL_RP: 3,
}

MODEL_PARAM_LABELS = {
    MODEL_LINEAR: ("K",),
    MODEL_SSL: ("qs", "b"),
    MODEL_SIPS: ("qs", "b", "f"),
    MODEL_FREUNDLICH: ("K", "n"),
    MODEL_TOTH: ("qs", "b", "t"),
    MODEL_BET: ("qs", "bs", "bl"),
    MODEL_QUADRATIC: ("qsat", "b", "c"),
    MODEL_RP: ("a", "b", "v"),
}


def normalize_model_name(name: str) -> str:
    s = str(name).strip().lower()

    if s in ("linear", "lin"):
        return "LIN"
    if s in ("langmuir", "ssl"):
        return "SSL"
    if s in ("sips",):
        return "SIPS"
    if s in ("freundlich", "fr"):
        return "FR"
    if s in ("toth",):
        return "TOTH"
    if s in ("bet",):
        return "BET"
    if s in ("quadratic", "quad"):
        return "QUAD"
    if s in ("redlich-peterson", "redlich_peterson", "rp"):
        return "RP"

    raise ValueError(f"Unknown isotherm model name: {name}")


def model_name_to_id(name: str) -> int:
    s = normalize_model_name(name)

    if s == "LIN":
        return MODEL_LINEAR
    if s == "SSL":
        return MODEL_SSL
    if s == "SIPS":
        return MODEL_SIPS
    if s == "FR":
        return MODEL_FREUNDLICH
    if s == "TOTH":
        return MODEL_TOTH
    if s == "BET":
        return MODEL_BET
    if s == "QUAD":
        return MODEL_QUADRATIC
    if s == "RP":
        return MODEL_RP

    raise ValueError(f"Unsupported model name: {name}")


def parse_inputs(components):
    if len(components) == 0:
        raise ValueError("components must not be empty.")

    n_comp = len(components)
    max_sites = max(len(comp["isotherms"]) for comp in components)

    if max_sites == 0:
        raise ValueError("Each component must contain at least one isotherm.")

    names = []
    model_ids_in = np.full((n_comp, max_sites), MODEL_MISSING, dtype=np.int32)
    p1_in = np.zeros((n_comp, max_sites), dtype=np.float64)
    p2_in = np.zeros((n_comp, max_sites), dtype=np.float64)
    p3_in = np.zeros((n_comp, max_sites), dtype=np.float64)

    for i, comp in enumerate(components):
        if "MoleculeName" not in comp:
            raise ValueError(f"Component {i} is missing 'MoleculeName'.")
        if "isotherms" not in comp:
            raise ValueError(f"Component {comp['MoleculeName']} is missing 'isotherms'.")

        names.append(comp["MoleculeName"])

        if len(comp["isotherms"]) == 0:
            raise ValueError(f"Component {comp['MoleculeName']} must contain at least one isotherm.")

        for s, iso in enumerate(comp["isotherms"]):
            if len(iso) < 1:
                raise ValueError(f"Invalid isotherm for component {comp['MoleculeName']}: {iso}")

            mid = model_name_to_id(iso[0])
            params = iso[1:]

            expected = MODEL_PARAM_COUNTS[mid]
            labels = MODEL_PARAM_LABELS[mid]

            if len(params) < expected:
                missing = labels[len(params):]
                warnings.warn(
                    f"Component '{comp['MoleculeName']}', site {s}, model '{iso[0]}': "
                    f"expected {expected} parameter(s) {labels}, got {len(params)}. "
                    f"Missing parameter(s) {missing} will be filled with 0.0.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            elif len(params) > expected:
                extra = params[expected:]
                warnings.warn(
                    f"Component '{comp['MoleculeName']}', site {s}, model '{iso[0]}': "
                    f"expected {expected} parameter(s) {labels}, got {len(params)}. "
                    f"Extra parameter(s) {extra} will be ignored.",
                    RuntimeWarning,
                    stacklevel=3,
                )

            model_ids_in[i, s] = mid

            # keep only the expected parameters, pad missing with zeros
            used = list(params[:expected])

            if len(used) >= 1:
                p1_in[i, s] = float(used[0])
            if len(used) >= 2:
                p2_in[i, s] = float(used[1])
            if len(used) >= 3:
                p3_in[i, s] = float(used[2])

    return names, model_ids_in, p1_in, p2_in, p3_in


def _validate_parsed(parsed):
    if not isinstance(parsed, tuple) or len(parsed) != 5:
        raise ValueError("parsed must be the tuple returned by parse(...).")

    _, model_ids_in, p1_in, p2_in, p3_in = parsed

    if not isinstance(model_ids_in, np.ndarray) or model_ids_in.dtype != np.int32:
        raise ValueError(
            "parsed model_ids_in must be a numpy int32 array. "
            f"Got dtype={getattr(model_ids_in, 'dtype', type(model_ids_in))}. "
            "Use parse(...) to produce a valid parsed tuple."
        )
    if model_ids_in.ndim != 2:
        raise ValueError("parsed model_ids_in must be 2D.")

    for name, arr in (("p1_in", p1_in), ("p2_in", p2_in), ("p3_in", p3_in)):
        if not isinstance(arr, np.ndarray) or arr.dtype != np.float64:
            raise ValueError(
                f"parsed {name} must be a numpy float64 array. "
                f"Got dtype={getattr(arr, 'dtype', type(arr))}. "
                "Use parse(...) to produce a valid parsed tuple."
            )
        if arr.ndim != 2:
            raise ValueError(f"parsed {name} must be 2D.")
        if arr.shape != model_ids_in.shape:
            raise ValueError(f"parsed {name} shape does not match model_ids_in.")

def _validate_p_part(p_part, n_comp):
    if p_part.ndim != 1:
        raise ValueError("p_part must be a 1D array-like.")
    if p_part.shape[0] != n_comp:
        raise ValueError(f"p_part has length {p_part.shape[0]}, expected {n_comp}.")
    if not np.all(np.isfinite(p_part)):
        raise ValueError("p_part must contain only finite values.")


def clip_partial_pressures(p_part):
    """
    Return input partial pressures clipped to the solver's positive lower bound.
    """
    p_part = np.asarray(p_part, dtype=np.float64)
    if not np.all(np.isfinite(p_part)):
        raise ValueError("p_part must contain only finite values.")
    return np.clip(p_part, P_PART_MIN, None)


def _check_bet_saturation(p_part, parsed):
    """
    Emit a RuntimeWarning if any BET component has b_l * p >= 1 - BET_P_LIMIT.
    Called from Python before entering Numba kernels so the warning is visible.
    The clamp still occurs inside the kernel; this only makes it audible.
    """
    _, model_ids_in, _p1, _p2, p3_in = parsed
    n_comp = model_ids_in.shape[0]
    max_sites = model_ids_in.shape[1]

    for i in range(n_comp):
        for s in range(max_sites):
            if model_ids_in[i, s] != MODEL_BET:
                continue
            bl = p3_in[i, s]
            if bl <= 0.0:
                continue
            p = float(p_part[i])
            if bl * p >= 1.0 - BET_P_LIMIT:
                warnings.warn(
                    f"BET component {i}, site {s}: b_l * p = {bl * p:.6g} >= 1. "
                    f"Pressure clamped to {(1.0 - BET_P_LIMIT) / bl:.6g}. "
                    "Results near BET saturation may be unreliable.",
                    RuntimeWarning,
                    stacklevel=3,
                )


# ============================================================
# NUMBA PACK
# ============================================================

@njit(cache=NUMBA_CACHE)
def pack(p_part_in, model_ids_in, p1_in, p2_in, p3_in):
    """
    Compact per-component site data for unary and IAST only.

    Note: this compresses out MODEL_MISSING sites independently for each
    component, so packed site columns are not guaranteed to preserve the
    original physical site alignment across components. That is fine for
    unary/IAST, but ext() must use the original aligned site layout.
    """
    n_comp = p_part_in.shape[0]
    max_sites = model_ids_in.shape[1]

    n_site = 0
    for i in range(n_comp):
        n_sites_this = 0
        for s in range(max_sites):
            if model_ids_in[i, s] != MODEL_MISSING:
                n_sites_this += 1
        if n_sites_this > n_site:
            n_site = n_sites_this

    p_part = np.empty(n_comp, dtype=np.float64)
    model_ids = np.full((n_comp, n_site), MODEL_MISSING, dtype=np.int32)
    qs_all = np.zeros((n_comp, n_site), dtype=np.float64)
    bs_all = np.zeros((n_comp, n_site), dtype=np.float64)
    fs_all = np.ones((n_comp, n_site), dtype=np.float64)
    K_all  = np.zeros((n_comp, n_site), dtype=np.float64)

    for i in range(n_comp):
        p_part[i] = p_part_in[i]

    for i in range(n_comp):
        sp = 0
        for s in range(max_sites):
            mid = model_ids_in[i, s]
            if mid == MODEL_MISSING:
                continue

            model_ids[i, sp] = mid

            p1 = p1_in[i, s]
            p2 = p2_in[i, s]
            p3 = p3_in[i, s]

            if mid == MODEL_LINEAR:
                K_all[i, sp] = p1

            elif mid == MODEL_SSL:
                qs_all[i, sp] = p1
                bs_all[i, sp] = p2

            elif mid == MODEL_SIPS:
                qs_all[i, sp] = p1
                bs_all[i, sp] = p2
                fs_all[i, sp] = p3

            elif mid == MODEL_FREUNDLICH:
                K_all[i, sp]  = p1
                fs_all[i, sp] = p2

            elif mid == MODEL_TOTH:
                qs_all[i, sp] = p1
                bs_all[i, sp] = p2
                fs_all[i, sp] = p3

            elif mid == MODEL_BET:
                qs_all[i, sp] = p1
                bs_all[i, sp] = p2
                K_all[i, sp]  = p3

            elif mid == MODEL_QUADRATIC:
                qs_all[i, sp] = p1
                bs_all[i, sp] = p2
                K_all[i, sp]  = p3

            elif mid == MODEL_RP:
                qs_all[i, sp] = p1
                bs_all[i, sp] = p2
                fs_all[i, sp] = p3

            sp += 1

    return p_part, model_ids, qs_all, bs_all, fs_all, K_all, n_comp, n_site


# ============================================================
# SOFTMAX PARAM
# ============================================================

@njit(cache=NUMBA_CACHE)
def softmax_y_to_x(y, n_comp):
    n_eq = n_comp - 1
    expy = np.empty(n_eq)
    s = 1.0

    for j in range(n_eq):
        yj = y[j]
        if yj > EXP_CLIP:
            yj = EXP_CLIP
        elif yj < -EXP_CLIP:
            yj = -EXP_CLIP
        ej = math.exp(yj)
        expy[j] = ej
        s += ej

    invs = 1.0 / s
    x = np.empty(n_comp)
    for j in range(n_eq):
        x[j] = expy[j] * invs
    x[n_comp - 1] = invs
    return x


# ============================================================
# HELPERS
# ============================================================

@njit(cache=NUMBA_CACHE)
def _safe_pow_bp(p, b, v):
    if p < EPS:
        p = EPS
    if b <= 0.0:
        return 0.0

    if abs(v) < SMALL_DEN:
        return b

    lp = math.log(p)
    lb = math.log(b)
    L = lb + v * lp

    if L > EXP_CLIP:
        return math.exp(EXP_CLIP)
    if L < -EXP_CLIP:
        return 0.0

    return math.exp(L)


@njit(cache=NUMBA_CACHE)
def _inv1p(x):
    if x <= 0.0:
        return 1.0
    if x > 1e30:
        return 0.0
    return 1.0 / (1.0 + x)


# ============================================================
# SINGLE-SITE q / pi
# ============================================================

@njit(cache=NUMBA_CACHE)
def q_site(p, qs, b, f, K, model_id):
    if p < EPS:
        p = EPS

    if model_id == MODEL_MISSING:
        return 0.0

    if model_id == MODEL_LINEAR:
        qv = K * p
        return qv if qv > 0.0 else 0.0

    if model_id == MODEL_SSL:
        if qs <= 0.0 or b <= 0.0:
            return 0.0
        z = b * p
        qv = qs * z / (1.0 + z)
        return qv if qv > 0.0 else 0.0

    if model_id == MODEL_SIPS:
        if qs <= 0.0 or b <= 0.0 or f <= 0.0:
            return 0.0
        z = (b * p) ** f
        qv = qs * z / (1.0 + z)
        return qv if qv > 0.0 else 0.0

    if model_id == MODEL_FREUNDLICH:
        n = f
        if K <= 0.0 or n <= 0.0:
            return 0.0
        qv = K * (p ** (1.0 / n))
        return qv if qv > 0.0 else 0.0

    if model_id == MODEL_TOTH:
        if qs <= 0.0 or b <= 0.0 or f <= 0.0:
            return 0.0
        bp = b * p
        den = (1.0 + (bp ** f)) ** (1.0 / f)
        qv = (qs * bp) / den
        return qv if qv > 0.0 else 0.0

    if model_id == MODEL_BET:
        bs = b
        bl = K
        if qs <= 0.0 or bs <= 0.0 or bl <= 0.0:
            return 0.0

        if bl * p >= 1.0 - BET_P_LIMIT:
            p = (1.0 - BET_P_LIMIT) / bl

        one_minus = 1.0 - bl * p
        if one_minus < SMALL_DEN:
            one_minus = SMALL_DEN

        den2 = 1.0 + (bs - bl) * p
        if den2 < SMALL_DEN:
            den2 = SMALL_DEN

        qv = qs * (bs * p) / (one_minus * den2)
        return qv if qv > 0.0 else 0.0

    if model_id == MODEL_QUADRATIC:
        qsat = qs
        c = K
        if qsat <= 0.0 or b <= 0.0 or c <= 0.0:
            return 0.0
        num = b * p + 2.0 * c * p * p
        den = 1.0 + b * p + c * p * p
        if den < SMALL_DEN:
            den = SMALL_DEN
        qv = qsat * num / den
        return qv if qv > 0.0 else 0.0

    a = qs
    bb = b
    v = f
    if a <= 0.0 or bb <= 0.0:
        return 0.0
    bpv = _safe_pow_bp(p, bb, v)
    qv = a * p * _inv1p(bpv)
    return qv if qv > 0.0 else 0.0


@njit(cache=NUMBA_CACHE)
def _pi_toth_series(p, qs, b, t):
    if p < EPS:
        p = EPS
    if qs <= 0.0 or b <= 0.0 or t <= 0.0:
        return 0.0

    z = (b * p) ** t
    v = z / (1.0 + z)
    if v < EPS:
        return qs * b * p

    a = 1.0 / t
    v_a = v ** a

    acc = 0.0
    v_k = 1.0
    for k in range(TOTH_SERIES_MAX_TERMS):
        term = v_a * v_k / (a + k)
        acc += term
        if abs(term) < TOTH_SERIES_TOL:
            break
        v_k *= v

    return (qs / t) * acc


_GL16_X = np.array([
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318,
    -0.7554044083550030, -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.09501250983763744, 0.09501250983763744,
    0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
    0.7554044083550030, 0.8656312023878318, 0.9445750230732326,
    0.9894009349916499
], dtype=np.float64)

_GL16_W = np.array([
    0.027152459411754095, 0.06225352393864789, 0.09515851168249278,
    0.12462897125553387, 0.14959598881657673, 0.16915651939500254,
    0.18260341504492358, 0.1894506104550685, 0.1894506104550685,
    0.18260341504492358, 0.16915651939500254, 0.14959598881657673,
    0.12462897125553387, 0.09515851168249278, 0.06225352393864789,
    0.027152459411754095
], dtype=np.float64)


@njit(cache=NUMBA_CACHE)
def _pi_rp_gl16(p, a, b, v):
    if p < EPS:
        p = EPS
    if a <= 0.0 or b <= 0.0:
        return 0.0

    if abs(v) < SMALL_DEN:
        return a * p / (1.0 + b)

    if abs(v - 1.0) < 1e-12:
        return (a / b) * math.log1p(b * p)

    # RP spreading pressure is integral_0^p a / (1 + b*t^v) dt.
    # For large b*p^v most of the curvature sits close to zero, so integrate
    # t = p*x^RP_PI_POWER on segmented x intervals instead of directly in t.
    acc = 0.0
    seg_width = 1.0 / RP_PI_SEGMENTS

    for seg in range(RP_PI_SEGMENTS):
        lo = seg * seg_width
        hi = lo + seg_width
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)

        for i in range(16):
            x = half * _GL16_X[i] + mid
            if x < EPS:
                x = EPS

            x_pow = x ** RP_PI_POWER
            t = p * x_pow
            if t < EPS:
                t = EPS

            jac = p * RP_PI_POWER * (x ** (RP_PI_POWER - 1.0))
            bpv = _safe_pow_bp(t, b, v)
            acc += half * _GL16_W[i] * jac * _inv1p(bpv)

    return a * acc


@njit(cache=NUMBA_CACHE)
def pi_site(p, qs, b, f, K, model_id):
    if p < EPS:
        p = EPS

    if model_id == MODEL_MISSING:
        return 0.0

    if model_id == MODEL_LINEAR:
        if K <= 0.0:
            return 0.0
        return K * p

    if model_id == MODEL_SSL:
        if qs <= 0.0 or b <= 0.0:
            return 0.0
        return qs * math.log1p(b * p)

    if model_id == MODEL_SIPS:
        if qs <= 0.0 or b <= 0.0 or f <= 0.0:
            return 0.0
        z = (b * p) ** f
        return (qs / f) * math.log1p(z)

    if model_id == MODEL_FREUNDLICH:
        n = f
        if K <= 0.0 or n <= 0.0:
            return 0.0
        return K * n * (p ** (1.0 / n))

    if model_id == MODEL_TOTH:
        return _pi_toth_series(p, qs, b, f)

    if model_id == MODEL_BET:
        bs = b
        bl = K
        if qs <= 0.0 or bs <= 0.0 or bl <= 0.0:
            return 0.0

        if bl * p >= 1.0 - BET_P_LIMIT:
            p = (1.0 - BET_P_LIMIT) / bl

        num = 1.0 + (bs - bl) * p
        den = 1.0 - bl * p
        if num < SMALL_DEN:
            num = SMALL_DEN
        if den < SMALL_DEN:
            den = SMALL_DEN

        return qs * math.log(num / den)

    if model_id == MODEL_QUADRATIC:
        qsat = qs
        c = K
        if qsat <= 0.0 or b <= 0.0 or c <= 0.0:
            return 0.0
        arg = 1.0 + b * p + c * p * p
        if arg < SMALL_DEN:
            arg = SMALL_DEN
        if arg > 1e300:
            arg = 1e300
        return qsat * math.log(arg)

    return _pi_rp_gl16(p, qs, b, f)


# ============================================================
# MULTISITE
# ============================================================

@njit(cache=NUMBA_CACHE)
def q_multi(p, i_comp, n_site, model_ids, qs_all, bs_all, fs_all, K_all):
    acc = 0.0
    for s in range(n_site):
        mid = model_ids[i_comp, s]
        if mid == MODEL_MISSING:
            continue
        acc += q_site(
            p,
            qs_all[i_comp, s],
            bs_all[i_comp, s],
            fs_all[i_comp, s],
            K_all[i_comp, s],
            mid,
        )
    return acc


@njit(cache=NUMBA_CACHE)
def pi_multi(p, i_comp, n_site, model_ids, qs_all, bs_all, fs_all, K_all):
    acc = 0.0
    for s in range(n_site):
        mid = model_ids[i_comp, s]
        if mid == MODEL_MISSING:
            continue
        acc += pi_site(
            p,
            qs_all[i_comp, s],
            bs_all[i_comp, s],
            fs_all[i_comp, s],
            K_all[i_comp, s],
            mid,
        )
    return acc


# ============================================================
# RESIDUALS / JACOBIAN
# ============================================================

@njit(cache=NUMBA_CACHE)
def residuals_y(y, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site):
    x = softmax_y_to_x(y, n_comp)

    pi_vals = np.empty(n_comp)
    for i in range(n_comp):
        xi = x[i]
        p0 = p_part[i] / xi
        if p0 < IAST_P_MIN:
            p0 = IAST_P_MIN
        pi_vals[i] = pi_multi(
            p0, i, n_site, model_ids, qs_all, bs_all, fs_all, K_all
        )

    r = np.empty(n_comp - 1)
    for k in range(n_comp - 1):
        pik = max(pi_vals[k], IAST_PI_LOG_FLOOR)
        pik1 = max(pi_vals[k + 1], IAST_PI_LOG_FLOOR)
        r[k] = math.log(pik) - math.log(pik1)

    return r, x, pi_vals


@njit(cache=NUMBA_CACHE)
def residual_norm(r):
    err2 = 0.0
    for i in range(r.size):
        err2 += r[i] * r[i]
    return math.sqrt(err2)


@njit(cache=NUMBA_CACHE)
def jacobian_analytic_y(y, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site):
    """
    Analytic Jacobian of the IAST residuals with respect to the softmax
    parameters y.

    Uses the Gibbs adsorption identity dpi/dp = q(p)/p, which holds for
    every isotherm model without any per-model branching.

    Residuals are log-spreading-pressure differences:
        r_k = log(pi_k) - log(pi_{k+1})

    Chain rule:
        dr_k/dy_j =
            (1/pi_k)(dpi_k/dx_k)(dx_k/dy_j)
            - (1/pi_{k+1})(dpi_{k+1}/dx_{k+1})(dx_{k+1}/dy_j)

    where
        dpi_i/dx_i = (q(p_i/x_i) / (p_i/x_i)) * (-p_i / x_i^2)
                   = -q(p_i/x_i) / x_i

    and the softmax Jacobian is
        dx_i/dy_j = x_i * (delta_{ij} - x_j)
    """
    n_eq = n_comp - 1
    x = softmax_y_to_x(y, n_comp)

    dpi_dx = np.empty(n_comp)
    for i in range(n_comp):
        xi = x[i]
        p0 = p_part[i] / xi
        if p0 < IAST_P_MIN:
            p0 = IAST_P_MIN
        qv = q_multi(p0, i, n_site, model_ids, qs_all, bs_all, fs_all, K_all)
        piv = pi_multi(p0, i, n_site, model_ids, qs_all, bs_all, fs_all, K_all)
        if piv < IAST_PI_LOG_FLOOR:
            piv = IAST_PI_LOG_FLOOR
        dpi_dx[i] = (-qv / xi) / piv

    J = np.zeros((n_eq, n_eq))
    for k in range(n_eq):
        for j in range(n_eq):
            dxk_dyj  = x[k]   * ((1.0 if k   == j else 0.0) - x[j])
            dxk1_dyj = x[k+1] * ((1.0 if k+1 == j else 0.0) - x[j])
            J[k, j]  = dpi_dx[k] * dxk_dyj - dpi_dx[k+1] * dxk1_dyj

    return J


# ============================================================
# LINEAR SOLVER
# ============================================================

@njit(cache=NUMBA_CACHE)
def solve_linear(A, b):
    n = b.size
    M = A.copy()
    rhs = b.copy()
    x = np.empty(n)

    for k in range(n):
        piv = k
        maxabs = abs(M[k, k])

        for i in range(k + 1, n):
            v = abs(M[i, k])
            if v > maxabs:
                maxabs = v
                piv = i

        if maxabs < PIVOT_TOL:
            return x, False

        if piv != k:
            for j in range(k, n):
                tmp = M[k, j]
                M[k, j] = M[piv, j]
                M[piv, j] = tmp
            tmp = rhs[k]
            rhs[k] = rhs[piv]
            rhs[piv] = tmp

        akk = M[k, k]
        for i in range(k + 1, n):
            f = M[i, k] / akk
            M[i, k] = 0.0
            for j in range(k + 1, n):
                M[i, j] -= f * M[k, j]
            rhs[i] -= f * rhs[k]

    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        den = M[i, i]
        if abs(den) < PIVOT_TOL:
            return x, False
        x[i] = s / den

    return x, True


# ============================================================
# NEWTON SOLVER
# ============================================================

@njit(cache=NUMBA_CACHE)
def solve_iast(p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site,
               max_iter=IAST_MAX_ITER, tol=IAST_TOL):
    n_eq = n_comp - 1

    pi_init = np.empty(n_comp)
    sum_pi_init = 0.0
    for i in range(n_comp):
        p0 = p_part[i]
        if p0 < IAST_P_MIN:
            p0 = IAST_P_MIN
        pi_init[i] = pi_multi(p0, i, n_site, model_ids, qs_all, bs_all, fs_all, K_all)
        if pi_init[i] < EPS:
            pi_init[i] = EPS
        sum_pi_init += pi_init[i]

    if sum_pi_init <= EPS:
        sum_pi_init = 0.0
        for i in range(n_comp):
            p0 = p_part[i]
            if p0 < IAST_P_MIN:
                p0 = IAST_P_MIN
            pi_init[i] = q_multi(p0, i, n_site, model_ids, qs_all, bs_all, fs_all, K_all)
            if pi_init[i] < EPS:
                pi_init[i] = EPS
            sum_pi_init += pi_init[i]

    x0 = np.empty(n_comp)
    for i in range(n_comp):
        x0[i] = pi_init[i] / sum_pi_init

    x_last = x0[n_comp - 1]
    y = np.empty(n_eq)
    for i in range(n_eq):
        y[i] = math.log(x0[i] / x_last)

    best_err = 1e300
    best_y = y.copy()
    damping = DAMPING_INIT

    for _it in range(max_iter):
        r, x, _ = residuals_y(y, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site)
        err = residual_norm(r)

        if err < best_err:
            best_err = err
            best_y = y.copy()

        if err < tol:
            return x, err, True

        J = jacobian_analytic_y(y, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site)

        bvec = np.empty(n_eq)
        for i in range(n_eq):
            bvec[i] = -r[i]

        dy, converged_linear = solve_linear(J, bvec)
        if not converged_linear:
            rb, xb, _ = residuals_y(best_y, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site)
            return xb, residual_norm(rb), False

        dy2 = 0.0
        for i in range(n_eq):
            dy2 += dy[i] * dy[i]
        dy_norm = math.sqrt(dy2)

        if dy_norm > STEP_NORM_MAX:
            sc = STEP_NORM_MAX / dy_norm
            for i in range(n_eq):
                dy[i] *= sc

        step = damping
        accepted = False

        for _ls in range(LINESEARCH_MAX_STEPS):
            y_new = y.copy()
            for i in range(n_eq):
                y_new[i] = y[i] + step * dy[i]

            r_new, _, _ = residuals_y(
                y_new, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site
            )
            err_new = residual_norm(r_new)

            if err_new < err:
                y = y_new
                accepted = True
                damping = min(1.0, step * DAMPING_GROW)
                break

            step *= DAMPING_SHRINK

        if not accepted:
            damping *= DAMPING_SHRINK
            if damping < DAMPING_MIN:
                rb, xb, _ = residuals_y(best_y, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site)
                return xb, residual_norm(rb), False

    rb, xb, _ = residuals_y(best_y, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site)
    return xb, residual_norm(rb), False


# ============================================================
# LOADINGS
# ============================================================

@njit(cache=NUMBA_CACHE)
def total_loading(x, p_part, qs_all, bs_all, fs_all, K_all, model_ids, n_comp, n_site):
    denom = 0.0
    for i in range(n_comp):
        xi = x[i]
        p0 = p_part[i] / xi
        if p0 < IAST_P_MIN:
            p0 = IAST_P_MIN
        qi = q_multi(p0, i, n_site, model_ids, qs_all, bs_all, fs_all, K_all)
        if qi < EPS:
            qi = EPS
        denom += xi / qi
    return 1.0 / max(denom, EPS)


@njit(cache=NUMBA_CACHE)
def component_loadings_iast(x, qT, n_comp):
    """
    Convert adsorbed-phase mole fractions into component loadings.

    In standard IAST, once the total loading qT is known, the component
    loadings are obtained from:
        q_i = x_i * qT
    where x_i is the adsorbed-phase mole fraction.
    """
    q = np.empty(n_comp)
    for i in range(n_comp):
        q[i] = x[i] * qT
    return q


# ============================================================
# EXTENDED MODEL (ALIGNED SITES)
# ============================================================

@njit(cache=NUMBA_CACHE)
def _ext_site_family_ok(model_ids_in, n_comp, s):
    has_ssl = False
    has_sips = False
    has_toth = False

    for i in range(n_comp):
        mid = model_ids_in[i, s]
        if mid == MODEL_MISSING:
            continue
        if mid == MODEL_SSL:
            has_ssl = True
        elif mid == MODEL_SIPS:
            has_sips = True
        elif mid == MODEL_TOTH:
            has_toth = True
        else:
            return False

    if has_sips and has_toth:
        return False

    return has_ssl or has_sips or has_toth


@njit(cache=NUMBA_CACHE)
def _ext_allowed(model_ids_in, n_comp, max_sites):
    for s in range(max_sites):
        if not _ext_site_family_ok(model_ids_in, n_comp, s):
            return False
    return True


@njit(cache=NUMBA_CACHE)
def extended_multisite_loadings_aligned(p_part, model_ids_in, p1_in, p2_in, p3_in, n_comp, max_sites):
    """
    Approximate extended mixture model using the original aligned site layout.

    Site columns are preserved exactly as given in the parsed input, so
    site index meaning stays consistent across components.

    Allowed families per site across components:
    - SSL/Sips
    - SSL/Toth

    For SSL/Sips:
        z_i = b_i p_i              for SSL
        z_i = (b_i p_i)^f_i        for Sips
        q_i,s = qsat_i z_i / (1 + sum_j z_j)

    For SSL/Toth:
        tau_i = 1                  for SSL
        tau_i = t_i                for Toth
        base  = 1 + sum_j (b_j p_j)^(tau_j)
        q_i,s = qsat_i (b_i p_i) / base^(1/tau_i)

    This is still approximate and not generally thermodynamically rigorous.
    """
    q = np.zeros(n_comp, dtype=np.float64)

    if not _ext_allowed(model_ids_in, n_comp, max_sites):
        for i in range(n_comp):
            q[i] = np.nan
        return q

    for s in range(max_sites):
        has_sips = False
        has_toth = False

        for i in range(n_comp):
            mid = model_ids_in[i, s]
            if mid == MODEL_SIPS:
                has_sips = True
            elif mid == MODEL_TOTH:
                has_toth = True

        # ----------------------------------------------------
        # Family 1: SSL / Sips
        # ----------------------------------------------------
        if has_sips:
            denom = 1.0

            for i in range(n_comp):
                mid = model_ids_in[i, s]
                if mid == MODEL_MISSING:
                    continue

                p = p_part[i]
                if p < EPS:
                    p = EPS

                b = p2_in[i, s]
                if b <= 0.0:
                    continue

                if mid == MODEL_SSL:
                    denom += b * p
                else:  # MODEL_SIPS
                    f = p3_in[i, s]
                    if f <= 0.0:
                        continue
                    denom += (b * p) ** f

            if denom < SMALL_DEN:
                denom = SMALL_DEN

            for i in range(n_comp):
                mid = model_ids_in[i, s]
                if mid == MODEL_MISSING:
                    continue

                qsat = p1_in[i, s]
                b = p2_in[i, s]
                if qsat <= 0.0 or b <= 0.0:
                    continue

                p = p_part[i]
                if p < EPS:
                    p = EPS

                if mid == MODEL_SSL:
                    num = b * p
                else:  # MODEL_SIPS
                    f = p3_in[i, s]
                    if f <= 0.0:
                        continue
                    num = (b * p) ** f

                q[i] += qsat * num / denom

        # ----------------------------------------------------
        # Family 2: SSL / Toth
        # ----------------------------------------------------
        else:
            base = 1.0

            for j in range(n_comp):
                mid = model_ids_in[j, s]
                if mid == MODEL_MISSING:
                    continue

                p = p_part[j]
                if p < EPS:
                    p = EPS

                b = p2_in[j, s]
                if b <= 0.0:
                    continue

                if mid == MODEL_SSL:
                    tau_j = 1.0
                else:  # MODEL_TOTH
                    tau_j = p3_in[j, s]
                    if tau_j <= 0.0:
                        continue

                base += (b * p) ** tau_j

            if base < SMALL_DEN:
                base = SMALL_DEN

            for i in range(n_comp):
                mid = model_ids_in[i, s]
                if mid == MODEL_MISSING:
                    continue

                qsat = p1_in[i, s]
                b = p2_in[i, s]
                if qsat <= 0.0 or b <= 0.0:
                    continue

                p = p_part[i]
                if p < EPS:
                    p = EPS

                if mid == MODEL_SSL:
                    tau_i = 1.0
                else:  # MODEL_TOTH
                    tau_i = p3_in[i, s]
                    if tau_i <= 0.0:
                        continue

                denom_i = base ** (1.0 / tau_i)
                if denom_i < SMALL_DEN:
                    denom_i = SMALL_DEN

                num = b * p
                q[i] += qsat * num / denom_i

    return q


# ============================================================
# INTERNAL NUMBA KERNELS
# ============================================================

@njit(cache=NUMBA_CACHE)
def iast_core(p_part, model_ids_in, p1_in, p2_in, p3_in, max_iter=IAST_MAX_ITER, tol=IAST_TOL):
    p_part, model_ids, qs_all, bs_all, fs_all, K_all, n_comp, n_site = pack(
        p_part, model_ids_in, p1_in, p2_in, p3_in
    )

    q_unary = np.empty(n_comp, dtype=np.float64)
    sum_unary = 0.0
    for i in range(n_comp):
        qv = q_multi(p_part[i], i, n_site, model_ids, qs_all, bs_all, fs_all, K_all)
        if not np.isfinite(qv) or qv < 0.0:
            qv = 0.0
        q_unary[i] = qv
        sum_unary += qv

    x_unary = np.empty(n_comp, dtype=np.float64)
    if sum_unary > EPS:
        for i in range(n_comp):
            x_unary[i] = q_unary[i] / sum_unary
    else:
        eq = 1.0 / n_comp
        for i in range(n_comp):
            x_unary[i] = eq

    active_count = 0
    active = np.zeros(n_comp, dtype=np.bool_)
    for i in range(n_comp):
        if x_unary[i] >= IAST_X_UNARY_FLOOR:
            active[i] = True
            active_count += 1

    q = q_unary.copy()
    err = 0.0
    ok = True

    if active_count > 1:
        p_sub = np.empty(active_count, dtype=np.float64)
        model_sub = np.empty((active_count, n_site), dtype=np.int64)
        qs_sub = np.empty((active_count, n_site), dtype=np.float64)
        bs_sub = np.empty((active_count, n_site), dtype=np.float64)
        fs_sub = np.empty((active_count, n_site), dtype=np.float64)
        k_sub = np.empty((active_count, n_site), dtype=np.float64)

        idx = 0
        for i in range(n_comp):
            if active[i]:
                p_sub[idx] = p_part[i]
                for s in range(n_site):
                    model_sub[idx, s] = model_ids[i, s]
                    qs_sub[idx, s] = qs_all[i, s]
                    bs_sub[idx, s] = bs_all[i, s]
                    fs_sub[idx, s] = fs_all[i, s]
                    k_sub[idx, s] = K_all[i, s]
                idx += 1

        x_sub, err_sub, ok_sub = solve_iast(
            p_sub, qs_sub, bs_sub, fs_sub, k_sub, model_sub, active_count, n_site,
            max_iter=max_iter, tol=tol
        )

        if ok_sub and np.isfinite(err_sub):
            qT_sub = total_loading(x_sub, p_sub, qs_sub, bs_sub, fs_sub, k_sub, model_sub, active_count, n_site)
            if np.isfinite(qT_sub) and qT_sub >= 0.0:
                q_sub = component_loadings_iast(x_sub, qT_sub, active_count)
                idx = 0
                for i in range(n_comp):
                    if active[i]:
                        qv = q_sub[idx]
                        if not np.isfinite(qv) or qv < 0.0:
                            ok_sub = False
                            break
                        q[i] = qv
                        idx += 1
                err = err_sub
                ok = ok_sub
            else:
                err = err_sub
                ok = False
        else:
            err = err_sub
            ok = False

    if not ok:
        for i in range(n_comp):
            q[i] = q_unary[i]

    out = np.empty((n_comp, 5), dtype=np.float64)
    okv = 1.0 if ok else 0.0
    q_total = 0.0
    for i in range(n_comp):
        q_total += q[i]

    for i in range(n_comp):
        out[i, 0] = p_part[i]
        out[i, 1] = 0.0 if q_total <= EPS else q[i] / q_total
        out[i, 2] = q[i]
        out[i, 3] = err
        out[i, 4] = okv

    return out


@njit(cache=NUMBA_CACHE)
def ext_core(p_part, model_ids_in, p1_in, p2_in, p3_in):
    n_comp = p_part.shape[0]
    max_sites = model_ids_in.shape[1]

    q_ext = extended_multisite_loadings_aligned(
        p_part, model_ids_in, p1_in, p2_in, p3_in, n_comp, max_sites
    )

    out = np.empty((n_comp, 3), dtype=np.float64)

    any_nan = False
    for i in range(n_comp):
        if np.isnan(q_ext[i]):
            any_nan = True
            break

    if any_nan:
        for i in range(n_comp):
            out[i, 0] = p_part[i]
            out[i, 1] = np.nan
            out[i, 2] = np.nan
        return out

    qtot = 0.0
    for i in range(n_comp):
        qtot += q_ext[i]

    for i in range(n_comp):
        out[i, 0] = p_part[i]
        out[i, 2] = q_ext[i]
        out[i, 1] = 0.0 if qtot <= 0.0 else (q_ext[i] / qtot)

    return out


@njit(cache=NUMBA_CACHE)
def unary_core(p_part, model_ids_in, p1_in, p2_in, p3_in):
    p_part, model_ids, qs_all, bs_all, fs_all, K_all, n_comp, n_site = pack(
        p_part, model_ids_in, p1_in, p2_in, p3_in
    )

    out = np.empty((n_comp, 2), dtype=np.float64)
    for i in range(n_comp):
        out[i, 0] = p_part[i]
        out[i, 1] = q_multi(p_part[i], i, n_site, model_ids, qs_all, bs_all, fs_all, K_all)
    return out


# ============================================================
# PUBLIC API
# ============================================================

def parse(components):
    """
    Parse user-friendly component input into numeric arrays for repeated use.

    Parameters
    ----------
    components : list of dict
        Component definitions with 'MoleculeName' and 'isotherms'.

    Returns
    -------
    tuple
        (names, model_ids_in, p1_in, p2_in, p3_in)
    """
    return parse_inputs(components)


def iast(p_part, components=None, parsed=None, max_iter=IAST_MAX_ITER, tol=IAST_TOL):
    """
    Run IAST for a mixture at a single condition.

    Parameters
    ----------
    p_part : array-like of shape (n_comp,)
        Partial pressure or concentration values.
    components : list of dict, optional
        User-friendly component input.
    parsed : tuple, optional
        Pre-parsed output from parse(...).
    max_iter : int, optional
        Maximum Newton iterations.
    tol : float, optional
        Tolerance for the log-spreading-pressure residual.

    Returns
    -------
    ndarray of shape (n_comp, 5)
        Columns: p_part, x, q, err, ok
    """
    p_part = np.asarray(p_part, dtype=np.float64)

    if parsed is None:
        if components is None:
            raise ValueError("Provide either components or parsed.")
        parsed = parse_inputs(components)

    _validate_parsed(parsed)
    _, model_ids_in, p1_in, p2_in, p3_in = parsed
    _validate_p_part(p_part, model_ids_in.shape[0])
    p_part = clip_partial_pressures(p_part)
    _check_bet_saturation(p_part, parsed)

    return iast_core(p_part, model_ids_in, p1_in, p2_in, p3_in, max_iter=max_iter, tol=tol)


def ext(p_part, components=None, parsed=None):
    """
    Run the approximate extended multisite mixture model.

    Parameters
    ----------
    p_part : array-like of shape (n_comp,)
        Partial pressure or concentration values.
    components : list of dict, optional
        User-friendly component input.
    parsed : tuple, optional
        Pre-parsed output from parse(...).

    Returns
    -------
    ndarray of shape (n_comp, 3)
        Columns: p_part, x, q

    Notes
    -----
    This method is approximate and is not generally thermodynamically
    consistent.

    Unlike unary() and iast(), ext() preserves the original parsed site
    columns exactly, so site index meaning stays aligned across components.

    It is only allowed when, for each site across components, all
    non-missing models belong to exactly one of these families:
    - SSL/Sips
    - SSL/Toth

    Validate against mixture data before relying on it for prediction.
    """
    p_part = np.asarray(p_part, dtype=np.float64)

    if parsed is None:
        if components is None:
            raise ValueError("Provide either components or parsed.")
        parsed = parse_inputs(components)

    _validate_parsed(parsed)
    _, model_ids_in, p1_in, p2_in, p3_in = parsed
    _validate_p_part(p_part, model_ids_in.shape[0])
    p_part = clip_partial_pressures(p_part)

    return ext_core(p_part, model_ids_in, p1_in, p2_in, p3_in)


def unary(p_part, components=None, parsed=None):
    """
    Run unary isotherm evaluations independently for each component.

    Parameters
    ----------
    p_part : array-like of shape (n_comp,)
        Partial pressure or concentration values.
    components : list of dict, optional
        User-friendly component input.
    parsed : tuple, optional
        Pre-parsed output from parse(...).

    Returns
    -------
    ndarray of shape (n_comp, 2)
        Columns: p_part, q
    """
    p_part = np.asarray(p_part, dtype=np.float64)

    if parsed is None:
        if components is None:
            raise ValueError("Provide either components or parsed.")
        parsed = parse_inputs(components)

    _validate_parsed(parsed)
    _, model_ids_in, p1_in, p2_in, p3_in = parsed
    _validate_p_part(p_part, model_ids_in.shape[0])
    p_part = clip_partial_pressures(p_part)

    return unary_core(p_part, model_ids_in, p1_in, p2_in, p3_in)
