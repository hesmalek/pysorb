import os
import time as tm

import numba
import numpy as np
from . import bt_mix as ps


R_GAS = 8.314462618153
Y_FLOOR = 1e-15
Q_FLOOR = 0.0
T_FLOOR = 100.0
P_FLOOR = 1_000.0
EXP_MIN = -80.0
EXP_MAX = 80.0
P_PART_MIN = 1e-15
P_PART_MAX = 1e8

NUMBA_CACHE = True  # set False during development to avoid stale cache issues

Q_METHOD_UNARY = 0
Q_METHOD_EXT = 1
Q_METHOD_IAST = 2


def _q_method_id(name):
    key = str(name).strip().lower()
    if key == "unary":
        return Q_METHOD_UNARY
    if key == "ext":
        return Q_METHOD_EXT
    if key == "iast":
        return Q_METHOD_IAST
    raise ValueError('q_eq_method must be "unary", "ext", or "iast".')


def _component_name(comp, index):
    return str(comp.get("MoleculeName", comp.get("name", f"component_{index + 1}")))


def _legacy_site_to_isotherm(site):
    return ["Sips", site["qs"], site["b0"], site.get("n", 1.0), site.get("d_h", 0.0)]


def _parse_isotherm_entry(entry, component_name, site_index):
    if isinstance(entry, dict):
        model_name = entry.get("model", entry.get("name", "Sips"))
        params = list(entry.get("params", ()))
        if not params:
            if "qs" in entry and "b0" in entry:
                params = [entry["qs"], entry["b0"], entry.get("n", 1.0)]
            elif "K" in entry:
                params = [entry["K"]]
        d_h = float(entry.get("d_h", entry.get("dh", entry.get("dH", 0.0))))
    else:
        if len(entry) < 1:
            raise ValueError(f"Invalid isotherm for component {component_name}, site {site_index}: {entry}")
        model_name = entry[0]
        mid = ps.model_name_to_id(model_name)
        n_params = ps.MODEL_PARAM_COUNTS[mid]
        if len(entry) < 1 + n_params:
            raise ValueError(
                f"Component {component_name}, site {site_index}, model {model_name}: "
                f"expected at least {n_params} parameter(s)."
            )
        params = list(entry[1:1 + n_params])
        d_h = float(entry[1 + n_params]) if len(entry) > 1 + n_params else 0.0

    mid = ps.model_name_to_id(model_name)
    n_params = ps.MODEL_PARAM_COUNTS[mid]
    if len(params) < n_params:
        raise ValueError(
            f"Component {component_name}, site {site_index}, model {model_name}: "
            f"expected {n_params} parameter(s), got {len(params)}."
        )
    return mid, [float(v) for v in params[:n_params]], d_h


def _parse_component_isotherms(components):
    names = [_component_name(comp, i) for i, comp in enumerate(components)]
    raw_isotherms = []
    for i, comp in enumerate(components):
        if "isotherms" in comp:
            entries = comp["isotherms"]
        elif "sites" in comp:
            entries = [_legacy_site_to_isotherm(site) for site in comp["sites"]]
        else:
            raise ValueError(f"Component {names[i]} must define 'isotherms' or legacy 'sites'.")
        if len(entries) < 1:
            raise ValueError(f"Component {names[i]} must define at least one isotherm.")
        raw_isotherms.append(entries)

    n_sites = max(len(entries) for entries in raw_isotherms)
    model_ids = np.full((len(components), n_sites), ps.MODEL_MISSING, dtype=np.int32)
    p1 = np.zeros((len(components), n_sites), dtype=np.float64)
    p2 = np.zeros((len(components), n_sites), dtype=np.float64)
    p3 = np.zeros((len(components), n_sites), dtype=np.float64)
    d_h = np.zeros((len(components), n_sites), dtype=np.float64)

    for i, entries in enumerate(raw_isotherms):
        for site_index, entry in enumerate(entries):
            mid, params, dh = _parse_isotherm_entry(entry, names[i], site_index)
            model_ids[i, site_index] = mid
            if len(params) >= 1:
                p1[i, site_index] = params[0]
            if len(params) >= 2:
                p2[i, site_index] = params[1]
            if len(params) >= 3:
                p3[i, site_index] = params[2]
            d_h[i, site_index] = dh

    return names, model_ids, p1, p2, p3, d_h


def _ext_site_family_ok_py(model_ids, site_index):
    has_ssl = False
    has_sips = False
    has_toth = False

    for comp in range(model_ids.shape[0]):
        mid = model_ids[comp, site_index]
        if mid == ps.MODEL_MISSING:
            continue
        if mid == ps.MODEL_SSL:
            has_ssl = True
        elif mid == ps.MODEL_SIPS:
            has_sips = True
        elif mid == ps.MODEL_TOTH:
            has_toth = True
        else:
            return False

    return (has_ssl or has_sips or has_toth) and not (has_sips and has_toth)


def _validate_ext_method(model_ids):
    for site_index in range(model_ids.shape[1]):
        if not _ext_site_family_ok_py(model_ids, site_index):
            raise ValueError(
                "q_eq_method='ext' supports only aligned SSL/Sips or SSL/Toth site families. "
                f"Site {site_index} has an unsupported model combination; use q_eq_method='iast' "
                "for general multisite mixtures."
            )


def _build_model_inputs(inputs):
    components = inputs["components"]
    if len(components) < 1:
        raise ValueError("At least one component is required.")

    y_in = np.array([comp["y_in"] for comp in components], dtype=np.float64)
    ldf = np.array([comp["ldf"] for comp in components], dtype=np.float64)
    d_h_ads = np.array([comp["d_h_ads"] for comp in components], dtype=np.float64)
    initial_y = np.array([comp.get("initial_y", 1e-12) for comp in components], dtype=np.float64)
    process = inputs["process"]
    for i, comp in enumerate(components):
        name = _component_name(comp, i)
        if "Cp_gas" not in comp:
            raise ValueError(f"Component {name} is missing Cp_gas [J/mol/K].")
        if "MW" not in comp:
            raise ValueError(f"Component {name} is missing MW [kg/mol].")
    component_cp_gas = np.array([comp["Cp_gas"] for comp in components], dtype=np.float64)
    component_mw = np.array([comp["MW"] for comp in components], dtype=np.float64)
    component_names, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh = _parse_component_isotherms(components)
    n_sites = iso_model_ids.shape[1]
    q_method_name = str(inputs.get("q_eq_method", inputs.get("loading_method", "ext"))).lower()
    q_method_id = _q_method_id(q_method_name)

    solver_cfg = inputs.get("solver", {})
    n_components = len(components)
    if "P_out" in process:
        p_out = float(process["P_out"])
    elif "p_out" in process:
        p_out = float(process["p_out"])
    elif "P0" in process:
        p_out = float(process["P0"])
    else:
        raise KeyError("process.P_out is required.")

    t_end = float(inputs["domain"]["t_end"])
    dt = float(inputs["domain"]["dt"])
    dt_min = float(inputs["domain"].get("dt_min", dt / 16.0))
    dt_max = float(inputs["domain"].get("dt_max", min(t_end, dt * 4.0)))
    expected_dt = min(dt, dt_max)
    default_max_steps = max(int(t_end / expected_dt) + 1000, 1000)

    model = {
        "n_components": n_components,
        "n_sites": n_sites,
        "state_size": 2 * n_components + 3,
        "component_names": component_names,
        "length": float(inputs["bed"]["L"]),
        "dia_in": float(inputs["bed"]["dia_in"]),
        "eps_b": float(inputs["bed"]["eps_b"]),
        "flow_in": float(inputs["process"]["Flow_in"]),
        "t0": float(inputs["process"]["T0"]),
        "t_in": float(inputs["process"]["T_in"]),
        "t_amb": float(inputs["process"]["T_amb"]),
        "p_amb": float(inputs["process"]["P_amb"]),
        "p_out": p_out,
        "mu": float(inputs["process"]["mu"]),
        "component_cp_gas": component_cp_gas,
        "component_mw": component_mw,
        "dia_p": float(inputs["pellet"]["dia_p"]),
        "rho_s": float(inputs["pellet"]["rho_s"]),
        "cp_s": float(inputs["pellet"]["Cp_s"]),
        "n_grid": int(inputs["domain"]["N"]),
        "t_end": t_end,
        "dt": dt,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "tol_err": float(inputs["domain"].get("tol_err", 2e-3)),
        "max_steps": int(inputs["domain"].get("max_steps", default_max_steps)),
        "dm": float(inputs["transport"]["Dm"]),
        "dl_model": str(inputs["transport"].get("DL_model", "Wakao")).lower(),
        "dl_custom": None if "DL" not in inputs["transport"] else float(inputs["transport"]["DL"]),
        "dl0_custom": None if "DL0" not in inputs["transport"] else float(inputs["transport"]["DL0"]),
        "h_in": float(inputs["heat_transfer"]["h_in"]),
        "h_out": float(inputs["heat_transfer"]["h_out"]),
        "kz": float(inputs["heat_transfer"]["kz"]),
        "kw": float(inputs["heat_transfer"]["kw"]),
        "rho_w": float(inputs["wall"]["rho_w"]),
        "cp_w": float(inputs["wall"]["Cp_w"]),
        "w_w": float(inputs["wall"]["w_w"]),
        "y_in": y_in,
        "ldf": ldf,
        "d_h_ads": d_h_ads,
        "initial_y": initial_y,
        "q_method_name": q_method_name,
        "q_method_id": q_method_id,
        "iso_model_ids": iso_model_ids,
        "iso_p1": iso_p1,
        "iso_p2_ref": iso_p2_ref,
        "iso_p3": iso_p3,
        "iso_dh": iso_dh,
        "iso_t_ref": float(inputs.get("isotherm_t_ref", 0.0) or 0.0),
        "newton_tol": float(solver_cfg.get("newton_tol", 5e-3)),
        "newton_max_iter": int(solver_cfg.get("newton_max_iter", 7)),
        "newton_delta": float(solver_cfg.get("newton_delta", 1e-6)),
        "newton_alpha": float(solver_cfg.get("newton_alpha", 1.0)),
        "newton_min_alpha": float(solver_cfg.get("newton_min_alpha", 1e-4)),
        "newton_line_search_max": int(solver_cfg.get("newton_line_search_max", 10)),
        "startup_dt_min": float(solver_cfg.get("startup_dt_min", 0.005)),
        "newton_retry_max_iter": int(solver_cfg.get("newton_retry_max_iter", 15)),
        "newton_retry_line_search_max": int(solver_cfg.get("newton_retry_line_search_max", 14)),
        "newton_retry_min_alpha": float(solver_cfg.get("newton_retry_min_alpha", 1e-5)),
        "max_rejected_steps": int(solver_cfg.get("max_rejected_steps", 2000)),
        "raise_on_incomplete": bool(solver_cfg.get("raise_on_incomplete", True)),
    }
    _validate_model_inputs(model)
    return model


def _validate_model_inputs(model):
    positive_fields = (
        "length",
        "dia_in",
        "flow_in",
        "t0",
        "t_in",
        "t_amb",
        "p_amb",
        "p_out",
        "mu",
        "dia_p",
        "rho_s",
        "cp_s",
        "dt",
        "dt_min",
        "dt_max",
        "tol_err",
        "dm",
        "h_in",
        "h_out",
        "kw",
        "rho_w",
        "cp_w",
        "w_w",
        "newton_tol",
        "newton_delta",
        "newton_alpha",
        "newton_min_alpha",
        "startup_dt_min",
        "newton_retry_min_alpha",
    )
    for field in positive_fields:
        if not np.isfinite(model[field]) or model[field] <= 0.0:
            raise ValueError(f"{field} must be a positive finite value.")

    if model["n_grid"] < 2:
        raise ValueError("domain.N must be at least 2.")
    if model["max_steps"] < 2:
        raise ValueError("domain.max_steps must be at least 2.")
    if model["newton_max_iter"] < 1:
        raise ValueError("solver.newton_max_iter must be at least 1.")
    if model["newton_line_search_max"] < 0:
        raise ValueError("solver.newton_line_search_max must be >= 0.")
    if model["newton_retry_line_search_max"] < 0:
        raise ValueError("solver.newton_retry_line_search_max must be >= 0.")
    if model["newton_retry_max_iter"] < 1:
        raise ValueError("solver.newton_retry_max_iter must be at least 1.")
    if model["newton_min_alpha"] > model["newton_alpha"]:
        raise ValueError("solver.newton_min_alpha must be <= solver.newton_alpha.")
    if model["newton_retry_min_alpha"] > model["newton_alpha"]:
        raise ValueError("solver.newton_retry_min_alpha must be <= solver.newton_alpha.")
    if model["max_rejected_steps"] < 1:
        raise ValueError("solver.max_rejected_steps must be at least 1.")
    if not 0.0 < model["eps_b"] < 1.0:
        raise ValueError("bed.eps_b must be between 0 and 1.")
    if model["dt_min"] > model["dt"]:
        raise ValueError("domain.dt_min must be less than or equal to domain.dt.")
    if model["dt_max"] < model["dt"]:
        raise ValueError("domain.dt_max must be greater than or equal to domain.dt.")
    if model["dl_model"] not in ("custom", "wakao", "gunn"):
        raise ValueError('transport.DL_model must be "custom", "Wakao", or "Gunn".')
    if model["dl_model"] == "custom":
        if model["dl_custom"] is None or not np.isfinite(model["dl_custom"]) or model["dl_custom"] <= 0.0:
            raise ValueError('transport.DL must be a positive finite value when DL_model is "custom".')
    elif model["dl_custom"] is not None and (not np.isfinite(model["dl_custom"]) or model["dl_custom"] <= 0.0):
        raise ValueError("transport.DL must be a positive finite value when provided.")
    if model["dl0_custom"] is not None and (not np.isfinite(model["dl0_custom"]) or model["dl0_custom"] <= 0.0):
        raise ValueError("transport.DL0 must be a positive finite value when provided.")

    for name, values in (
        ("y_in", model["y_in"]),
        ("initial_y", model["initial_y"]),
        ("ldf[1/s]", model["ldf"]),
        ("component_cp_gas", model["component_cp_gas"]),
        ("component_mw", model["component_mw"]),
        ("iso_p1", model["iso_p1"]),
        ("iso_p2_ref", model["iso_p2_ref"]),
        ("iso_p3", model["iso_p3"]),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} values must be finite.")
        if np.any(values < 0.0):
            raise ValueError(f"{name} values must be non-negative.")

    if np.any(model["component_cp_gas"] <= 0.0):
        raise ValueError("component Cp_gas values must be positive.")
    if np.any(model["component_mw"] <= 0.0):
        raise ValueError("component MW values must be positive.")

    if not np.all(np.isfinite(model["d_h_ads"])) or not np.all(np.isfinite(model["iso_dh"])):
        raise ValueError("enthalpy values must be finite.")
    if not np.isfinite(model["iso_t_ref"]) or model["iso_t_ref"] < 0.0:
        raise ValueError("isotherm_t_ref must be greater than 0 K when provided.")
    if np.any(model["iso_model_ids"] < ps.MODEL_MISSING):
        raise ValueError("isotherm model ids are invalid.")
    if model["q_method_id"] == Q_METHOD_EXT:
        _validate_ext_method(model["iso_model_ids"])
    if not np.isclose(np.sum(model["y_in"]), 1.0, rtol=1e-6, atol=1e-9):
        raise ValueError("component y_in values must sum to 1.0.")
    if not np.isclose(np.sum(model["initial_y"]), 1.0, rtol=1e-6, atol=1e-9):
        raise ValueError("component initial_y values must sum to 1.0.")


@numba.njit(cache=NUMBA_CACHE)
def rho_g(p, t, mw_mix):
    return p / (R_GAS * t) * mw_mix


@numba.njit(cache=NUMBA_CACHE)
def weighted_gas_property(y, values, n_components):
    weighted = 0.0
    y_sum = 0.0
    for comp in range(n_components):
        yv = y[comp]
        if not np.isfinite(yv) or yv < Y_FLOOR:
            yv = Y_FLOOR
        weighted += yv * values[comp]
        y_sum += yv
    if y_sum <= 0.0:
        return values[0]
    return weighted / y_sum


@numba.njit(cache=NUMBA_CACHE)
def weighted_gas_property_from_state(u, base, values, n_components):
    weighted = 0.0
    y_sum = 0.0
    for comp in range(n_components):
        yv = u[base + comp]
        if not np.isfinite(yv) or yv < Y_FLOOR:
            yv = Y_FLOOR
        weighted += yv * values[comp]
        y_sum += yv
    if y_sum <= 0.0:
        return values[0]
    return weighted / y_sum


@numba.njit(cache=NUMBA_CACHE)
def ergun_face_velocity(p_left, p_right, t_left, t_right, dz, eps_b, mu, dia_p, mw_face):
    pressure_gradient = (p_left - p_right) / dz
    if not np.isfinite(pressure_gradient):
        return 0.0

    direction = 1.0
    if pressure_gradient < 0.0:
        direction = -1.0
        pressure_gradient = -pressure_gradient
    if pressure_gradient <= 0.0:
        return 0.0

    p_face = 0.5 * (p_left + p_right)
    if p_face < P_FLOOR:
        p_face = P_FLOOR
    t_face = 0.5 * (t_left + t_right)
    if t_face < T_FLOOR:
        t_face = T_FLOOR

    rho_face = rho_g(p_face, t_face, mw_face)
    aterm = 150.0 * mu * (1.0 - eps_b) * (1.0 - eps_b) / (eps_b * eps_b * dia_p * dia_p)
    bterm = 1.75 * (1.0 - eps_b) * rho_face / (eps_b * dia_p)
    if bterm <= 1e-30:
        return direction * pressure_gradient / (aterm + 1e-30)

    disc = aterm * aterm + 4.0 * bterm * pressure_gradient
    if disc < 0.0:
        disc = 0.0
    return direction * (-aterm + np.sqrt(disc)) / (2.0 * bterm)


@numba.njit(cache=NUMBA_CACHE)
def compute_face_velocities(u, n_grid, n_components, dz, eps_b, mu, dia_p, component_mw, nu_in):
    nv = 2 * n_components + 3
    t_idx = 2 * n_components
    p_idx = t_idx + 2
    nu_face = np.empty(n_grid, dtype=np.float64)
    nu_face[0] = nu_in

    for face in range(1, n_grid):
        left = (face - 1) * nv
        right = face * nv
        mw_left = weighted_gas_property_from_state(u, left, component_mw, n_components)
        mw_right = weighted_gas_property_from_state(u, right, component_mw, n_components)
        nu_face[face] = ergun_face_velocity(
            u[left + p_idx],
            u[right + p_idx],
            u[left + t_idx],
            u[right + t_idx],
            dz,
            eps_b,
            mu,
            dia_p,
            0.5 * (mw_left + mw_right),
        )

    return nu_face


@numba.njit(cache=NUMBA_CACHE)
def q_eq(y, t_val, p_val, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, n_components):
    t_safe = t_val
    if t_safe < T_FLOOR:
        t_safe = T_FLOOR
    p_safe = p_val
    if p_safe < P_FLOOR:
        p_safe = P_FLOOR
    p_bar = p_safe / 1e5

    p_part = np.empty(n_components, dtype=np.float64)
    p2 = iso_p2_ref.copy()
    n_sites = iso_model_ids.shape[1]

    for comp in range(n_components):
        y_safe = y[comp]
        if y_safe < Y_FLOOR:
            y_safe = Y_FLOOR
        p_part[comp] = y_safe * p_bar
        if p_part[comp] < P_PART_MIN:
            p_part[comp] = P_PART_MIN
        elif p_part[comp] > P_PART_MAX:
            p_part[comp] = P_PART_MAX

        for site in range(n_sites):
            if iso_t_ref > 0.0:
                arg = (-iso_dh[comp, site] / R_GAS) * ((1.0 / t_safe) - (1.0 / iso_t_ref))
            else:
                arg = -iso_dh[comp, site] / (R_GAS * t_safe)
            if arg < EXP_MIN:
                arg = EXP_MIN
            elif arg > EXP_MAX:
                arg = EXP_MAX
            p2[comp, site] = iso_p2_ref[comp, site] * np.exp(arg)

    if q_method_id == Q_METHOD_IAST:
        out = ps.iast_core(p_part, iso_model_ids, iso_p1, p2, iso_p3)
        qe = out[:, 2].copy()
        for comp in range(n_components):
            if not np.isfinite(qe[comp]) or qe[comp] < 0.0:
                qe[comp] = 0.0
        return qe
    if q_method_id == Q_METHOD_UNARY:
        out = ps.unary_core(p_part, iso_model_ids, iso_p1, p2, iso_p3)
        return out[:, 1].copy()

    out = ps.ext_core(p_part, iso_model_ids, iso_p1, p2, iso_p3)
    qe = out[:, 2].copy()
    for comp in range(n_components):
        if not np.isfinite(qe[comp]):
            qe[comp] = 0.0
    return qe


@numba.njit(cache=NUMBA_CACHE)
def q_eq_iast_failure_fallback_used(y, t_val, p_val, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, n_components):
    if q_method_id != Q_METHOD_IAST:
        return False

    t_safe = t_val
    if t_safe < T_FLOOR:
        t_safe = T_FLOOR
    p_safe = p_val
    if p_safe < P_FLOOR:
        p_safe = P_FLOOR
    p_bar = p_safe / 1e5

    p_part = np.empty(n_components, dtype=np.float64)
    p2 = iso_p2_ref.copy()
    n_sites = iso_model_ids.shape[1]

    for comp in range(n_components):
        y_safe = y[comp]
        if y_safe < Y_FLOOR:
            y_safe = Y_FLOOR
        p_part[comp] = y_safe * p_bar
        if p_part[comp] < P_PART_MIN:
            p_part[comp] = P_PART_MIN
        elif p_part[comp] > P_PART_MAX:
            p_part[comp] = P_PART_MAX

        for site in range(n_sites):
            if iso_t_ref > 0.0:
                arg = (-iso_dh[comp, site] / R_GAS) * ((1.0 / t_safe) - (1.0 / iso_t_ref))
            else:
                arg = -iso_dh[comp, site] / (R_GAS * t_safe)
            if arg < EXP_MIN:
                arg = EXP_MIN
            elif arg > EXP_MAX:
                arg = EXP_MAX
            p2[comp, site] = iso_p2_ref[comp, site] * np.exp(arg)

    out = ps.iast_core(p_part, iso_model_ids, iso_p1, p2, iso_p3)
    return out[0, 4] < 0.5


@numba.njit(cache=NUMBA_CACHE)
def count_iast_failure_fallback_rows(results_nm, nrows, n_grid, n_components, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref):
    if q_method_id != Q_METHOD_IAST:
        return 0, 0

    nv = 2 * n_components + 3
    t_idx = 2 * n_components
    p_idx = t_idx + 2
    failure_fallback_rows = 0
    failure_fallback_cells = 0

    for row in range(nrows):
        row_has_failure_fallback = False
        for i in range(n_grid):
            base = i * nv
            y = np.empty(n_components, dtype=np.float64)
            for comp in range(n_components):
                y[comp] = results_nm[row, base + comp]

            if q_eq_iast_failure_fallback_used(
                y,
                results_nm[row, base + t_idx],
                results_nm[row, base + p_idx],
                q_method_id,
                iso_model_ids,
                iso_p1,
                iso_p2_ref,
                iso_p3,
                iso_dh,
                iso_t_ref,
                n_components,
            ):
                failure_fallback_cells += 1
                row_has_failure_fallback = True

        if row_has_failure_fallback:
            failure_fallback_rows += 1

    return failure_fallback_rows, failure_fallback_cells


@numba.njit(cache=NUMBA_CACHE)
def project_state(u, n_grid, n_components):
    nv = 2 * n_components + 3
    t_idx = 2 * n_components
    tw_idx = t_idx + 1
    p_idx = t_idx + 2

    for i in range(n_grid):
        base = i * nv
        for comp in range(n_components):
            y_idx = base + comp
            if not np.isfinite(u[y_idx]) or u[y_idx] < Y_FLOOR:
                u[y_idx] = Y_FLOOR

        for comp in range(n_components):
            q_idx = base + n_components + comp
            if not np.isfinite(u[q_idx]) or u[q_idx] < Q_FLOOR:
                u[q_idx] = Q_FLOOR

        if not np.isfinite(u[base + t_idx]) or u[base + t_idx] < T_FLOOR:
            u[base + t_idx] = T_FLOOR
        if not np.isfinite(u[base + tw_idx]) or u[base + tw_idx] < T_FLOOR:
            u[base + tw_idx] = T_FLOOR
        if not np.isfinite(u[base + p_idx]) or u[base + p_idx] < P_FLOOR:
            u[base + p_idx] = P_FLOOR

    return u


@numba.njit(cache=NUMBA_CACHE)
def apply_inlet_guess(u, n_components, y_in, t_in):
    t_idx = 2 * n_components
    tw_idx = t_idx + 1
    for comp in range(n_components):
        u[comp] = y_in[comp]
    u[t_idx] = t_in
    u[tw_idx] = t_in
    return u


@numba.njit(cache=NUMBA_CACHE)
def compute_qe_cache(u, n_grid, n_components, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref):
    nv = 2 * n_components + 3
    t_idx = 2 * n_components
    p_idx = t_idx + 2
    qe_cache = np.empty((n_grid, n_components), dtype=np.float64)

    for i in range(n_grid):
        base = i * nv
        y = np.empty(n_components, dtype=np.float64)
        for comp in range(n_components):
            y[comp] = u[base + comp]
        qe = q_eq(
            y,
            u[base + t_idx],
            u[base + p_idx],
            q_method_id,
            iso_model_ids,
            iso_p1,
            iso_p2_ref,
            iso_p3,
            iso_dh,
            iso_t_ref,
            n_components,
        )
        for comp in range(n_components):
            qe_cache[i, comp] = qe[comp]

    return qe_cache


@numba.njit(cache=NUMBA_CACHE)
def residual_node_major(
    u,
    u_old,
    dt_val,
    n_grid,
    n_components,
    n_sites,
    dz,
    eps_b,
    mu,
    dia_p,
    component_cp_gas,
    component_mw,
    rho_s,
    cp_s,
    t_in,
    t_amb,
    p_out,
    dl,
    dl0,
    y_in,
    ldf,
    q_method_id,
    iso_model_ids,
    iso_p1,
    iso_p2_ref,
    iso_p3,
    iso_dh,
    iso_t_ref,
    d_h_ads,
    kz,
    kw,
    rho_w,
    cp_w,
    r_in,
    r_out,
    r_sq_diff,
    h_in,
    h_out,
    nu_in,
):
    nv = 2 * n_components + 3
    t_idx = 2 * n_components
    tw_idx = t_idx + 1
    p_idx = t_idx + 2

    r = np.zeros(u.shape[0], dtype=np.float64)
    nu_face = compute_face_velocities(u, n_grid, n_components, dz, eps_b, mu, dia_p, component_mw, nu_in)

    idz = 1.0 / dz
    idz2 = idz * idz
    idt = 1.0 / dt_val
    eps_f = (1.0 - eps_b) / eps_b
    aconst = 150.0 * mu * (1.0 - eps_b) * (1.0 - eps_b) / (eps_b * eps_b * dia_p * dia_p)
    qe_cache = compute_qe_cache(u, n_grid, n_components, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref)

    for i in range(n_grid):
        base = i * nv
        t_bed = u[base + t_idx]
        t_wall = u[base + tw_idx]
        p = u[base + p_idx]

        y = np.empty(n_components, dtype=np.float64)
        q = np.empty(n_components, dtype=np.float64)
        for comp in range(n_components):
            y[comp] = u[base + comp]
            q[comp] = u[base + n_components + comp]
        qe = qe_cache[i]

        cp_gas_local = weighted_gas_property(y, component_cp_gas, n_components)
        mw_local = weighted_gas_property(y, component_mw, n_components)
        rho = rho_g(p, t_bed, mw_local)
        bterm = 1.75 * (1.0 - eps_b) * rho / (eps_b * dia_p)

        if i == 0:
            bp = (i + 1) * nv
            tp = u[bp + t_idx]
            pp = u[bp + p_idx]

            for comp in range(n_components):
                yp = u[bp + comp]
                r[base + comp] = dl0 * (yp - y[comp]) * idz + nu_in * (y_in[comp] - y[comp])
                q_idx = n_components + comp
                r[base + q_idx] = (q[comp] - u_old[base + q_idx]) * idt - ldf[comp] * (qe[comp] - q[comp])

            r[base + t_idx] = kz * (tp - t_bed) * idz + eps_b * nu_in * rho * cp_gas_local * (t_in - t_bed)
            r[base + tw_idx] = t_wall - t_in
            r[base + p_idx] = (p - pp) * idz - aconst * nu_in - bterm * nu_in * nu_in
            continue

        if i == n_grid - 1:
            bm = (i - 1) * nv
            tm = u[bm + t_idx]
            twm = u[bm + tw_idx]

            for comp in range(n_components):
                ym = u[bm + comp]
                r[base + comp] = (y[comp] - ym) * idz
                q_idx = n_components + comp
                r[base + q_idx] = (q[comp] - u_old[base + q_idx]) * idt - ldf[comp] * (qe[comp] - q[comp])

            r[base + t_idx] = (t_bed - tm) * idz
            r[base + tw_idx] = (t_wall - twm)

            r[base + p_idx] = p - p_out
            continue

        bm = (i - 1) * nv
        bp = (i + 1) * nv
        tm = u[bm + t_idx]
        tp = u[bp + t_idx]
        twm = u[bm + tw_idx]
        twp = u[bp + tw_idx]
        pm = u[bm + p_idx]
        pp = u[bp + p_idx]
        nu_left = nu_face[i]
        nu_right = nu_face[i + 1]

        dtemp = (t_bed - u_old[base + t_idx]) * idt
        dtw = (t_wall - u_old[base + tw_idx]) * idt
        dp = (p - u_old[base + p_idx]) * idt
        q_sum = 0.0
        dq_sum = 0.0
        adsorption_heat = 0.0

        for comp in range(n_components):
            q_idx = n_components + comp
            ym = u[bm + comp]
            yp = u[bp + comp]
            qm = u_old[base + q_idx]
            dy = (y[comp] - u_old[base + comp]) * idt
            dq = (q[comp] - qm) * idt

            disp = (pp / tp * (yp - y[comp]) * idz - p / t_bed * (y[comp] - ym) * idz) * idz
            conv = (nu_right * p * y[comp] / t_bed - nu_left * pm * ym / tm) * idz

            r[base + comp] = (
                dy
                + y[comp] / p * dp
                - y[comp] / t_bed * dtemp
                - (t_bed / p) * dl * disp
                + (t_bed / p) * conv
                + (R_GAS * t_bed / p) * eps_f * dq * rho_s
            )
            r[base + q_idx] = dq - ldf[comp] * (qe[comp] - q[comp])

            q_sum += q[comp]
            dq_sum += dq
            adsorption_heat += d_h_ads[comp] * dq

        tdiff = kz * (tp - 2.0 * t_bed + tm) * idz2
        r[base + t_idx] = (
            (1.0 - eps_b) * (rho_s * cp_s + cp_gas_local * q_sum * rho_s) * dtemp
            - tdiff
            + eps_b * cp_gas_local * dp / R_GAS
            + eps_b * cp_gas_local * (nu_right * p - nu_left * pm) * idz / R_GAS
            + (1.0 - eps_b) * cp_gas_local * t_bed * dq_sum * rho_s
            + (1.0 - eps_b) * adsorption_heat * rho_s
            + 2.0 * h_in / r_in * (t_bed - t_wall)
        )

        twdiff = kw * (twp - 2.0 * t_wall + twm) * idz2
        r[base + tw_idx] = (
            rho_w * cp_w * dtw
            - twdiff
            - 2.0 * r_in * h_in / r_sq_diff * (t_bed - t_wall)
            + 2.0 * r_out * h_out / r_sq_diff * (t_wall - t_amb)
        )

        r[base + p_idx] = (
            dp / p
            - dtemp / t_bed
            + (t_bed / p) * (nu_right * p / t_bed - nu_left * pm / tm) * idz
            + (R_GAS * t_bed / p) * eps_f * dq_sum * rho_s
        )

    return r, nu_face


@numba.njit(cache=NUMBA_CACHE)
def residual_node_major_window(
    u,
    u_old,
    dt_val,
    n_grid,
    n_components,
    n_sites,
    dz,
    eps_b,
    mu,
    dia_p,
    component_cp_gas,
    component_mw,
    rho_s,
    cp_s,
    t_in,
    t_amb,
    p_out,
    dl,
    dl0,
    y_in,
    ldf,
    q_method_id,
    iso_model_ids,
    iso_p1,
    iso_p2_ref,
    iso_p3,
    iso_dh,
    iso_t_ref,
    d_h_ads,
    kz,
    kw,
    rho_w,
    cp_w,
    r_in,
    r_out,
    r_sq_diff,
    h_in,
    h_out,
    nu_in,
    row_start,
    row_stop,
    qe_cache,
    perturbed_node,
    recompute_perturbed_qe,
):
    nv = 2 * n_components + 3
    t_idx = 2 * n_components
    tw_idx = t_idx + 1
    p_idx = t_idx + 2

    r = np.zeros((row_stop - row_start) * nv, dtype=np.float64)
    nu_face = compute_face_velocities(u, n_grid, n_components, dz, eps_b, mu, dia_p, component_mw, nu_in)

    idz = 1.0 / dz
    idz2 = idz * idz
    idt = 1.0 / dt_val
    eps_f = (1.0 - eps_b) / eps_b
    aconst = 150.0 * mu * (1.0 - eps_b) * (1.0 - eps_b) / (eps_b * eps_b * dia_p * dia_p)

    for i in range(row_start, row_stop):
        base = i * nv
        rbase = (i - row_start) * nv
        t_bed = u[base + t_idx]
        t_wall = u[base + tw_idx]
        p = u[base + p_idx]

        y = np.empty(n_components, dtype=np.float64)
        q = np.empty(n_components, dtype=np.float64)
        for comp in range(n_components):
            y[comp] = u[base + comp]
            q[comp] = u[base + n_components + comp]
        qe = np.empty(n_components, dtype=np.float64)
        if recompute_perturbed_qe and i == perturbed_node:
            qe_updated = q_eq(y, t_bed, p, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, n_components)
            for comp in range(n_components):
                qe[comp] = qe_updated[comp]
        else:
            for comp in range(n_components):
                qe[comp] = qe_cache[i, comp]

        cp_gas_local = weighted_gas_property(y, component_cp_gas, n_components)
        mw_local = weighted_gas_property(y, component_mw, n_components)
        rho = rho_g(p, t_bed, mw_local)
        bterm = 1.75 * (1.0 - eps_b) * rho / (eps_b * dia_p)

        if i == 0:
            bp = (i + 1) * nv
            tp = u[bp + t_idx]
            pp = u[bp + p_idx]

            for comp in range(n_components):
                yp = u[bp + comp]
                r[rbase + comp] = dl0 * (yp - y[comp]) * idz + nu_in * (y_in[comp] - y[comp])
                q_idx = n_components + comp
                r[rbase + q_idx] = (q[comp] - u_old[base + q_idx]) * idt - ldf[comp] * (qe[comp] - q[comp])

            r[rbase + t_idx] = kz * (tp - t_bed) * idz + eps_b * nu_in * rho * cp_gas_local * (t_in - t_bed)
            r[rbase + tw_idx] = t_wall - t_in
            r[rbase + p_idx] = (p - pp) * idz - aconst * nu_in - bterm * nu_in * nu_in
            continue

        if i == n_grid - 1:
            bm = (i - 1) * nv
            tm = u[bm + t_idx]
            twm = u[bm + tw_idx]

            for comp in range(n_components):
                ym = u[bm + comp]
                r[rbase + comp] = (y[comp] - ym) * idz
                q_idx = n_components + comp
                r[rbase + q_idx] = (q[comp] - u_old[base + q_idx]) * idt - ldf[comp] * (qe[comp] - q[comp])

            r[rbase + t_idx] = (t_bed - tm) * idz
            r[rbase + tw_idx] = t_wall - twm
            r[rbase + p_idx] = p - p_out
            continue

        bm = (i - 1) * nv
        bp = (i + 1) * nv
        tm = u[bm + t_idx]
        tp = u[bp + t_idx]
        twm = u[bm + tw_idx]
        twp = u[bp + tw_idx]
        pm = u[bm + p_idx]
        pp = u[bp + p_idx]
        nu_left = nu_face[i]
        nu_right = nu_face[i + 1]

        dtemp = (t_bed - u_old[base + t_idx]) * idt
        dtw = (t_wall - u_old[base + tw_idx]) * idt
        dp = (p - u_old[base + p_idx]) * idt
        q_sum = 0.0
        dq_sum = 0.0
        adsorption_heat = 0.0

        for comp in range(n_components):
            q_idx = n_components + comp
            ym = u[bm + comp]
            yp = u[bp + comp]
            qm = u_old[base + q_idx]
            dy = (y[comp] - u_old[base + comp]) * idt
            dq = (q[comp] - qm) * idt

            disp = (pp / tp * (yp - y[comp]) * idz - p / t_bed * (y[comp] - ym) * idz) * idz
            conv = (nu_right * p * y[comp] / t_bed - nu_left * pm * ym / tm) * idz

            r[rbase + comp] = (
                dy
                + y[comp] / p * dp
                - y[comp] / t_bed * dtemp
                - (t_bed / p) * dl * disp
                + (t_bed / p) * conv
                + (R_GAS * t_bed / p) * eps_f * dq * rho_s
            )
            r[rbase + q_idx] = dq - ldf[comp] * (qe[comp] - q[comp])

            q_sum += q[comp]
            dq_sum += dq
            adsorption_heat += d_h_ads[comp] * dq

        tdiff = kz * (tp - 2.0 * t_bed + tm) * idz2
        r[rbase + t_idx] = (
            (1.0 - eps_b) * (rho_s * cp_s + cp_gas_local * q_sum * rho_s) * dtemp
            - tdiff
            + eps_b * cp_gas_local * dp / R_GAS
            + eps_b * cp_gas_local * (nu_right * p - nu_left * pm) * idz / R_GAS
            + (1.0 - eps_b) * cp_gas_local * t_bed * dq_sum * rho_s
            + (1.0 - eps_b) * adsorption_heat * rho_s
            + 2.0 * h_in / r_in * (t_bed - t_wall)
        )

        twdiff = kw * (twp - 2.0 * t_wall + twm) * idz2
        r[rbase + tw_idx] = (
            rho_w * cp_w * dtw
            - twdiff
            - 2.0 * r_in * h_in / r_sq_diff * (t_bed - t_wall)
            + 2.0 * r_out * h_out / r_sq_diff * (t_wall - t_amb)
        )

        r[rbase + p_idx] = (
            dp / p
            - dtemp / t_bed
            + (t_bed / p) * (nu_right * p / t_bed - nu_left * pm / tm) * idz
            + (R_GAS * t_bed / p) * eps_f * dq_sum * rho_s
        )

    return r


@numba.njit(cache=NUMBA_CACHE)
def build_block_tridiag_jacobian(
    u,
    u_old,
    dt_val,
    n_grid,
    n_components,
    n_sites,
    dz,
    eps_b,
    mu,
    dia_p,
    component_cp_gas,
    component_mw,
    rho_s,
    cp_s,
    t_in,
    t_amb,
    p_out,
    dl,
    dl0,
    y_in,
    ldf,
    q_method_id,
    iso_model_ids,
    iso_p1,
    iso_p2_ref,
    iso_p3,
    iso_dh,
    iso_t_ref,
    d_h_ads,
    kz,
    kw,
    rho_w,
    cp_w,
    r_in,
    r_out,
    r_sq_diff,
    h_in,
    h_out,
    nu_in,
    delta,
):
    nv = 2 * n_components + 3
    a = np.zeros((n_grid, nv, nv), dtype=np.float64)
    b = np.zeros((n_grid, nv, nv), dtype=np.float64)
    c = np.zeros((n_grid, nv, nv), dtype=np.float64)
    t_idx = 2 * n_components
    p_idx = t_idx + 2
    qe_cache = compute_qe_cache(u, n_grid, n_components, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref)

    r0 = residual_node_major_window(
        u, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s, cp_s, t_in,
        t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w, cp_w, r_in, r_out,
        r_sq_diff, h_in, h_out, nu_in, 0, n_grid, qe_cache, -1, False
    )

    for j in range(n_grid):
        base_j = j * nv
        for k in range(nv):
            idx = base_j + k
            saved = u[idx]
            step = delta * (1.0 + abs(saved))
            u[idx] = saved + step
            recompute_perturbed_qe = k < n_components or k == t_idx or k == p_idx

            row_start = j - 1
            if row_start < 0:
                row_start = 0
            row_stop = j + 2
            if row_stop > n_grid:
                row_stop = n_grid

            rp = residual_node_major_window(
                u, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s,
                cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w,
                cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in, row_start, row_stop, qe_cache, j, recompute_perturbed_qe
            )

            for i in range(row_start, row_stop):
                base_i = i * nv
                win_base = (i - row_start) * nv
                for row in range(nv):
                    val = (rp[win_base + row] - r0[base_i + row]) / step
                    if i == j:
                        b[i, row, k] = val
                    elif i == j - 1:
                        c[i, row, k] = val
                    else:
                        a[i, row, k] = val

            u[idx] = saved

    return a, b, c, r0


@numba.njit(cache=NUMBA_CACHE)
def solve_block_tridiag(a, b, c, rhs, n_grid):
    nv = b.shape[1]
    cp = np.zeros((n_grid, nv, nv), dtype=np.float64)
    dp = np.zeros((n_grid, nv), dtype=np.float64)

    if n_grid > 1:
        cp[0] = np.linalg.solve(b[0], c[0])
    dp[0] = np.linalg.solve(b[0], rhs[0])

    for i in range(1, n_grid):
        denom = b[i] - a[i] @ cp[i - 1]
        dp[i] = np.linalg.solve(denom, rhs[i] - a[i] @ dp[i - 1])
        if i < n_grid - 1:
            cp[i] = np.linalg.solve(denom, c[i])

    x = np.zeros((n_grid, nv), dtype=np.float64)
    x[n_grid - 1] = dp[n_grid - 1]
    for i in range(n_grid - 2, -1, -1):
        x[i] = dp[i] - cp[i] @ x[i + 1]

    return x


@numba.njit(cache=NUMBA_CACHE)
def scaled_residual_norm(r, u):
    total = 0.0
    for i in range(r.size):
        scale = abs(u[i])
        if scale < 1.0:
            scale = 1.0
        val = abs(r[i]) / scale
        total += val * val
    return np.sqrt(total / r.size)


@numba.njit(cache=NUMBA_CACHE)
def scaled_step_error(u_a, u_b):
    total = 0.0
    for i in range(u_a.size):
        scale = abs(u_a[i])
        other_scale = abs(u_b[i])
        if other_scale > scale:
            scale = other_scale
        if scale < 1.0:
            scale = 1.0
        val = abs(u_a[i] - u_b[i]) / scale
        total += val * val
    return np.sqrt(total / u_a.size)


@numba.njit(cache=NUMBA_CACHE)
def newton_block_tridiag(
    u0,
    u_old,
    dt_val,
    n_grid,
    n_components,
    n_sites,
    dz,
    eps_b,
    mu,
    dia_p,
    component_cp_gas,
    component_mw,
    rho_s,
    cp_s,
    t_in,
    t_amb,
    p_out,
    dl,
    dl0,
    y_in,
    ldf,
    q_method_id,
    iso_model_ids,
    iso_p1,
    iso_p2_ref,
    iso_p3,
    iso_dh,
    iso_t_ref,
    d_h_ads,
    kz,
    kw,
    rho_w,
    cp_w,
    r_in,
    r_out,
    r_sq_diff,
    h_in,
    h_out,
    nu_in,
    tol,
    max_iter,
    delta,
    alpha,
    min_alpha,
    line_search_max,
):
    nv = 2 * n_components + 3
    u = u0.copy()
    apply_inlet_guess(u, n_components, y_in, t_in)
    project_state(u, n_grid, n_components)
    final_norm = 0.0

    for iteration in range(max_iter):
        a, b, c, r = build_block_tridiag_jacobian(
            u, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s,
            cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w,
            cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in, delta
        )
        final_norm = scaled_residual_norm(r, u)
        if final_norm < tol:
            _, nu = residual_node_major(
                u, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw,
                rho_s, cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz,
                kw, rho_w, cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in
            )
            return u, nu, True, iteration + 1, final_norm

        rhs = np.zeros((n_grid, nv), dtype=np.float64)
        for i in range(n_grid):
            base = i * nv
            for row in range(nv):
                rhs[i, row] = -r[base + row]

        dx = solve_block_tridiag(a, b, c, rhs, n_grid)

        accepted = False
        best_norm = final_norm
        best_u = u.copy()
        fallback_found = False
        fallback_norm = final_norm
        fallback_u = u.copy()
        trial_alpha = alpha
        for _ls in range(line_search_max + 1):
            u_trial = u.copy()
            for i in range(n_grid):
                base = i * nv
                for row in range(nv):
                    u_trial[base + row] += trial_alpha * dx[i, row]
            project_state(u_trial, n_grid, n_components)

            r_trial, _ = residual_node_major(
                u_trial, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas,
                component_mw, rho_s, cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh,
                iso_t_ref, d_h_ads, kz, kw, rho_w, cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in
            )
            trial_norm = scaled_residual_norm(r_trial, u_trial)
            if np.isfinite(trial_norm):
                if not fallback_found:
                    fallback_found = True
                    fallback_norm = trial_norm
                    fallback_u = u_trial
                if trial_norm < best_norm:
                    best_norm = trial_norm
                    best_u = u_trial
                if trial_norm <= final_norm * (1.0 - 1e-4 * trial_alpha):
                    u = u_trial
                    accepted = True
                    break

            trial_alpha *= 0.5
            if trial_alpha < min_alpha:
                break

        if not accepted:
            if best_norm < final_norm:
                u = best_u
                final_norm = best_norm
            elif fallback_found:
                u = fallback_u
                final_norm = fallback_norm
            else:
                return u, np.full(n_grid, nu_in, dtype=np.float64), False, iteration + 1, final_norm

    _, nu = residual_node_major(
        u, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s,
        cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w, cp_w,
        r_in, r_out, r_sq_diff, h_in, h_out, nu_in
    )
    return u, nu, False, max_iter, final_norm


@numba.njit(cache=NUMBA_CACHE)
def newton_block_tridiag_with_retry(
    u0,
    u_old,
    dt_val,
    n_grid,
    n_components,
    n_sites,
    dz,
    eps_b,
    mu,
    dia_p,
    component_cp_gas,
    component_mw,
    rho_s,
    cp_s,
    t_in,
    t_amb,
    p_out,
    dl,
    dl0,
    y_in,
    ldf,
    q_method_id,
    iso_model_ids,
    iso_p1,
    iso_p2_ref,
    iso_p3,
    iso_dh,
    iso_t_ref,
    d_h_ads,
    kz,
    kw,
    rho_w,
    cp_w,
    r_in,
    r_out,
    r_sq_diff,
    h_in,
    h_out,
    nu_in,
    tol,
    max_iter,
    retry_max_iter,
    delta,
    alpha,
    min_alpha,
    line_search_max,
    retry_min_alpha,
    retry_line_search_max,
):
    u, nu, ok, iterations, norm = newton_block_tridiag(
        u0, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s,
        cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w,
        cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in, tol, max_iter, delta, alpha, min_alpha,
        line_search_max
    )
    if ok:
        return u, nu, True, iterations, norm, False

    if retry_max_iter <= max_iter and retry_line_search_max <= line_search_max and retry_min_alpha >= min_alpha:
        return u, nu, False, iterations, norm, False

    u_retry, nu_retry, ok_retry, retry_iterations, retry_norm = newton_block_tridiag(
        u0, u_old, dt_val, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s,
        cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w,
        cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in, tol, retry_max_iter, delta, alpha,
        retry_min_alpha, retry_line_search_max
    )
    return u_retry, nu_retry, ok_retry, retry_iterations, retry_norm, True


@numba.njit(cache=NUMBA_CACHE)
def adaptive_simulation_block(
    t_end,
    dt0,
    dt_min,
    dt_max,
    tol_err,
    max_steps,
    u_init,
    n_grid,
    n_components,
    n_sites,
    dz,
    eps_b,
    mu,
    dia_p,
    component_cp_gas,
    component_mw,
    rho_s,
    cp_s,
    t_in,
    t_amb,
    p_out,
    dl,
    dl0,
    y_in,
    ldf,
    q_method_id,
    iso_model_ids,
    iso_p1,
    iso_p2_ref,
    iso_p3,
    iso_dh,
    iso_t_ref,
    d_h_ads,
    kz,
    kw,
    rho_w,
    cp_w,
    r_in,
    r_out,
    r_sq_diff,
    h_in,
    h_out,
    nu_in,
    newton_tol,
    newton_max_iter,
    newton_retry_max_iter,
    newton_delta,
    newton_alpha,
    newton_min_alpha,
    newton_line_search_max,
    startup_dt_min_config,
    newton_retry_min_alpha,
    newton_retry_line_search_max,
    max_rejected_steps,
    report_times,
    stop_signal_path,
):
    times = np.empty(max_steps, dtype=np.float64)
    results = np.empty((max_steps, u_init.size + n_grid), dtype=np.float64)
    converged = np.zeros(max_steps, dtype=np.bool_)
    newton_iterations = np.zeros(max_steps, dtype=np.int64)
    newton_norms = np.zeros(max_steps, dtype=np.float64)
    step_subdivisions = np.ones(max_steps, dtype=np.int64)
    accepted_dt = np.zeros(max_steps, dtype=np.float64)

    u = u_init.copy()
    k = 0
    times[k] = 0.0
    for i in range(u.size):
        results[k, i] = u[i]
    nu0 = compute_face_velocities(u, n_grid, n_components, dz, eps_b, mu, dia_p, component_mw, nu_in)
    for i in range(n_grid):
        # Store face-based interstitial velocity; superficial velocity is eps_b * u_interstitial.
        results[k, u.size + i] = nu0[i]
    converged[k] = True

    t = 0.0
    dt = dt0
    rejected_steps = 0
    status_code = 0
    fail_stage = 0
    fail_newton_iterations = 0
    fail_newton_norm = 0.0
    fail_dt = 0.0
    fail_time = 0.0
    retry_uses = 0
    next_report = 0
    report_header_printed = False
    time_tol = 1e-12
    if t_end > 1.0:
        time_tol = 1e-12 * t_end
    startup_dt_min = dt_min
    if newton_max_iter <= 10:
        startup_dt_min = startup_dt_min_config
        if startup_dt_min > dt0:
            startup_dt_min = dt0
        if startup_dt_min > dt_min:
            startup_dt_min = dt_min
        if startup_dt_min < time_tol:
            startup_dt_min = time_tol
    prev_err = -1.0

    while t < t_end - time_tol and k < max_steps - 1:
        if t + dt > t_end:
            dt = t_end - t
        if dt <= time_tol:
            t = t_end
            break
        active_dt_min = dt_min
        if k == 0 and startup_dt_min < dt_min:
            active_dt_min = startup_dt_min

        u_old = u.copy()
        u_full, _, ok_full, iter_full, norm_full, retried_full = newton_block_tridiag_with_retry(
            u, u_old, dt, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s,
            cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w,
            cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in, newton_tol, newton_max_iter,
            newton_retry_max_iter, newton_delta, newton_alpha, newton_min_alpha, newton_line_search_max,
            newton_retry_min_alpha, newton_retry_line_search_max
        )
        if retried_full:
            retry_uses += 1
        if not ok_full:
            rejected_steps += 1
            if rejected_steps >= max_rejected_steps:
                status_code = 6
                fail_stage = 1
                fail_newton_iterations = iter_full
                fail_newton_norm = norm_full
                fail_dt = dt
                fail_time = t
                break
            if dt <= active_dt_min:
                status_code = 2
                fail_stage = 1
                fail_newton_iterations = iter_full
                fail_newton_norm = norm_full
                fail_dt = dt
                fail_time = t
                break
            dt *= 0.5
            if dt < active_dt_min:
                dt = active_dt_min
            continue

        u_h1, _, ok_h1, iter_h1, norm_h1, retried_h1 = newton_block_tridiag_with_retry(
            u, u_old, 0.5 * dt, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas, component_mw, rho_s,
            cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh, iso_t_ref, d_h_ads, kz, kw, rho_w,
            cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in, newton_tol, newton_max_iter,
            newton_retry_max_iter, newton_delta, newton_alpha, newton_min_alpha, newton_line_search_max,
            newton_retry_min_alpha, newton_retry_line_search_max
        )
        if retried_h1:
            retry_uses += 1
        if not ok_h1:
            rejected_steps += 1
            if rejected_steps >= max_rejected_steps:
                status_code = 6
                fail_stage = 2
                fail_newton_iterations = iter_h1
                fail_newton_norm = norm_h1
                fail_dt = 0.5 * dt
                fail_time = t
                break
            if dt <= active_dt_min:
                status_code = 3
                fail_stage = 2
                fail_newton_iterations = iter_h1
                fail_newton_norm = norm_h1
                fail_dt = 0.5 * dt
                fail_time = t
                break
            dt *= 0.5
            if dt < active_dt_min:
                dt = active_dt_min
            continue

        u_h2, nu_h, ok_h2, iter_h2, norm_h2, retried_h2 = newton_block_tridiag_with_retry(
            u_full, u_h1, 0.5 * dt, n_grid, n_components, n_sites, dz, eps_b, mu, dia_p, component_cp_gas,
            component_mw, rho_s, cp_s, t_in, t_amb, p_out, dl, dl0, y_in, ldf, q_method_id, iso_model_ids, iso_p1, iso_p2_ref, iso_p3, iso_dh,
            iso_t_ref, d_h_ads, kz, kw, rho_w, cp_w, r_in, r_out, r_sq_diff, h_in, h_out, nu_in, newton_tol,
            newton_max_iter, newton_retry_max_iter, newton_delta, newton_alpha, newton_min_alpha,
            newton_line_search_max, newton_retry_min_alpha, newton_retry_line_search_max
        )
        if retried_h2:
            retry_uses += 1
        if not ok_h2:
            rejected_steps += 1
            if rejected_steps >= max_rejected_steps:
                status_code = 6
                fail_stage = 3
                fail_newton_iterations = iter_h2
                fail_newton_norm = norm_h2
                fail_dt = 0.5 * dt
                fail_time = t
                break
            if dt <= active_dt_min:
                status_code = 4
                fail_stage = 3
                fail_newton_iterations = iter_h2
                fail_newton_norm = norm_h2
                fail_dt = 0.5 * dt
                fail_time = t
                break
            dt *= 0.5
            if dt < active_dt_min:
                dt = active_dt_min
            continue

        err = scaled_step_error(u_h2, u_full)

        if err < tol_err:
            t += dt
            u = u_h2
            k += 1
            times[k] = t
            accepted_dt[k] = dt
            for i in range(u.size):
                results[k, i] = u[i]
            for i in range(n_grid):
                # Store face-based interstitial velocity; superficial velocity is eps_b * u_interstitial.
                results[k, u.size + i] = nu_h[i]
            converged[k] = True
            newton_iterations[k] = iter_h2
            newton_norms[k] = norm_h2
            step_subdivisions[k] = 2

            while next_report < report_times.size and t >= report_times[next_report]:
                with numba.objmode():
                    if not report_header_printed:
                        print("fib_report dt_s", flush=True)
                    print(report_times[next_report], dt, flush=True)
                report_header_printed = True
                next_report += 1

            stop_requested = False
            with numba.objmode(stop_requested="boolean"):
                stop_requested = bool(stop_signal_path and os.path.exists(stop_signal_path))
            if stop_requested:
                status_code = 7
                break

            if prev_err > 0.0:
                factor = 0.9 * (tol_err / (err + 1e-15)) ** 0.35 * (prev_err / (err + 1e-15)) ** 0.15
            else:
                factor = 0.9 * np.sqrt(tol_err / (err + 1e-15))
            if factor < 0.5:
                factor = 0.5
            elif factor > 2.0:
                factor = 2.0
            dt *= factor
            if dt > dt_max:
                dt = dt_max
            if dt < active_dt_min:
                dt = active_dt_min
            prev_err = err
        else:
            rejected_steps += 1
            if rejected_steps >= max_rejected_steps:
                status_code = 6
                fail_stage = 4
                fail_newton_iterations = iter_h2
                fail_newton_norm = norm_h2
                fail_dt = dt
                fail_time = t
                break
            if dt <= active_dt_min:
                status_code = 5
                fail_stage = 4
                fail_newton_iterations = iter_h2
                fail_newton_norm = norm_h2
                fail_dt = dt
                fail_time = t
                break
            factor = 0.9 * np.sqrt(tol_err / (err + 1e-15))
            if factor < 0.5:
                factor = 0.5
            elif factor > 2.0:
                factor = 2.0
            dt *= factor
            if dt < active_dt_min:
                dt = active_dt_min
            prev_err = err

    if status_code == 0 and t < t_end - time_tol:
        status_code = 1

    return (
        times,
        results,
        converged,
        newton_iterations,
        newton_norms,
        step_subdivisions,
        accepted_dt,
        k + 1,
        status_code,
        rejected_steps,
        t,
        dt,
        fail_stage,
        fail_time,
        fail_dt,
        fail_newton_iterations,
        fail_newton_norm,
        retry_uses,
    )


def _compute_u_superficial_inputs(model):
    area = np.pi * model["dia_in"] ** 2 / 4.0
    inlet_mw = float(np.dot(model["y_in"], model["component_mw"]) / np.sum(model["y_in"]))
    u_superficial_inlet_basis = model["flow_in"] * 1e-6 / 60.0 / area * (model["t_in"] / model["t_amb"]) * (model["p_amb"] / model["p_out"])
    d_pdl = (
        150.0 * model["mu"] * (1.0 - model["eps_b"]) ** 2 / model["eps_b"] ** 2 * (u_superficial_inlet_basis / model["eps_b"]) / model["dia_p"] ** 2
        + 1.75 * (1.0 - model["eps_b"]) * rho_g(model["p_out"], model["t0"], inlet_mw) * (u_superficial_inlet_basis / model["eps_b"]) ** 2 / model["eps_b"] / model["dia_p"]
    )
    u_superficial = model["flow_in"] * 1e-6 / 60.0 / area * (model["t_in"] / model["t_amb"]) * (model["p_amb"] / (model["p_out"] + d_pdl * model["length"]))
    for _ in range(2):
        d_pdl = (
            150.0 * model["mu"] * (1.0 - model["eps_b"]) ** 2 / model["eps_b"] ** 2 * (0.5 * (u_superficial_inlet_basis + u_superficial) / model["eps_b"]) / model["dia_p"] ** 2
            + 1.75
            * (1.0 - model["eps_b"])
            * rho_g(0.5 * (model["p_out"] + (model["p_out"] + d_pdl * model["length"])), model["t0"], inlet_mw)
            * (0.5 * (u_superficial_inlet_basis + u_superficial) / model["eps_b"]) ** 2
            / model["eps_b"]
            / model["dia_p"]
        )
        u_superficial = model["flow_in"] * 1e-6 / 60.0 / area * (model["t_in"] / model["t_amb"]) * (model["p_amb"] / (model["p_out"] + d_pdl * model["length"]))
    return u_superficial_inlet_basis, u_superficial, d_pdl


def _compute_axial_dispersion(model, u_interstitial):
    rp = 0.5 * model["dia_p"]
    if model["dl_model"] == "custom":
        return model["dl_custom"]
    if model["dl_model"] == "gunn":
        return 0.7 * model["dm"] + u_interstitial * rp
    return 20.0 * model["dm"] / model["eps_b"] + u_interstitial * rp


def run_simulation(inputs, report_times=None, stop_signal_path=""):
    model = _build_model_inputs(inputs)
    start = tm.time()

    n_grid = model["n_grid"]
    n_components = model["n_components"]
    n_sites = model["n_sites"]
    nv = model["state_size"]
    t_idx = 2 * n_components
    tw_idx = t_idx + 1
    p_idx = t_idx + 2

    dz = model["length"] / (n_grid - 1)
    z_axis = np.linspace(0.0, model["length"], n_grid)
    r_in = model["dia_in"] / 2.0
    r_out = r_in + model["w_w"]
    r_sq_diff = r_out * r_out - r_in * r_in

    u_superficial_inlet_basis, u_superficial, d_pdl = _compute_u_superficial_inputs(model)
    nu_in = u_superficial / model["eps_b"]
    dl = _compute_axial_dispersion(model, nu_in)
    dl0 = dl if model["dl0_custom"] is None else model["dl0_custom"]

    y_init = np.empty((n_components, n_grid), dtype=np.float64)
    for comp in range(n_components):
        y_init[comp, :] = model["initial_y"][comp]

    initial_y_eq = np.maximum(model["initial_y"], Y_FLOOR)
    initial_y_sum = np.sum(initial_y_eq)
    if initial_y_sum <= 0.0 or not np.isfinite(initial_y_sum):
        initial_y_eq = np.maximum(model["y_in"], Y_FLOOR)
        initial_y_sum = np.sum(initial_y_eq)
    initial_y_eq = initial_y_eq / initial_y_sum
    qe_init = q_eq(
        initial_y_eq,
        model["t0"],
        model["p_out"],
        model["q_method_id"],
        model["iso_model_ids"],
        model["iso_p1"],
        model["iso_p2_ref"],
        model["iso_p3"],
        model["iso_dh"],
        model["iso_t_ref"],
        n_components,
    )
    q_init = np.empty((n_components, n_grid), dtype=np.float64)
    for comp in range(n_components):
        q_init[comp, :] = qe_init[comp]

    t_bed = np.full(n_grid, model["t0"], dtype=np.float64)
    t_wall = np.full(n_grid, model["t0"], dtype=np.float64)
    # Start Newton from a physically consistent Ergun-like pressure profile:
    # inlet pressure is higher, outlet pressure equals p_out.
    pressure = model["p_out"] + d_pdl * (model["length"] - z_axis)
    pressure = pressure.astype(np.float64)

    u_init = np.empty(n_grid * nv, dtype=np.float64)
    for i in range(n_grid):
        base = i * nv
        for comp in range(n_components):
            u_init[base + comp] = y_init[comp, i]
            u_init[base + n_components + comp] = q_init[comp, i]
        u_init[base + t_idx] = t_bed[i]
        u_init[base + tw_idx] = t_wall[i]
        u_init[base + p_idx] = pressure[i]

    project_state(u_init, n_grid, n_components)

    if report_times is None:
        report_times_array = np.empty(0, dtype=np.float64)
    else:
        report_times_array = np.array(report_times, dtype=np.float64)
        report_times_array = report_times_array[np.isfinite(report_times_array)]
        report_times_array = report_times_array[(report_times_array >= 0.0) & (report_times_array <= model["t_end"])]
        report_times_array.sort()

    (
        times_buf,
        results_nm,
        converged,
        newton_iterations,
        newton_norms,
        step_subdivisions,
        accepted_dt,
        nrows,
        status_code,
        rejected_steps,
        final_time,
        final_dt,
        fail_stage,
        fail_time,
        fail_dt,
        fail_newton_iterations,
        fail_newton_norm,
        retry_uses,
    ) = adaptive_simulation_block(
        model["t_end"],
        model["dt"],
        model["dt_min"],
        model["dt_max"],
        model["tol_err"],
        model["max_steps"],
        u_init,
        n_grid,
        n_components,
        n_sites,
        dz,
        model["eps_b"],
        model["mu"],
        model["dia_p"],
        model["component_cp_gas"],
        model["component_mw"],
        model["rho_s"],
        model["cp_s"],
        model["t_in"],
        model["t_amb"],
        model["p_out"],
        dl,
        dl0,
        model["y_in"],
        model["ldf"],
        model["q_method_id"],
        model["iso_model_ids"],
        model["iso_p1"],
        model["iso_p2_ref"],
        model["iso_p3"],
        model["iso_dh"],
        model["iso_t_ref"],
        model["d_h_ads"],
        model["kz"],
        model["kw"],
        model["rho_w"],
        model["cp_w"],
        r_in,
        r_out,
        r_sq_diff,
        model["h_in"],
        model["h_out"],
        nu_in,
        model["newton_tol"],
        model["newton_max_iter"],
        model["newton_retry_max_iter"],
        model["newton_delta"],
        model["newton_alpha"],
        model["newton_min_alpha"],
        model["newton_line_search_max"],
        model["startup_dt_min"],
        model["newton_retry_min_alpha"],
        model["newton_retry_line_search_max"],
        model["max_rejected_steps"],
        report_times_array,
        stop_signal_path or "",
    )

    time_array = times_buf[:nrows].copy()
    results_nm = results_nm[:nrows, :].copy()
    converged = converged[:nrows].copy()
    newton_iterations = newton_iterations[:nrows].copy()
    newton_norms = newton_norms[:nrows].copy()
    step_subdivisions = step_subdivisions[:nrows].copy()
    accepted_dt = accepted_dt[:nrows].copy()

    block_count = 2 * n_components + 5
    results = np.zeros((nrows, block_count * n_grid), dtype=np.float64)
    u_size = n_grid * nv
    face_axis = np.empty(n_grid, dtype=np.float64)
    face_axis[0] = 0.0
    for i in range(1, n_grid):
        face_axis[i] = z_axis[i] - 0.5 * dz
    for s in range(nrows):
        u = results_nm[s, :u_size]
        u_interstitial_face_block = results_nm[s, u_size:u_size + n_grid]
        for i in range(n_grid):
            base = i * nv
            for comp in range(n_components):
                results[s, comp * n_grid + i] = u[base + comp]
                results[s, (n_components + comp) * n_grid + i] = u[base + n_components + comp]
            results[s, (2 * n_components + 0) * n_grid + i] = u[base + t_idx]
            results[s, (2 * n_components + 1) * n_grid + i] = u[base + tw_idx]
            results[s, (2 * n_components + 2) * n_grid + i] = u[base + p_idx]
            results[s, (2 * n_components + 3) * n_grid + i] = u_interstitial_face_block[i] * model["eps_b"]
            results[s, (2 * n_components + 4) * n_grid + i] = u_interstitial_face_block[i]

    sh_dict = {}
    columns = z_axis / model["length"]
    names = []
    for comp in range(n_components):
        names.append(f"y_{comp + 1}")
    for comp in range(n_components):
        names.append(f"q_{comp + 1}")
    names.extend(["Tbed", "Tw", "P", "u_superficial_face", "u_interstitial_face"])
    for i, name in enumerate(names):
        sh_dict[name] = results[:, i * n_grid:(i + 1) * n_grid].copy()
    sh_dict["u_superficial"] = sh_dict["u_superficial_face"]
    sh_dict["u_interstitial"] = sh_dict["u_interstitial_face"]

    elapsed = tm.time() - start
    mass = np.pi * model["dia_in"] * model["dia_in"] * model["length"] / 4.0 * model["rho_s"] * (1.0 - model["eps_b"])
    cc_min = u_superficial * np.pi * model["dia_in"] * model["dia_in"] / 4.0 * 1e6 * 60.0
    status = _simulation_status(status_code, final_time, model["t_end"], final_dt, rejected_steps, nrows, model["max_steps"])
    iast_failure_fallback_rows, iast_failure_fallback_cells = count_iast_failure_fallback_rows(
        results_nm,
        nrows,
        n_grid,
        n_components,
        model["q_method_id"],
        model["iso_model_ids"],
        model["iso_p1"],
        model["iso_p2_ref"],
        model["iso_p3"],
        model["iso_dh"],
        model["iso_t_ref"],
    )
    iast_warning = ""
    if nrows > 0:
        rejected_ratio_pct = 100.0 * rejected_steps / nrows
        if iast_failure_fallback_rows > 2 and rejected_ratio_pct > 30.0:
            iast_warning = (
                f"Rejected ratio = {rejected_ratio_pct:.3g}%\n"
                "IAST warning: some rows did not converge; check isotherm parameters, or trace pressures."
            )
    status["iast_failure_fallback_rows"] = int(iast_failure_fallback_rows)
    status["iast_failure_fallback_cells"] = int(iast_failure_fallback_cells)
    status["iast_warning"] = iast_warning
    status["diagnostics"] = _final_state_diagnostics(results_nm, nrows, n_grid, n_components)
    status["diagnostics"].update(
        {
            "fail_stage": int(fail_stage),
            "fail_time": float(fail_time),
            "fail_dt": float(fail_dt),
            "fail_newton_iterations": int(fail_newton_iterations),
            "fail_newton_norm": float(fail_newton_norm),
            "newton_retry_uses": int(retry_uses),
        }
    )

    output = {
        "inputs": inputs,
        "model": model,
        "time": time_array,
        "results": results,
        "sheets": sh_dict,
        "z_axis": z_axis,
        "columns": columns,
        "face_z_axis": face_axis,
        "face_columns": face_axis / model["length"],
        "u_superficial": u_superficial,
        "u_superficial_inlet_basis": u_superficial_inlet_basis,
        "DL": dl,
        "DL0": dl0,
        # Backward-compatible aliases; prefer u_superficial names in new code.
        "vel": u_superficial,
        "vel0": u_superficial_inlet_basis,
        "d_pdl": d_pdl,
        "mass": mass,
        "cc_min": cc_min,
        "elapsed_seconds": elapsed,
        "converged": converged,
        "newton_iterations": newton_iterations,
        "newton_norms": newton_norms,
        "step_subdivisions": step_subdivisions,
        "accepted_dt": accepted_dt,
        "status": status,
    }
    if model["raise_on_incomplete"] and not status["completed"]:
        raise RuntimeError(status["message"])
    return output


def _final_state_diagnostics(results_nm, nrows, n_grid, n_components):
    nv = 2 * n_components + 3
    t_idx = 2 * n_components
    tw_idx = t_idx + 1
    p_idx = t_idx + 2
    if nrows <= 0:
        return {}
    u = results_nm[nrows - 1, : n_grid * nv]
    y_vals = []
    q_vals = []
    t_vals = []
    tw_vals = []
    p_vals = []
    for i in range(n_grid):
        base = i * nv
        for comp in range(n_components):
            y_vals.append(u[base + comp])
            q_vals.append(u[base + n_components + comp])
        t_vals.append(u[base + t_idx])
        tw_vals.append(u[base + tw_idx])
        p_vals.append(u[base + p_idx])
    y_arr = np.asarray(y_vals)
    q_arr = np.asarray(q_vals)
    t_arr = np.asarray(t_vals)
    tw_arr = np.asarray(tw_vals)
    p_arr = np.asarray(p_vals)
    return {
        "y_min": float(np.nanmin(y_arr)),
        "y_max": float(np.nanmax(y_arr)),
        "q_min": float(np.nanmin(q_arr)),
        "q_max": float(np.nanmax(q_arr)),
        "t_bed_min": float(np.nanmin(t_arr)),
        "t_bed_max": float(np.nanmax(t_arr)),
        "t_wall_min": float(np.nanmin(tw_arr)),
        "t_wall_max": float(np.nanmax(tw_arr)),
        "p_min": float(np.nanmin(p_arr)),
        "p_max": float(np.nanmax(p_arr)),
    }


def _simulation_status(status_code, final_time, t_end, final_dt, rejected_steps, nrows, max_steps):
    messages = {
        0: "Simulation completed.",
        1: "Simulation stopped before t_end because max_steps was reached.",
        2: "Simulation stopped because the full Newton step failed at dt_min.",
        3: "Simulation stopped because the first half Newton step failed at dt_min.",
        4: "Simulation stopped because the second half Newton step failed at dt_min.",
        5: "Simulation stopped because the adaptive error remained above tolerance at dt_min.",
        6: "Simulation stopped because max_rejected_steps was reached.",
        7: "Simulation stopped by user after the last accepted time step.",
    }
    completed = status_code == 0
    max_steps_exceeded = status_code == 1
    return {
        "code": int(status_code),
        "completed": completed,
        "max_steps_exceeded": max_steps_exceeded,
        "message": messages.get(int(status_code), "Simulation stopped with an unknown status."),
        "final_time": float(final_time),
        "target_time": float(t_end),
        "final_dt": float(final_dt),
        "rejected_steps": int(rejected_steps),
        "accepted_rows": int(nrows),
        "max_steps": int(max_steps),
    }
