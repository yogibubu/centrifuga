#!/usr/bin/env python3
"""Utility workflows exposing a minimal subset of order-2 quartic utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import sympy as sp

from compare_gaussian_sextic import _degenerate_mode_metadata
from quartic_channels import channel_h12h12, tau_to_watson_a
from symmetry_metadata import assign_normal_mode_irreps

EH_TO_MHZ = 6.57968392061e9
AU_FREQ_TO_CMINV = 219474.6313705
CMINV_TO_HZ = 2.99792458e10
CMINV_TO_MHZ = CMINV_TO_HZ / 1.0e6


def _watson_dict_to_float(watson: dict[str, sp.Expr]) -> dict[str, float]:
    return {key: float(EH_TO_MHZ * sp.N(value)) for key, value in watson.items()}


def _compressed_tau_to_float(tau: dict[str, sp.Expr]) -> dict[str, float]:
    return {key: float(EH_TO_MHZ * sp.N(value)) for key, value in tau.items()}


def compute_order2_quartic(model) -> dict[str, object]:
    """Return order-2 quartic data derived from a harmonic inertia model."""
    omega = [abs(f) / AU_FREQ_TO_CMINV for f in np.asarray(model.vib_freq_cm, dtype=float)]
    mu1 = sp.MutableDenseNDimArray(np.asarray(model.dInv_au, dtype=float))
    tau = channel_h12h12(mu1, omega)
    tau_mhz = _compressed_tau_to_float(tau)
    watson_a = _watson_dict_to_float(tau_to_watson_a(tau))
    # In the A/S reductions the numerical values coincide at this level.
    rotor_limit = classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    special_quartic = project_special_quartic_constants(
        tau_mhz,
        np.asarray(model.abc_mhz, dtype=float),
        np.asarray(model.moments_amu_a2, dtype=float),
    )
    result = {
        "rotor_limit": rotor_limit,
        "tau_mhz": tau_mhz,
        "watson_a_mhz": watson_a,
        "watson_s_mhz": watson_a.copy(),
        "special_quartic_projection": special_quartic,
        "linear_ltype_terms": linear_ltype_terms(model, quartic_special=special_quartic),
    }
    return result


def classify_rotor_limit(abc: Iterable[float], moments: Iterable[float]) -> dict[str, object]:
    """Classify rotor limits for symmetry-adapted projections."""
    abc = np.asarray(abc, dtype=float)
    moments = np.asarray(moments, dtype=float).reshape(3)
    finite = abc[np.isfinite(abc)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0

    def _close(x: float, y: float) -> bool:
        return abs(x - y) <= max(1.0e-3, 1.0e-6 * max(abs(x), abs(y), scale, 1.0))

    if np.all(np.isfinite(moments)) and abs(moments[0]) <= max(1.0e-10, 1.0e-6 * max(abs(moments[2]), 1.0)):
        return {
            "kind": "linear",
            "symmetry_axis": "a",
            "degenerate_axes": ("b", "c"),
            "is_special_limit": True,
        }
    if np.all(np.isfinite(abc)) and _close(abc[0], abc[1]) and _close(abc[1], abc[2]):
        return {
            "kind": "spherical_top",
            "symmetry_axis": None,
            "degenerate_axes": ("a", "b", "c"),
            "is_special_limit": True,
        }
    if (not np.isfinite(abc[0])) or (_close(abc[1], abc[2]) and abc[0] > 100.0 * max(abs(abc[1]), 1.0)):
        return {
            "kind": "linear",
            "symmetry_axis": "a",
            "degenerate_axes": ("b", "c"),
            "is_special_limit": True,
        }
    if _close(abc[1], abc[2]):
        return {
            "kind": "symmetric_top_prolate",
            "symmetry_axis": "a",
            "degenerate_axes": ("b", "c"),
            "is_special_limit": True,
        }
    if _close(abc[0], abc[1]):
        return {
            "kind": "symmetric_top_oblate",
            "symmetry_axis": "c",
            "degenerate_axes": ("a", "b"),
            "is_special_limit": True,
        }
    return {
        "kind": "asymmetric_top",
        "symmetry_axis": None,
        "degenerate_axes": (),
        "is_special_limit": False,
    }


def linear_ltype_terms(model, *, quartic_special=None, sextic_special=None):
    """Return a minimal pairwise l-type effective model for linear molecules.

    The present implementation adopts a circular-doublet basis as the
    primary convention for each near-degenerate bending pair ``(i,j)``
    and also reports the equivalent real-doublet form. In the primary
    circular basis ``|+>, |->`` the minimal active channel is

    - ``X_l``,
    - ``J^2 X_l``,
    - ``(J^2)^2 X_l``,

    where the rotationally resolved coefficients are inferred from the
    linear-limit scalar constants ``D`` and ``H`` and from the dominant
    parallel Coriolis coupling of the pair. In the equivalent real
    basis built from the same doublet, the same splitting is reported as
    ``O_t = |u><u| - |v><v|``. This remains a minimal pairwise model, not yet the
    full effective Hamiltonian for all linear-molecule l-type
    interactions.
    """
    rotor_limit = classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    if rotor_limit["kind"] != "linear":
        return None

    pair_meta = _degenerate_mode_metadata(model, rotor_limit)
    if not pair_meta:
        return None
    mode_irreps = assign_normal_mode_irreps(model)

    b_candidates = []
    for val in np.asarray(model.abc_mhz, dtype=float).reshape(3):
        if np.isfinite(val) and val > 1.0e-9:
            b_candidates.append(float(val / CMINV_TO_MHZ))
    b_linear_cm = float(np.mean(b_candidates)) if b_candidates else None

    d_hz = None
    if quartic_special is not None:
        qmap = quartic_special.get("quartic_mhz", {})
        if "D" in qmap:
            d_hz = float(qmap["D"]) * 1.0e6
    h_hz = None
    if sextic_special is not None:
        smap = sextic_special.get("sextic_hz", {})
        if "H" in smap:
            h_hz = float(smap["H"])

    full_basis_real = [
        {"operator": "I_t", "matrix": [[1.0, 0.0], [0.0, 1.0]]},
        {"operator": "O_t", "matrix": [[1.0, 0.0], [0.0, -1.0]]},
        {"operator": "X_t", "matrix": [[0.0, 1.0], [1.0, 0.0]]},
        {"operator": "Y_t", "matrix": [[0.0, -1.0], [1.0, 0.0]]},
    ]
    full_basis_circular = [
        {"operator": "I_l", "matrix": [[1.0, 0.0], [0.0, 1.0]]},
        {"operator": "Z_l", "matrix": [[1.0, 0.0], [0.0, -1.0]]},
        {"operator": "X_l", "matrix": [[0.0, 1.0], [1.0, 0.0]]},
        {"operator": "Y_l", "matrix": [[0.0, -1.0], [1.0, 0.0]]},
    ]

    pairs: list[dict[str, object]] = []
    irrep_counts: dict[str, int] = {}
    for meta in pair_meta:
        zeta_parallel = float(meta.get("dominant_coriolis_abs", 0.0))
        freq_cm = float(meta["freq_cm"])
        q_tj = None if d_hz is None else abs(zeta_parallel) * abs(d_hz)
        q_th = None if h_hz is None else abs(zeta_parallel) * abs(h_hz)
        q_l_leading_hz = 0.0
        q_l_watson_hz = 0.0
        watson_terms: list[dict[str, float | int]] = []
        if b_linear_cm is not None and abs(freq_cm) > 1.0e-12:
            q_l_leading_hz = float((2.0 * abs(b_linear_cm) / abs(freq_cm)) * CMINV_TO_HZ)
            freq_all = np.asarray(model.vib_freq_cm, dtype=float).reshape(-1)
            zeta_all = np.asarray(model.coriolis_zeta_pairs_xyz, dtype=float)
            corr = 0.0
            for s in range(freq_all.size):
                if s in meta["pair"]:
                    continue
                fst = float(np.max([abs(zeta_all[ax, i, s]) for ax in range(3) for i in meta["pair"]]))
                denom = float(freq_all[s] ** 2 - freq_cm**2)
                term = 0.0 if abs(denom) <= 1.0e-12 else float((fst * fst) / denom)
                corr += term
                watson_terms.append({"mode": int(s), "freq_cm": float(freq_all[s]), "f_st": fst, "term": term})
            q_l_watson_hz = float(q_l_leading_hz * (1.0 + corr))
        q_tabs = 0.0
        if q_tj is not None:
            q_tabs += q_tj * q_tj
        if q_th is not None:
            q_tabs += q_th * q_th
        q_tabs = float(np.sqrt(q_tabs))
        if q_tabs <= 0.0:
            q_tabs = abs(zeta_parallel) * (abs(b_linear_cm) * CMINV_TO_HZ if b_linear_cm is not None else 0.0)
        item = {
            "modes": tuple(int(x) for x in meta["pair"]),
            "freq_cm": freq_cm,
            "zeta_parallel": zeta_parallel,
            "q_t_abs_hz": q_tabs,
            "operator_basis": "circular_doublet",
            "operator_label": "X_l",
            "operator_basis_equiv": "real_doublet",
            "operator_label_equiv": "O_t",
            "full_operator_basis_real": full_basis_real,
            "full_operator_basis_alt": full_basis_circular,
        }
        if mode_irreps is not None:
            i_mode, j_mode = int(meta["pair"][0]), int(meta["pair"][1])
            ir_i = mode_irreps[i_mode]
            ir_j = mode_irreps[j_mode]
            item["mode_irreps"] = (ir_i, ir_j)
            if ir_i == ir_j:
                item["pair_irrep"] = ir_i
                irrep_counts[ir_i] = irrep_counts.get(ir_i, 0) + 1
                item["pair_index_within_irrep"] = irrep_counts[ir_i]
                item["pair_label"] = f"{ir_i}({irrep_counts[ir_i]})"
        if q_tj is not None:
            item["q_tJ_diagnostic_hz"] = float(q_tj)
        if q_th is not None:
            item["q_tH_diagnostic_hz"] = float(q_th)
        q_tj_val = float(q_tj) if q_tj is not None else 0.0
        q_th_val = float(q_th) if q_th is not None else 0.0
        item["conventional_constants_hz"] = {
            "q_t": float(q_tabs),
            "q_tJ": q_tj_val,
            "q_tH": q_th_val,
        }
        item["literature_constants_hz"] = {
            "q_l": float(q_tabs),
            "q_lJ": q_tj_val,
            "q_lH": q_th_val,
        }
        item["literature_harmonic_estimate_hz"] = {
            "q_l_leading": q_l_leading_hz,
            "formula": "q_e^(0) = 2 B_linear / omega_t",
        }
        item["literature_watson_estimate_hz"] = {
            "q_l_watson": q_l_watson_hz,
            "formula": "q_e = (2 B_linear / omega_t) * (1 + sum_s f_st^2 / (omega_s^2 - omega_t^2))",
            "terms": watson_terms,
        }
        item["spectroscopic_linear_constants_hz"] = {
            "q_e0": q_l_leading_hz,
            "q_eW": q_l_watson_hz,
            "q_v": None,
            "note": "q_e0 and q_eW are standard spectroscopic estimates; q_v requires additional vibrational-state corrections not yet included here.",
        }
        item["effective_linear_model_hz"] = {
            "primary_basis": ["I_l", "Z_l", "X_l", "Y_l"],
            "model": "H_eff^(lin) = q_e^(W) X_l + q_J^(pair) J^2 X_l + q_H^(pair) (J^2)^2 X_l",
            "constants_hz": {
                "q_e0": q_l_leading_hz,
                "q_eW": q_l_watson_hz,
                "q_J_pair": q_tj_val,
                "q_H_pair": q_th_val,
            },
        }
        item["conventional_pair_mapping"] = {
            "status": "minimal_pairwise_ready",
            "active_channel": "X_l",
            "inactive_channels": ["I_l", "Z_l", "Y_l"],
            "constants_hz": {
                "q_l": float(q_tabs),
                "q_lJ": q_tj_val,
                "q_lH": q_th_val,
            },
            "legacy_alias_hz": {
                "q_t": float(q_tabs),
                "q_tJ": q_tj_val,
                "q_tH": q_th_val,
            },
            "note": "Current linear pair mapping is explicit on the circular-doublet carrier; q_l, q_l^J, q_l^H are the primary circular-basis constants and q_t, q_t^J, q_t^H are kept as legacy aliases.",
        }
        item["pair_basis_coefficients_real_hz"] = {
            "I_t": 0.0,
            "O_t": float(q_tabs),
            "X_t": 0.0,
            "Y_t": 0.0,
        }
        item["pair_basis_coefficients_alt_hz"] = {
            "I_l": 0.0,
            "Z_l": 0.0,
            "X_l": float(q_tabs),
            "Y_l": 0.0,
        }
        item["pair_hamiltonian_matrix_real_hz"] = [
            [float(q_tabs), 0.0],
            [0.0, -float(q_tabs)],
        ]
        item["pair_hamiltonian_matrix_alt_hz"] = [
            [0.0, float(q_tabs)],
            [float(q_tabs), 0.0],
        ]
        item["pair_basis_coefficients_real_J2_hz"] = {
            "I_t": 0.0,
            "O_t": q_tj_val,
            "X_t": 0.0,
            "Y_t": 0.0,
        }
        item["pair_basis_coefficients_alt_J2_hz"] = {
            "I_l": 0.0,
            "Z_l": 0.0,
            "X_l": q_tj_val,
            "Y_l": 0.0,
        }
        item["pair_basis_coefficients_real_J4_hz"] = {
            "I_t": 0.0,
            "O_t": q_th_val,
            "X_t": 0.0,
            "Y_t": 0.0,
        }
        item["pair_basis_coefficients_alt_J4_hz"] = {
            "I_l": 0.0,
            "Z_l": 0.0,
            "X_l": q_th_val,
            "Y_l": 0.0,
        }
        item["solver_facing_circular_basis"] = ["I_l", "Z_l", "X_l", "Y_l"]
        item["solver_facing_blocks_hz"] = {
            "J0": {
                "vector": [0.0, 0.0, q_l_watson_hz, 0.0],
                "matrix": [[0.0, q_l_watson_hz], [q_l_watson_hz, 0.0]],
            },
            "J2": {
                "vector": [0.0, 0.0, q_tj_val, 0.0],
                "matrix": [[0.0, q_tj_val], [q_tj_val, 0.0]],
            },
            "J4": {
                "vector": [0.0, 0.0, q_th_val, 0.0],
                "matrix": [[0.0, q_th_val], [q_th_val, 0.0]],
            },
        }
        item["operator_terms_hz"] = [
            {"operator": "O_t", "coefficient_hz": float(q_tabs)},
            {"operator": "J^2 O_t", "coefficient_hz": q_tj_val},
            {"operator": "(J^2)^2 O_t", "coefficient_hz": q_th_val},
        ]
        item["operator_terms_alt_hz"] = [
            {"operator": "X_l", "coefficient_hz": float(q_tabs)},
            {"operator": "J^2 X_l", "coefficient_hz": q_tj_val},
            {"operator": "(J^2)^2 X_l", "coefficient_hz": q_th_val},
        ]
        pairs.append(item)
    pure_rotational_branch = {
        "kind": "linear_pure_rotational",
        "quartic_special": quartic_special,
        "sextic_special": sextic_special,
        "scalars": {
            "D_mhz": float(d_hz / 1.0e6) if d_hz is not None else None,
            "H_hz": float(h_hz) if h_hz is not None else None,
        },
    }
    pairwise_ltype_branch = {
        "kind": "linear_pairwise_ltype",
        "pair_count": len(pairs),
        "active_operator_channel": "X_l",
        "inactive_operator_channels": ["I_l", "Z_l", "Y_l"],
        "driving_scalars": {
            "D_mhz": float(d_hz / 1.0e6) if d_hz is not None else None,
            "H_hz": float(h_hz) if h_hz is not None else None,
        },
        "rotational_feeds": {
            "quartic_feed_operator": "J^2 X_l",
            "sextic_feed_operator": "(J^2)^2 X_l",
            "quartic_feed_present": bool(d_hz is not None),
            "sextic_feed_present": bool(h_hz is not None),
        },
    }
    return {
        "B_linear_cm": b_linear_cm,
        "pairs": pairs,
        "kind": "linear",
        "literature_convention": {
            "name": "circular_vibrational_angular_momentum_basis",
            "basis": ["I_l", "Z_l", "X_l", "Y_l"],
            "active_minimal_channel": "X_l",
            "justification": "Natural basis for degenerate bendings in linear molecules; closest to l-type spectroscopy and already aligned with the solver-facing tensor blocks.",
        },
        "pure_rotational_branch": pure_rotational_branch,
        "pairwise_ltype_branch": pairwise_ltype_branch,
        "diagnostic_only": False,
        "operator_basis": "circular_doublet",
        "operator_definition": "X_l = |+><-| + |-><+|",
        "operator_basis_equiv": "real_doublet",
        "operator_definition_equiv": "O_t = |u><u| - |v><v|",
        "full_operator_basis_real": full_basis_real,
        "full_operator_basis_circular": full_basis_circular,
        "operator_basis_alt": "real_doublet",
        "operator_definition_alt": "O_t = |u><u| - |v><v|",
        "full_operator_basis_alt": full_basis_real,
        "effective_model": "H_l^(ij) = q_t X_l + q_t^J J^2 X_l + q_t^H (J^2)^2 X_l",
        "effective_model_alt": "H_l^(ij) = q_t O_t + q_t^J J^2 O_t + q_t^H (J^2)^2 O_t",
    }


def _quartic_axis_extract(tau: dict[str, float], axis: str) -> dict[str, float]:
    if axis == "a":
        perp_diag = 0.5 * (tau["tau_yyyy"] + tau["tau_zzzz"])
        perp_cross = 0.5 * tau["tau_yyzz"]
        t_pp = 0.5 * (perp_diag + perp_cross)
        t_pq = 0.5 * (tau["tau_xxyy"] + tau["tau_xxzz"]) - 2.0 * t_pp
        t_qq = tau["tau_xxxx"] - t_pp - t_pq
        return {"DJ": t_pp, "DJK": t_pq, "DK": t_qq}
    if axis == "c":
        perp_diag = 0.5 * (tau["tau_xxxx"] + tau["tau_yyyy"])
        perp_cross = 0.5 * tau["tau_xxyy"]
        t_pp = 0.5 * (perp_diag + perp_cross)
        t_pq = 0.5 * (tau["tau_xxzz"] + tau["tau_yyzz"]) - 2.0 * t_pp
        t_qq = tau["tau_zzzz"] - t_pp - t_pq
        return {"DJ": t_pp, "DJK": t_pq, "DK": t_qq}
    raise ValueError(f"Unsupported symmetry axis '{axis}'.")


def project_special_quartic_constants(quartic_input: dict[str, float], abc: np.ndarray, moments: np.ndarray) -> dict[str, object] | None:
    """Project quartic tensor content onto symmetry-adapted special-limit constants."""
    rotor_limit = classify_rotor_limit(abc, moments)
    if not rotor_limit["is_special_limit"]:
        return None
    if {"tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"} <= set(quartic_input):
        tau = {key: float(quartic_input[key]) for key in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")}
    else:
        return {"quartic_mhz": {key: float(val) for key, val in quartic_input.items()}, "kind": rotor_limit["kind"]}

    kind = rotor_limit["kind"]
    if kind == "linear":
        dval = np.mean([tau["tau_yyyy"], tau["tau_zzzz"], 0.5 * tau["tau_yyzz"]])
        return {"quartic_mhz": {"D": float(dval)}, "kind": kind}
    if kind == "spherical_top":
        dval = np.mean(
            [
                tau["tau_xxxx"],
                tau["tau_yyyy"],
                tau["tau_zzzz"],
                0.5 * tau["tau_xxyy"],
                0.5 * tau["tau_xxzz"],
                0.5 * tau["tau_yyzz"],
            ]
        )
        return {"quartic_mhz": {"D_sph": float(dval)}, "kind": kind}
    axis = str(rotor_limit["symmetry_axis"])
    return {"quartic_mhz": _quartic_axis_extract(tau, axis), "kind": kind}


def _sextic_axis_extract(phi: dict[str, float], axis: str) -> dict[str, float]:
    if axis == "a":
        h_j = np.mean([phi["bbb"], phi["ccc"], phi["bbc"] / 3.0, phi["bcc"] / 3.0])
        h_jk = np.mean([phi["abb"], phi["acc"], phi["abc"] / 2.0]) - 3.0 * h_j
        h_kj = np.mean([phi["aab"], phi["aac"]]) - h_jk - 3.0 * h_j
        h_k = phi["aaa"] - h_kj - h_jk - h_j
        return {"H_J": float(h_j), "H_JK": float(h_jk), "H_KJ": float(h_kj), "H_K": float(h_k)}
    if axis == "c":
        h_j = np.mean([phi["aaa"], phi["bbb"], phi["aab"] / 3.0, phi["abb"] / 3.0])
        h_jk = np.mean([phi["aac"], phi["bcc"], phi["abc"] / 2.0]) - 3.0 * h_j
        h_kj = np.mean([phi["acc"], phi["bbc"]]) - h_jk - 3.0 * h_j
        h_k = phi["ccc"] - h_kj - h_jk - h_j
        return {"H_J": float(h_j), "H_JK": float(h_jk), "H_KJ": float(h_kj), "H_K": float(h_k)}
    raise ValueError(f"Unsupported symmetry axis '{axis}'.")


def project_special_sextic_constants(linear_cart: dict[str, float], abc: np.ndarray, moments: np.ndarray) -> dict[str, object] | None:
    """Project sextic Cartesian coefficients onto special-limit constants."""
    if not linear_cart:
        return None
    rotor_limit = classify_rotor_limit(abc, moments)
    if not rotor_limit["is_special_limit"]:
        return None
    phi = {key: float(linear_cart[key]) for key in linear_cart}
    kind = rotor_limit["kind"]
    if kind == "linear":
        hval = np.mean([phi["bbb"], phi["ccc"], phi["bbc"] / 3.0, phi["bcc"] / 3.0])
        return {"sextic_hz": {"H": float(hval)}, "kind": kind}
    if kind == "spherical_top":
        hval = np.mean(
            [
                phi["aaa"],
                phi["bbb"],
                phi["ccc"],
                phi["aab"] / 3.0,
                phi["aac"] / 3.0,
                phi["abb"] / 3.0,
                phi["acc"] / 3.0,
                phi["bbc"] / 3.0,
                phi["bcc"] / 3.0,
                phi["abc"] / 6.0,
            ]
        )
        return {"sextic_hz": {"H_sph": float(hval)}, "kind": kind}
    axis = str(rotor_limit["symmetry_axis"])
    return {"sextic_hz": _sextic_axis_extract(phi, axis), "kind": kind}
