#!/usr/bin/env python3
"""Unified GUI for quartic/sextic axis-representation transforms.

Included in one program:
1) Tensor quartic transform: Watson -> tau -> permute -> Watson
2) Tensor sextic transform on the validated invariant subspace
3) Optional legacy quartic comparison for A reduction

Representations: I, II, III.
Reduction is preserved (input reduction == output reduction).
"""

from __future__ import annotations

from pathlib import Path
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sympy as sp

from rovib_distortion import transform_constants as transform_quartic_yamada_a_legacy
from rovib_distortion import (
    AU_FREQ_TO_CMINV,
    harmonic_inertia_model_from_geometry_hessian,
    inertia_tensor,
    molecule_from_xyz_text,
    read_hessian,
    read_xyz,
    rotational_constants,
)
from gaussian_vpt_parser import (
    align_gaussian_cubic_force_constants,
    parse_gaussian_alpha_data,
    parse_gaussian_harmonic_data,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
)
from compare_gaussian_sextic import sextic_cubic_hierarchy_hz, sextic_h22_linear_candidate_hz
from compare_gaussian_sextic import split_cubic_force_constants
from vibrot_alpha import (
    alpha_matrix_from_cubic_two_index_cm,
    expand_cubic_two_index_matrix,
    read_cubic_two_index_matrix,
)
from distortion_workflow import (
    classify_rotor_limit,
    compute_order2_quartic,
    linear_ltype_terms,
    project_special_quartic_constants,
    project_special_sextic_constants,
)
from symmetry_metadata import (
    point_group_from_geometry,
    point_group_metadata_from_model,
    rotor_type_for_symmetry,
    symbols_from_atomic_numbers,
)
from quartic_channels import (
    BOHR_TO_ANG,
    CMINV_TO_MHZ,
    channel_h22,
    channel_h22_from_mu1_intrinsic,
    gaussian_asymmetric_a_from_t,
    gaussian_symmetric_from_t,
    gaussian_t_from_tauprime,
)


REPRESENTATIONS = ("I", "II", "III")
REDUCTIONS = ("A", "S")

QUARTIC_A_NAMES = ("AJ", "AJK", "AK", "dJ", "dK")
QUARTIC_S_NAMES = ("DJ", "DJK", "DK", "d1", "d2")
SEXTIC_A_NAMES = ("PhiJ", "PhiJK", "PhiKJ", "PhiK", "phiJ", "phiJK", "phiK")
SEXTIC_S_NAMES = ("HJ", "HJK", "HKJ", "HK", "h1", "h2", "h3")
SEXTIC_KEYS = SEXTIC_S_NAMES


def _as_float(v: str) -> float:
    s = v.strip().replace("D", "E").replace("d", "e")
    return float(s) if s else 0.0


def _norm_rep(rep: str) -> str:
    rep = rep.strip().upper().replace("R", "")
    if rep not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{rep}'. Use I, II, or III.")
    return rep


def _norm_reduction(red: str) -> str:
    red = red.strip().upper()
    if red not in REDUCTIONS:
        raise ValueError(f"Unknown reduction '{red}'. Use A or S.")
    return red


def _parse_mode_selection(spec: str) -> set[int]:
    """Parse a 1-based mode selection like ``1,3-5,8``."""
    out: set[int] = set()
    text = spec.strip()
    if not text:
        return out
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            i = int(left.strip())
            j = int(right.strip())
            if i <= 0 or j <= 0:
                raise ValueError("Mode indices must be positive integers.")
            if j < i:
                i, j = j, i
            out.update(range(i, j + 1))
            continue
        idx = int(item)
        if idx <= 0:
            raise ValueError("Mode indices must be positive integers.")
        out.add(idx)
    return out


def _cycle_shift(rep_from: str, rep_to: str) -> int:
    order = ["I", "II", "III"]
    i = order.index(_norm_rep(rep_from))
    j = order.index(_norm_rep(rep_to))
    return (j - i) % 3


def _rotate_abc(A: float, B: float, C: float, rep_from: str, rep_to: str) -> tuple[float, float, float]:
    s = _cycle_shift(rep_from, rep_to)
    if s == 0:
        return A, B, C
    if s == 1:
        return B, C, A
    return C, A, B


# --- Quartic Yamada / Yamada-like maps --------------------------------------------

def get_Q_A(rep: str, A: float, B: float, C: float) -> np.ndarray:
    rep = _norm_rep(rep)

    r1 = np.array([-1.0, -1.0, -1.0, 0.0, 0.0])
    r2 = np.array([-1.0, 0.0, 0.0, -2.0, 0.0])
    r3 = np.array([-1.0, 0.0, 0.0, 2.0, 0.0])
    r4 = np.array([-3.0, -1.0, 0.0, 0.0, 0.0])

    if rep == "I":
        rows = [r1, r2, r3, r4, np.array([A + B + C, -(B + C) / 2.0, 0.0, B - C, B - C])]
    elif rep == "II":
        rows = [r3, r1, r2, r4, np.array([A + B + C, -(A + B) / 2.0, 0.0, A - B, A - B])]
    else:
        rows = [r2, r3, r1, r4, np.array([A + B + C, -(C + A) / 2.0, 0.0, C - A, C - A])]
    return np.vstack(rows)


def get_Q_S(rep: str, A: float, B: float, C: float) -> np.ndarray:
    """S-reduction quartic map from explicit S-Hamiltonian coefficient matching."""
    rep = _norm_rep(rep)
    if rep == "I":
        return np.array(
            [
                [-1.0, 0.0, 0.0, -1.0, -1.0],
                [-1.0, 0.0, 0.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, 0.0, 0.0],
                [-3.0, -1.0, 0.0, 0.0, 0.0],
                [-2.0 * (A + B + C), -(B + C), 0.0, -B + C, 0.0],
            ]
        )
    if rep == "II":
        return np.array(
            [
                [-1.0, -1.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, -1.0, -1.0],
                [-1.0, 0.0, 0.0, 1.0, 1.0],
                [-3.0, -1.0, 0.0, 0.0, 0.0],
                [-2.0 * (A + B + C), -(A + B), 0.0, -A + B, 0.0],
            ]
        )
    return np.array(
        [
            [-1.0, 0.0, 0.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -1.0, -1.0],
            [-3.0, -1.0, 0.0, 0.0, 0.0],
            [-2.0 * (A + B + C), -(A + C), 0.0, A - C, 0.0],
        ]
    )


def transform_quartic_reference(
    D: np.ndarray,
    A: float,
    B: float,
    C: float,
    rep_from: str,
    rep_to: str,
    reduction: str,
) -> np.ndarray:
    """Legacy comparison path.

    For A reduction this uses the long-standing Yamada-based implementation in
    ``rovib_distortion.py``. For S reduction there is no independently validated
    legacy backend in this app, so we fall back to the tensor/pseudoinverse
    route and treat it as the authoritative transform.
    """
    rep_from = _norm_rep(rep_from)
    rep_to = _norm_rep(rep_to)
    reduction = _norm_reduction(reduction)
    D = np.asarray(D, dtype=float).reshape(5)

    if reduction == "A":
        return transform_quartic_yamada_a_legacy(D, A, B, C, rep_from, rep_to)

    return transform_quartic_tensor(D, A, B, C, rep_from, rep_to, reduction)


# --- Quartic tensor algorithm -------------------------------------------------------

def _perm_tau_indices(shift: int) -> np.ndarray:
    # tau = [aaaa, bbbb, cccc, aabb, aacc, bbcc]
    if shift == 0:
        return np.array([0, 1, 2, 3, 4, 5], dtype=int)
    if shift == 1:
        return np.array([1, 2, 0, 5, 3, 4], dtype=int)
    return np.array([2, 0, 1, 4, 5, 3], dtype=int)


def _tauprime_from_tau_vec(tau: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau, dtype=float).reshape(6)
    return np.array(
        [
            [tau[0], tau[3] / 2.0, tau[4] / 2.0],
            [tau[3] / 2.0, tau[1], tau[5] / 2.0],
            [tau[4] / 2.0, tau[5] / 2.0, tau[2]],
        ],
        dtype=float,
    )


def _quartic_spectral_invariants_from_tau(tau: np.ndarray) -> dict[str, object]:
    """Return Tau' and its spectral/polynomial invariants from compressed tau."""
    tauprime = _tauprime_from_tau_vec(tau)
    evals = np.linalg.eigvalsh(tauprime)
    evals = np.sort(evals)[::-1]
    tr = float(np.trace(tauprime))
    j2 = 0.5 * (tr * tr - float(np.trace(tauprime @ tauprime)))
    det = float(np.linalg.det(tauprime))
    return {
        "tauprime": tauprime,
        "lambda": evals,
        "I1": tr,
        "I2": j2,
        "I3": det,
    }


def _sigma_values_quartic(A: float, B: float, C: float) -> tuple[float, float]:
    sigma = (2.0 * A - B - C) / (B - C)
    sigma1 = 1.0 / sigma
    return sigma, sigma1


def _wrap_angle_pi(x: float) -> float:
    y = (x + math.pi) % (2.0 * math.pi) - math.pi
    if y <= -math.pi:
        y += 2.0 * math.pi
    return y


def _zyz_angles_from_rotation(rmat: np.ndarray) -> tuple[float, float, float]:
    beta = float(np.arccos(max(-1.0, min(1.0, float(rmat[2, 2])))))
    if abs(np.sin(beta)) > 1.0e-12:
        alpha = float(np.arctan2(rmat[1, 2], rmat[0, 2]))
        gamma = float(np.arctan2(rmat[2, 1], -rmat[2, 0]))
    else:
        alpha = float(np.arctan2(rmat[0, 1], rmat[0, 0]))
        gamma = 0.0
    return alpha, beta, gamma


def _tauprime_from_spectral_zyz(l1: float, l2: float, l3: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rz1 = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=float)
    rz2 = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rmat = rz1 @ ry @ rz2
    return rmat @ np.diag([l1, l2, l3]) @ rmat.T


def _quartic_reduced_3plus2_from_tau(tau: np.ndarray, A: float, B: float, C: float) -> dict[str, float]:
    """Return a practical 3+2 parameterization on the pseudoinverse slice.

    Start from the full spectral data of Tau' and recover gamma by imposing the
    quartic pseudoinverse gauge-fixing condition on Tau':

        F = 2 Tau'12 + (Sigma-1) Tau'13 - (Sigma+1) Tau'23 = 0.
    """
    spec = _quartic_spectral_invariants_from_tau(tau)
    tauprime = spec["tauprime"]
    eigvals, eigvecs = np.linalg.eigh(tauprime)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    rmat = eigvecs[:, order]
    if np.linalg.det(rmat) < 0.0:
        rmat[:, 2] *= -1.0
    alpha, beta, gamma0 = _zyz_angles_from_rotation(rmat)
    sigma, _ = _sigma_values_quartic(A, B, C)

    def f(gamma: float) -> float:
        tp = _tauprime_from_spectral_zyz(float(eigvals[0]), float(eigvals[1]), float(eigvals[2]), alpha, beta, gamma)
        return float(2.0 * tp[0, 1] + (sigma - 1.0) * tp[0, 2] - (sigma + 1.0) * tp[1, 2])

    gammas = np.linspace(-math.pi, math.pi, 721)
    vals = np.array([f(g) for g in gammas], dtype=float)
    best = int(np.argmin(np.abs(vals)))
    gamma = float(gammas[best])
    # Bisection on best sign-change bracket if present.
    for i in range(len(gammas) - 1):
        if vals[i] * vals[i + 1] < 0.0:
            gl, gr = float(gammas[i]), float(gammas[i + 1])
            fl, fr = vals[i], vals[i + 1]
            for _ in range(80):
                gm = 0.5 * (gl + gr)
                fm = f(gm)
                gamma = gm
                if abs(fm) < 1.0e-14:
                    break
                if fl * fm <= 0.0:
                    gr, fr = gm, fm
                else:
                    gl, fl = gm, fm
            break
    return {
        "lambda1": float(eigvals[0]),
        "lambda2": float(eigvals[1]),
        "lambda3": float(eigvals[2]),
        "alpha_zyz": float(alpha),
        "beta_zyz": float(beta),
        "gamma_zyz_recovered": float(_wrap_angle_pi(gamma)),
        "gamma_zyz_raw": float(_wrap_angle_pi(gamma0)),
        "constraint_F_recovered": float(f(gamma)),
    }


def _quartic_forward_constants(reduction: str, tau: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Return Watson quartic constants from compressed tau via Tau' -> T -> cylindrical."""
    red = _norm_reduction(reduction)
    tauprime = _tauprime_from_tau_vec(tau)
    t = tauprime / 4.0
    t11, t22, t33 = t[0, 0], t[1, 1], t[2, 2]
    t12, t13, t23 = t[0, 1], t[0, 2], t[1, 2]
    t400 = (3.0 * t11 + 3.0 * t22 + 2.0 * t12) / 8.0
    t220 = t13 + t23 - 2.0 * t400
    t040 = t33 - t220 - t400
    t202 = (t11 - t22) / 4.0
    t022 = (t13 - t23) / 2.0 - t202
    t004 = (t11 + t22 - 2.0 * t12) / 16.0
    sigma = (2.0 * A - B - C) / (B - C)
    sigma1 = 1.0 / sigma
    if red == "A":
        return np.array(
            [
                -t400 - 2.0 * t004,
                -t220 + 12.0 * t004,
                -t040 - 10.0 * t004,
                -t202,
                -t022 - 4.0 * sigma * t004,
            ],
            dtype=float,
        )
    return np.array(
        [
            -t400 + 0.5 * t022 * sigma1,
            -t220 - 3.0 * t022 * sigma1,
            -t040 + 2.5 * t022 * sigma1,
            t202,
            t004 + 0.25 * t022 * sigma1,
        ],
        dtype=float,
    )


def _M4(A: float, B: float, C: float, reduction: str) -> np.ndarray:
    """Build the exact quartic map D = M4 tau by basis probing."""
    M = np.zeros((5, 6), dtype=float)
    for k in range(6):
        e = np.zeros(6, dtype=float)
        e[k] = 1.0
        M[:, k] = _quartic_forward_constants(reduction, e, A, B, C)
    return M


def _transform_matrix_quartic(A: float, B: float, C: float, rep_from: str, rep_to: str, reduction: str) -> np.ndarray:
    T = np.zeros((5, 5), dtype=float)
    for k in range(5):
        e = np.zeros(5, dtype=float)
        e[k] = 1.0
        T[:, k] = transform_quartic_reference(e, A, B, C, rep_from, rep_to, reduction)
    return T


def quartic_transform_matrix(
    A: float,
    B: float,
    C: float,
    rep_from: str,
    rep_to: str,
    reduction: str,
    method: str,
) -> np.ndarray:
    """Return 5x5 matrix for quartic transform D_out = T * D_in."""
    rep_from = _norm_rep(rep_from)
    rep_to = _norm_rep(rep_to)
    reduction = _norm_reduction(reduction)

    if method == "reference":
        return _transform_matrix_quartic(A, B, C, rep_from, rep_to, reduction)

    # Tensor path uses linear mapping with basis probing.
    return _transform_matrix_quartic(A, B, C, rep_from, rep_to, reduction)


def stability_metrics(T: np.ndarray, A: float, B: float, C: float) -> dict[str, float]:
    """Compute numerical-stability indicators for a quartic transform matrix."""
    T = np.asarray(T, dtype=float).reshape(5, 5)
    svals = np.linalg.svd(T, compute_uv=False)
    k2 = float(np.linalg.cond(T, 2))
    k1 = float(np.linalg.cond(T, 1))
    kinf = float(np.linalg.cond(T, np.inf))
    sigma_min = float(np.min(svals))
    sigma_max = float(np.max(svals))
    amp = k2 * np.finfo(float).eps
    return {
        "cond2": k2,
        "cond1": k1,
        "condinf": kinf,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "amp_eps": float(amp),
        "A_minus_B": float(A - B),
        "B_minus_C": float(B - C),
        "A_minus_C": float(A - C),
    }


def compute_s111(A: float, B: float, C: float, D_khz: np.ndarray, reduction: str) -> float:
    """Compute Watson s111 using rotational constants in MHz and quartics in kHz."""
    d = np.asarray(D_khz, dtype=float).reshape(5) / 1000.0  # kHz -> MHz
    red = _norm_reduction(reduction)
    xj = d[3]
    xk = d[4]
    if abs(A - B) < 1e-14 or abs(B - C) < 1e-14 or abs(A - C) < 1e-14:
        return float("nan")
    return (1.0 / (2.0 * (A - B))) * (xj / (B - C) - xk / (A - C))


def compute_T_over_B(B: float, D_khz: np.ndarray) -> float:
    """Compute T/B with T = max |quartic| and B converted to kHz."""
    B_khz = 1000.0 * float(B)
    if abs(B_khz) < 1e-14:
        return float("nan")
    T_khz = float(np.max(np.abs(np.asarray(D_khz, dtype=float).reshape(5))))
    return T_khz / B_khz


def stability_warning(cond2: float) -> str:
    if cond2 >= 1.0e4:
        return "SEVERE (cond2 >= 1e4)"
    if cond2 >= 1.0e3:
        return "HIGH (cond2 >= 1e3)"
    return "OK"


def transform_quartic_tensor(
    D: np.ndarray,
    A: float,
    B: float,
    C: float,
    rep_from: str,
    rep_to: str,
    reduction: str,
) -> np.ndarray:
    D = np.asarray(D, dtype=float).reshape(5)
    rep_from = _norm_rep(rep_from)
    rep_to = _norm_rep(rep_to)
    reduction = _norm_reduction(reduction)

    M_from = _M4(A, B, C, reduction)
    tau = np.linalg.pinv(M_from) @ D

    idx = _perm_tau_indices(_cycle_shift(rep_from, rep_to))
    P = np.eye(6)[idx, :]
    tau_p = P @ tau

    A2, B2, C2 = _rotate_abc(A, B, C, rep_from, rep_to)
    M_to = _M4(A2, B2, C2, reduction)
    return M_to @ tau_p


def _sympy_modepair_tensor(arr: np.ndarray) -> sp.MutableDenseNDimArray:
    n_modes = arr.shape[2]
    return sp.MutableDenseNDimArray(
        [sp.Float(arr[a, b, k, l]) for a in range(3) for b in range(3) for k in range(n_modes) for l in range(n_modes)],
        (3, 3, n_modes, n_modes),
    )


def _sympy_rank2_tensor(arr: np.ndarray) -> sp.MutableDenseNDimArray:
    return sp.MutableDenseNDimArray([sp.Float(arr[a, b]) for a in range(3) for b in range(3)], (3, 3))


def _gaussian_tauprime_from_compressed_tau(tau: dict[str, sp.Expr]) -> dict[tuple[int, int], sp.Expr]:
    return {
        (0, 0): tau["tau_xxxx"],
        (1, 1): tau["tau_yyyy"],
        (2, 2): tau["tau_zzzz"],
        (0, 1): sp.simplify(tau["tau_xxyy"] / 2),
        (1, 0): sp.simplify(tau["tau_xxyy"] / 2),
        (0, 2): sp.simplify(tau["tau_xxzz"] / 2),
        (2, 0): sp.simplify(tau["tau_xxzz"] / 2),
        (1, 2): sp.simplify(tau["tau_yyzz"] / 2),
        (2, 1): sp.simplify(tau["tau_yyzz"] / 2),
    }


def _h22_from_fchk(fchk_path: str, representation: str, reduction: str) -> dict[str, object]:
    rep = _norm_rep(representation)
    red = _norm_reduction(reduction)
    fchk = parse_gaussian_fchk_harmonic_data(fchk_path)
    model = harmonic_inertia_model_from_geometry_hessian(
        fchk.masses_amu,
        fchk.coords_bohr * BOHR_TO_ANG,
        fchk.cartesian_force_constants,
        representation=rep,
        symbols=symbols_from_atomic_numbers(fchk.atomic_numbers),
    )
    omega_au = tuple(sp.Float(abs(x) / AU_FREQ_TO_CMINV) for x in model.vib_freq_cm)
    n_modes = model.dInv_au.shape[2]
    mu1 = sp.MutableDenseNDimArray(
        [sp.Float(model.dInv_au[a, b, k]) for a in range(3) for b in range(3) for k in range(n_modes)],
        (3, 3, n_modes),
    )
    mu2 = _sympy_modepair_tensor(np.asarray(model.d2Inv_au, dtype=float))
    intrinsic = _sympy_modepair_tensor(np.asarray(model.d2Inv_intrinsic_au, dtype=float))
    inertia0 = _sympy_rank2_tensor(np.asarray(model.i_tensor_au, dtype=float))
    tau_h22 = channel_h22(mu2, omega_au, sp.Integer(1))
    h22_decomp = channel_h22_from_mu1_intrinsic(mu1, intrinsic, inertia0, omega_au, sp.Integer(1))
    sigma, sigma1 = _sigma_values_quartic(*[float(x) for x in model.abc_mhz])
    taup = _gaussian_tauprime_from_compressed_tau(tau_h22)
    tmat = gaussian_t_from_tauprime(taup)
    watson_a = {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in gaussian_asymmetric_a_from_t(tmat, sp.Float(sigma)).items()}
    watson_s = {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in gaussian_symmetric_from_t(tmat, sp.Float(sigma1)).items()}

    decomp_khz: dict[str, dict[str, float]] = {}
    for name, tau in h22_decomp.items():
        taup_piece = _gaussian_tauprime_from_compressed_tau(tau)
        tmat_piece = gaussian_t_from_tauprime(taup_piece)
        if red == "A":
            piece = gaussian_asymmetric_a_from_t(tmat_piece, sp.Float(sigma))
        else:
            piece = gaussian_symmetric_from_t(tmat_piece, sp.Float(sigma1))
        decomp_khz[name] = {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in piece.items()}

    total = watson_a if red == "A" else watson_s
    return {
        "representation": rep,
        "reduction": red,
        "abc_mhz": tuple(float(x) for x in model.abc_mhz),
        "h22_total_khz": total,
        "h22_A_khz": watson_a,
        "h22_S_khz": watson_s,
        "h22_decomposition_khz": decomp_khz,
        "maxabs_total_khz": float(max(abs(v) for v in total.values())),
    }


def _build_harmonic_model_from_inputs(
    representation: str,
    *,
    fchk_path: str = "",
    xyz_path: str = "",
    hessian_path: str = "",
):
    rep = _norm_rep(representation)
    has_fchk = bool(fchk_path.strip())
    has_xyzh = bool(xyz_path.strip() and hessian_path.strip())
    if has_fchk and has_xyzh:
        raise ValueError("Use either .fchk or xyz+hessian, not both.")
    if not has_fchk and not has_xyzh:
        raise ValueError("Provide either .fchk or both xyz and Hessian.")
    if has_fchk:
        fchk = parse_gaussian_fchk_harmonic_data(fchk_path)
        model = harmonic_inertia_model_from_geometry_hessian(
            fchk.masses_amu,
            fchk.coords_bohr * BOHR_TO_ANG,
            fchk.cartesian_force_constants,
            representation=rep,
            symbols=symbols_from_atomic_numbers(fchk.atomic_numbers),
        )
        source = fchk_path
    else:
        mol = _read_xyz_input(xyz_path)
        h = read_hessian(hessian_path, mol.n_atoms)
        model = harmonic_inertia_model_from_geometry_hessian(
            mol.masses_amu,
            mol.coords_ang,
            h,
            representation=rep,
            symbols=mol.symbols,
        )
        source = f"{xyz_path} + {hessian_path}"
    return model, source


def _format_freqs_cm(freq_cm: np.ndarray, limit: int = 12) -> str:
    vals = [f"{float(x):.3f}" for x in np.asarray(freq_cm, dtype=float)]
    if len(vals) <= limit:
        return ", ".join(vals)
    head = ", ".join(vals[:limit])
    return f"{head}, ... ({len(vals)} modes)"


def _append_point_group_report(widget: tk.Text, model) -> None:
    meta = point_group_metadata_from_model(model)
    if not meta:
        return
    widget.insert(
        tk.END,
        "symmetry: "
        f"point group={meta['point_group']}, "
        f"sigma={meta['rotational_symmetry_number']}, "
        f"rotor class={meta['rotor_type_for_symmetry']}\n",
    )


def _point_group_from_xyz_file(xyz_path: str) -> dict[str, object]:
    mol = _read_xyz_input(xyz_path)
    i_tensor, com = inertia_tensor(mol.masses_amu, mol.coords_ang)
    moments, abc_mhz, principal_axes = rotational_constants(i_tensor)
    coords_com = mol.coords_ang - com
    coords_pa = coords_com @ principal_axes
    rotor_type = rotor_type_for_symmetry(np.asarray(abc_mhz, dtype=float), np.asarray(moments, dtype=float))
    sigma, point_group = point_group_from_geometry(mol.symbols, coords_pa, rotor_type, tol=1.0e-3)
    return {
        "point_group": point_group,
        "rotational_symmetry_number": int(sigma),
        "rotor_type_for_symmetry": rotor_type,
        "abc_mhz_from_xyz": tuple(float(x) for x in abc_mhz),
    }


def _read_xyz_input(xyz_input: str):
    raw = xyz_input.strip()
    if not raw:
        raise ValueError("Empty XYZ input.")
    pth = Path(raw)
    if "\n" not in raw and pth.exists():
        return read_xyz(pth)
    return molecule_from_xyz_text(raw)


def _append_linear_ltype_report(widget: tk.Text, ltype: dict[str, object] | None, *, title: str) -> None:
    if not ltype:
        return
    pairs = ltype.get("pairs", [])
    if not pairs:
        return
    widget.insert(tk.END, f"{title}\n")
    b_lin = ltype.get("B_linear_cm")
    if b_lin is not None:
        widget.insert(tk.END, f"  B_linear={float(b_lin):.8g} cm^-1\n")
    for pair in pairs:
        modes = pair["modes"]
        line = (
            f"  modes {modes[0] + 1}/{modes[1] + 1}: "
            f"nu={float(pair['freq_cm']):.6g} cm^-1, "
            f"zeta_parallel={float(pair['zeta_parallel']):.6g}, "
            f"|q_t|={float(pair['q_t_abs_hz']):.6g} Hz"
        )
        if "q_tJ_diagnostic_hz" in pair:
            line += f", |q_t^J|={float(pair['q_tJ_diagnostic_hz']):.6g} Hz"
        if "q_tH_diagnostic_hz" in pair:
            line += f", |q_t^H|={float(pair['q_tH_diagnostic_hz']):.6g} Hz"
        widget.insert(tk.END, line + "\n")
    widget.insert(
        tk.END,
        "  Interpretation: these are the linear-molecule l-type doubling terms detected from near-degenerate bending pairs.\n",
    )


# --- Sextic tensor algorithm --------------------------------------------------------

def _sextic_a_to_s_matrix(A: float, B: float, C: float) -> np.ndarray:
    s = (2.0 * A - B - C) / (B - C)
    return np.array(
        [
            [1.0, -1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0 / 3.0, 2.0 / 3.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, s, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -s, 1.0],
        ],
        dtype=float,
    )


def _sextic_s_to_a_matrix(A: float, B: float, C: float) -> np.ndarray:
    if abs(B - C) < 1.0e-14 or abs(2.0 * A - B - C) < 1.0e-14:
        raise ValueError("S->A sextic conversion is singular for B=C or 2A-B-C=0.")
    s = (2.0 * A - B - C) / (B - C)
    return np.array(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],          # PhiJ
            [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],          # PhiJK
            [0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],          # PhiKJ
            [0.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0],        # PhiK
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],          # phi_j
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / (2.0 * s), -1.0 / (2.0 * s)],  # phi_jk
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],          # phi_k
        ],
        dtype=float,
    )


def _B6_S(rep: str) -> np.ndarray:
    rep = _norm_rep(rep)
    if rep == "I":
        return np.array(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 3.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    if rep == "II":
        return np.array(
            [
                [2.0, 3.0 / 5.0, -1.0 / 10.0, 4.0 / 5.0, 13.0 / 20.0],
                [-3.0 / 2.0, -3.0 / 10.0, 3.0 / 10.0, 11.0 / 10.0, -19.0 / 20.0],
                [3.0 / 2.0, -6.0 / 5.0, 1.0 / 5.0, -13.0 / 5.0, -1.0 / 20.0],
                [-1.0, 9.0 / 10.0, -2.0 / 5.0, 17.0 / 10.0, 7.0 / 20.0],
                [1.0, 3.0 / 2.0, 1.0 / 2.0, -1.0 / 2.0, 0.0],
                [1.0, -2.0, 0.0, -1.0, 0.0],
                [5.0 / 2.0, 3.0 / 2.0, 0.0, -3.0 / 2.0, 1.0],
            ],
            dtype=float,
        )
    return np.array(
        [
            [2.0, 3.0 / 5.0, 1.0 / 5.0, 4.0 / 5.0, 7.0 / 20.0],
            [-1.0 / 2.0, -3.0 / 10.0, -3.0 / 5.0, -9.0 / 10.0, -1.0 / 20.0],
            [-3.0 / 2.0, -6.0 / 5.0, 1.0 / 10.0, 17.0 / 5.0, 1.0 / 20.0],
            [1.0, 9.0 / 10.0, 3.0 / 10.0, -23.0 / 10.0, -7.0 / 20.0],
            [-1.0, -3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, -1.0],
            [-1.0, -2.0, 0.0, 3.0, 0.0],
            [-3.0 / 2.0, -3.0 / 2.0, 0.0, -1.0 / 2.0, -1.0],
        ],
        dtype=float,
    )


def _sextic_s_rep_transform(rep_from: str, rep_to: str) -> np.ndarray:
    B_from = _B6_S(rep_from)
    B_to = _B6_S(rep_to)
    return B_to @ np.linalg.pinv(B_from)


def _nonzero_condition_metrics(T: np.ndarray, tol: float = 1.0e-12) -> dict[str, float]:
    """Condition metrics on the nonzero singular spectrum of a rank-deficient map."""
    svals = np.linalg.svd(np.asarray(T, dtype=float), compute_uv=False)
    nz = svals[svals > tol]
    if nz.size == 0:
        return {
            "cond2_nz": float("inf"),
            "sigma_min_nz": 0.0,
            "sigma_max_nz": 0.0,
            "rank_num": 0.0,
            "amp_eps_nz": float("inf"),
        }
    cond = float(nz[0] / nz[-1])
    return {
        "cond2_nz": cond,
        "sigma_min_nz": float(nz[-1]),
        "sigma_max_nz": float(nz[0]),
        "rank_num": float(nz.size),
        "amp_eps_nz": float(cond * np.finfo(float).eps),
    }


def _sextic_condition_metrics(rep_from: str, rep_to: str) -> dict[str, float]:
    """Numerical conditioning diagnostics for the sextic physical subspace."""
    B_from = _B6_S(rep_from)
    B_to = _B6_S(rep_to)
    T_phys = _sextic_s_rep_transform(rep_from, rep_to)
    out = {}
    for key, mat in (
        ("B_from", B_from),
        ("B_to", B_to),
        ("T_phys", T_phys),
    ):
        met = _nonzero_condition_metrics(mat)
        for mk, mv in met.items():
            out[f"{key}_{mk}"] = mv
    return out


def _sextic_physical_subspace_residual(H: np.ndarray, rep: str, reduction: str, A: float, B: float, C: float) -> float:
    """Distance of a sextic constant vector from the modeled physical 5D subspace."""
    H = np.asarray(H, dtype=float).reshape(7)
    reduction = _norm_reduction(reduction)
    if reduction == "A":
        Hs = _sextic_a_to_s_matrix(A, B, C) @ H
    else:
        Hs = H
    Bmat = _B6_S(rep)
    proj = Bmat @ np.linalg.pinv(Bmat) @ Hs
    return float(np.max(np.abs(proj - Hs)))


def _sextic_decomposition(
    H: np.ndarray,
    rep: str,
    reduction: str,
    A: float,
    B: float,
    C: float,
) -> dict[str, np.ndarray | float]:
    """Decompose sextic constants into physical 5D coordinates plus 2D residual.

    The sextic transform is performed on a canonical five-dimensional
    representation-independent subspace S_6, expressed operationally in
    Watson's S reduction:

        H_S = B_rep sigma + r

    where:
    - sigma are the 5 physical coordinates on the modeled sextic subspace S_6
    - r is the orthogonal residual in the 2D complement

    Thus the CeDiTT sextic 5+2 decomposition is not merely a numerical fit:
    it is the decomposition of a 7-component Watson vector with respect to the
    invariant operator subspace S_6 and its complementary 2D residual sector.
    Structurally, the same physical 5D sector may be viewed as:
    - a harmonic 1+4 sector: one scalar plus one even rank-6 component
    - a representation-adapted 1+1+1+2 sector under cyclic axis relabelings.
    """
    H = np.asarray(H, dtype=float).reshape(7)
    reduction = _norm_reduction(reduction)
    if reduction == "A":
        Hs = _sextic_a_to_s_matrix(A, B, C) @ H
    else:
        Hs = H.copy()

    Bmat = _B6_S(rep)
    pinv = np.linalg.pinv(Bmat)
    sigma = pinv @ Hs
    Hs_phys = Bmat @ sigma
    residual = Hs - Hs_phys

    # Orthonormal basis of the 2D complement in the S-canonical 7D space.
    u, _, _ = np.linalg.svd(Bmat, full_matrices=True)
    gauge_basis = u[:, 5:]
    gauge_coords = gauge_basis.T @ residual

    return {
        "Hs": Hs,
        "sigma": sigma,
        "Hs_phys": Hs_phys,
        "residual": residual,
        "residual_maxabs": float(np.max(np.abs(residual))),
        "residual_norm2": float(np.linalg.norm(residual)),
        "gauge_basis": gauge_basis,
        "gauge_coords": gauge_coords,
    }


def transform_sextic_tensor(
    H: np.ndarray,
    A: float,
    B: float,
    C: float,
    rep_from: str,
    rep_to: str,
    red_from: str,
    red_to: str,
) -> np.ndarray:
    """Transform sextic constants through the validated 5D invariant S-subspace.

    The representation change is performed on the canonical S-reduction
    invariant subspace; A<->S conversion is handled explicitly before/after
    the representation transform.
    """
    H = np.asarray(H, dtype=float).reshape(7)
    rep_from = _norm_rep(rep_from)
    rep_to = _norm_rep(rep_to)
    red_from = _norm_reduction(red_from)
    red_to = _norm_reduction(red_to)
    A2, B2, C2 = _rotate_abc(A, B, C, rep_from, rep_to)

    if red_from == "S":
        Hs_in = H
    else:
        Hs_in = _sextic_a_to_s_matrix(A, B, C) @ H

    Hs_out = _sextic_s_rep_transform(rep_from, rep_to) @ Hs_in

    if red_to == "S":
        return Hs_out
    return _sextic_s_to_a_matrix(A2, B2, C2) @ Hs_out


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CeDiTT1.0")
        self.geometry("1080x760")
        self._window_icon: tk.PhotoImage | None = None
        self._set_window_icon()

        self.vars: dict[str, tk.StringVar] = {}

        self.q_in_labels: list[ttk.Label] = []
        self.q_o1_labels: list[ttk.Label] = []
        self.q_o2_labels: list[ttk.Label] = []
        self.s_in_labels: list[ttk.Label] = []
        self.s_o1_labels: list[ttk.Label] = []
        self.s_o2_labels: list[ttk.Label] = []
        self.last_quartic_result: dict[str, object] | None = None

        self._build_ui()

    def _set_window_icon(self) -> None:
        icon_path = Path(__file__).resolve().parent / "assets" / "icons" / "app_icon_256.png"
        if not icon_path.exists():
            return
        try:
            self._window_icon = tk.PhotoImage(file=str(icon_path))
            self.iconphoto(True, self._window_icon)
        except tk.TclError:
            self._window_icon = None

    def _sv(self, k: str, default: str = "") -> tk.StringVar:
        v = tk.StringVar(value=default)
        self.vars[k] = v
        return v

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        ttk.Label(top, text="A (MHz)").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, width=15, textvariable=self._sv("A", "")).grid(row=0, column=1, padx=4)
        ttk.Label(top, text="B (MHz)").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, width=15, textvariable=self._sv("B", "")).grid(row=0, column=3, padx=4)
        ttk.Label(top, text="C (MHz)").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, width=15, textvariable=self._sv("C", "")).grid(row=0, column=5, padx=4)

        ttk.Label(top, text="Required for manual transforms only; harmonic-input paths compute them internally.").grid(row=0, column=6, padx=(10, 0), sticky="w")

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        q_tab = ttk.Frame(notebook, padding=10)
        s_tab = ttk.Frame(notebook, padding=10)
        notebook.add(q_tab, text="Quartic")
        notebook.add(s_tab, text="Sextic")

        self._build_quartic_tab(q_tab)
        self._build_sextic_tab(s_tab)

    def _build_quartic_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Input rep").grid(row=0, column=0, sticky="w")
        ttk.Combobox(ctrl, width=8, state="readonly", values=REPRESENTATIONS, textvariable=self._sv("q_rep_in", "I")).grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Reduction").grid(row=0, column=2, sticky="w")
        red_box = ttk.Combobox(ctrl, width=8, state="readonly", values=REDUCTIONS, textvariable=self._sv("q_red", "A"))
        red_box.grid(row=0, column=3, padx=4)

        ttk.Label(ctrl, text="Method").grid(row=0, column=4, sticky="w")
        ttk.Label(ctrl, text="Tensor / pseudoinverse").grid(row=0, column=5, padx=4, sticky="w")
        ttk.Label(ctrl, text="XYZ (optional)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ctrl, width=48, textvariable=self._sv("q_symm_xyz", "")).grid(row=1, column=1, columnspan=4, padx=4, sticky="we", pady=(6, 0))
        ttk.Button(ctrl, text="Browse", command=self._browse_q_symm_xyz).grid(row=1, column=5, padx=(4, 0), pady=(6, 0))

        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(grid, text="Input quartic constants", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=5, sticky="w")

        for i, name in enumerate(QUARTIC_A_NAMES):
            lbl = ttk.Label(grid, text=name)
            lbl.grid(row=1, column=i, sticky="w")
            self.q_in_labels.append(lbl)
            ttk.Entry(grid, width=16, textvariable=self._sv(f"q_in_{name}", "")).grid(row=2, column=i, padx=4, pady=2)

        ttk.Label(grid, text="Output rep #1", font=("TkDefaultFont", 10, "bold")).grid(row=3, column=0, columnspan=5, sticky="w", pady=(10, 0))
        ttk.Label(grid, textvariable=self._sv("q_out_rep1", "II")).grid(row=3, column=2, sticky="w")
        for i, name in enumerate(QUARTIC_A_NAMES):
            lbl = ttk.Label(grid, text=name)
            lbl.grid(row=4, column=i, sticky="w")
            self.q_o1_labels.append(lbl)
            ttk.Entry(grid, width=16, textvariable=self._sv(f"q_out1_{name}", ""), state="readonly").grid(row=5, column=i, padx=4, pady=2)

        ttk.Label(grid, text="Output rep #2", font=("TkDefaultFont", 10, "bold")).grid(row=6, column=0, columnspan=5, sticky="w", pady=(10, 0))
        ttk.Label(grid, textvariable=self._sv("q_out_rep2", "III")).grid(row=6, column=2, sticky="w")
        for i, name in enumerate(QUARTIC_A_NAMES):
            lbl = ttk.Label(grid, text=name)
            lbl.grid(row=7, column=i, sticky="w")
            self.q_o2_labels.append(lbl)
            ttk.Entry(grid, width=16, textvariable=self._sv(f"q_out2_{name}", ""), state="readonly").grid(row=8, column=i, padx=4, pady=2)

        h22_frame = ttk.LabelFrame(parent, text="Optional H22 diagnostic from harmonic input (.fchk)", padding=8)
        h22_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(h22_frame, text="Formatted checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(h22_frame, width=60, textvariable=self._sv("q_h22_fchk", "")).grid(row=0, column=1, padx=4, sticky="we")
        ttk.Button(h22_frame, text="Browse", command=self._browse_h22_fchk).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(h22_frame, text="XYZ").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(h22_frame, width=60, textvariable=self._sv("q_h22_xyz", "")).grid(row=1, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(h22_frame, text="Browse", command=self._browse_h22_xyz).grid(row=1, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(h22_frame, text="Hessian").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(h22_frame, width=60, textvariable=self._sv("q_h22_hessian", "")).grid(row=2, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(h22_frame, text="Browse", command=self._browse_h22_hessian).grid(row=2, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(
            h22_frame,
            text=(
                "Standard transforms use only centrifugal constants; these extra inputs are only for H22 diagnostics. "
                "XYZ can be given either as a file path or pasted directly into the field. "
                "When used with a Hessian, XYZ and Hessian must refer to the same Cartesian orientation."
            ),
        ).grid(row=3, column=1, sticky="w", pady=(4, 0))
        h22_frame.columnconfigure(1, weight=1)

        alpha_frame = ttk.LabelFrame(parent, text="Gaussian alpha parser / mode filtering", padding=8)
        alpha_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(alpha_frame, text="Gaussian anharmonic log").grid(row=0, column=0, sticky="w")
        ttk.Entry(alpha_frame, width=60, textvariable=self._sv("q_alpha_log", "")).grid(row=0, column=1, padx=4, sticky="we")
        ttk.Button(alpha_frame, text="Browse", command=self._browse_q_alpha_log).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(alpha_frame, text="Exclude modes").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(alpha_frame, width=60, textvariable=self._sv("q_alpha_excluded", "")).grid(row=1, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Label(
            alpha_frame,
            text=(
                "Read the mode-resolved Gaussian Vibro-Rot alpha Matrix and recompute the summed alpha correction "
                "after excluding selected modes. Use 1-based mode indices, e.g. 1,3-5. "
                "This is a parser/filter utility; it does not yet reconstruct the internal perturbative formula."
            ),
        ).grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Label(alpha_frame, text="Alpha from harmonic model (.fchk)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(alpha_frame, width=60, textvariable=self._sv("q_alpha_fchk", "")).grid(row=3, column=1, padx=4, sticky="we", pady=(8, 0))
        ttk.Button(alpha_frame, text="Browse", command=self._browse_q_alpha_fchk).grid(row=3, column=2, padx=(4, 0), pady=(8, 0))
        ttk.Label(alpha_frame, text="Alpha XYZ").grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(alpha_frame, width=60, textvariable=self._sv("q_alpha_xyz", "")).grid(row=4, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(alpha_frame, text="Browse", command=self._browse_q_alpha_xyz).grid(row=4, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(alpha_frame, text="Alpha Hessian").grid(row=5, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(alpha_frame, width=60, textvariable=self._sv("q_alpha_hessian", "")).grid(row=5, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(alpha_frame, text="Browse", command=self._browse_q_alpha_hessian).grid(row=5, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(alpha_frame, text="Alpha cubic 2-index").grid(row=6, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(alpha_frame, width=60, textvariable=self._sv("q_alpha_cubic_2idx", "")).grid(row=6, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(alpha_frame, text="Browse", command=self._browse_q_alpha_cubic_2idx).grid(row=6, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(
            alpha_frame,
            text=(
                "Internal higher-order VPT route for alpha: use the harmonic model together with the semi-diagonal "
                "cubic matrix to compute the Gaussian-style VPT2 alpha corrections. "
                "The current implementation covers asymmetric tops and applies symmetry-adapted stabilization to symmetric tops and linears."
            ),
        ).grid(row=7, column=1, sticky="w", pady=(4, 0))
        alpha_frame.columnconfigure(1, weight=1)

        qbtn = ttk.Frame(parent)
        qbtn.pack(anchor="w", pady=(12, 6))
        ttk.Button(qbtn, text="Compute quartic transforms", command=self._run_quartic).pack(side=tk.LEFT)
        ttk.Button(qbtn, text="Load quartics from harmonic input", command=self._load_quartics_from_harmonic_input).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(qbtn, text="Compute H22 from .fchk", command=self._run_h22_from_fchk).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(qbtn, text="Parse alpha / apply mode filter", command=self._run_alpha_parser).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(qbtn, text="Compute alpha from harmonic+cubic", command=self._prepare_alpha_inputs).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(qbtn, text="Legacy check", command=self._run_quartic_legacy_check).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(qbtn, text="Export CSV", command=self._export_quartic_csv).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(qbtn, text="Export LaTeX", command=self._export_quartic_latex).pack(side=tk.LEFT, padx=(8, 0))

        self.q_report = tk.Text(parent, height=11, wrap="word")
        self.q_report.pack(fill=tk.BOTH, expand=True)

        self.vars["q_red"].trace_add("write", self._refresh_quartic_labels)
        self._refresh_quartic_labels()

    def _build_sextic_tab(self, parent: ttk.Frame) -> None:
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Input rep").grid(row=0, column=0, sticky="w")
        ttk.Combobox(ctrl, width=8, state="readonly", values=REPRESENTATIONS, textvariable=self._sv("s_rep_in", "I")).grid(row=0, column=1, padx=4)

        ttk.Label(ctrl, text="Input reduction").grid(row=0, column=2, sticky="w")
        ttk.Combobox(ctrl, width=8, state="readonly", values=REDUCTIONS, textvariable=self._sv("s_red_in", "A")).grid(row=0, column=3, padx=4)
        ttk.Label(ctrl, text="Output reduction").grid(row=0, column=4, sticky="w")
        ttk.Combobox(ctrl, width=8, state="readonly", values=REDUCTIONS, textvariable=self._sv("s_red_out", "A")).grid(row=0, column=5, padx=4)
        ttk.Label(ctrl, text="XYZ (optional)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ctrl, width=48, textvariable=self._sv("s_symm_xyz", "")).grid(row=1, column=1, columnspan=4, padx=4, sticky="we", pady=(6, 0))
        ttk.Button(ctrl, text="Browse", command=self._browse_s_symm_xyz).grid(row=1, column=5, padx=(4, 0), pady=(6, 0))

        h22s_frame = ttk.LabelFrame(parent, text="Optional sextic analysis from harmonic input", padding=8)
        h22s_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(h22s_frame, text="Formatted checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(h22s_frame, width=60, textvariable=self._sv("s_h22_fchk", "")).grid(row=0, column=1, padx=4, sticky="we")
        ttk.Button(h22s_frame, text="Browse", command=self._browse_s_h22_fchk).grid(row=0, column=2, padx=(4, 0))
        ttk.Label(h22s_frame, text="XYZ").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(h22s_frame, width=60, textvariable=self._sv("s_h22_xyz", "")).grid(row=1, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(h22s_frame, text="Browse", command=self._browse_s_h22_xyz).grid(row=1, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(h22s_frame, text="Hessian").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(h22s_frame, width=60, textvariable=self._sv("s_h22_hessian", "")).grid(row=2, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(h22s_frame, text="Browse", command=self._browse_s_h22_hessian).grid(row=2, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(h22s_frame, text="Cubic 2-index").grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(h22s_frame, width=60, textvariable=self._sv("s_cubic_2idx", "")).grid(row=3, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(h22s_frame, text="Browse", command=self._browse_s_cubic_2idx).grid(row=3, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(h22s_frame, text="Cubic log").grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(h22s_frame, width=60, textvariable=self._sv("s_cubic_log", "")).grid(row=4, column=1, padx=4, sticky="we", pady=(4, 0))
        ttk.Button(h22s_frame, text="Browse", command=self._browse_s_cubic_log).grid(row=4, column=2, padx=(4, 0), pady=(4, 0))
        ttk.Label(
            h22s_frame,
            text=(
                "Standard sextic transforms use only centrifugal constants; these extra inputs are only for harmonic sextic analysis "
                "and for the H22-induced diagnostic. XYZ can be given either as a file path or pasted directly into the field. "
                "When used with a Hessian, XYZ and Hessian must refer to the same Cartesian orientation. "
                "The main low-cost route is geometry + Hessian, optionally completed by a cubic 2-index N x N matrix with entry (i,j)=phi_iij. "
                "If a Gaussian anharmonic log is also provided, the app uses it only to recover the genuine 3-index cubic remainder."
            ),
        ).grid(row=5, column=1, sticky="w", pady=(4, 0))
        h22s_frame.columnconfigure(1, weight=1)

        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(grid, text="Input sextic constants", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=7, sticky="w")

        for i, key in enumerate(SEXTIC_KEYS):
            lbl = ttk.Label(grid, text=SEXTIC_A_NAMES[i])
            lbl.grid(row=1, column=i, sticky="w")
            self.s_in_labels.append(lbl)
            ttk.Entry(grid, width=14, textvariable=self._sv(f"s_in_{key}", "")).grid(row=2, column=i, padx=3, pady=2)

        ttk.Label(grid, text="Output rep #1", font=("TkDefaultFont", 10, "bold")).grid(row=3, column=0, columnspan=7, sticky="w", pady=(10, 0))
        ttk.Label(grid, textvariable=self._sv("s_out_rep1", "II")).grid(row=3, column=2, sticky="w")
        for i, key in enumerate(SEXTIC_KEYS):
            lbl = ttk.Label(grid, text=SEXTIC_A_NAMES[i])
            lbl.grid(row=4, column=i, sticky="w")
            self.s_o1_labels.append(lbl)
            ttk.Entry(grid, width=14, textvariable=self._sv(f"s_out1_{key}", ""), state="readonly").grid(row=5, column=i, padx=3, pady=2)

        ttk.Label(grid, text="Output rep #2", font=("TkDefaultFont", 10, "bold")).grid(row=6, column=0, columnspan=7, sticky="w", pady=(10, 0))
        ttk.Label(grid, textvariable=self._sv("s_out_rep2", "III")).grid(row=6, column=2, sticky="w")
        for i, key in enumerate(SEXTIC_KEYS):
            lbl = ttk.Label(grid, text=SEXTIC_A_NAMES[i])
            lbl.grid(row=7, column=i, sticky="w")
            self.s_o2_labels.append(lbl)
            ttk.Entry(grid, width=14, textvariable=self._sv(f"s_out2_{key}", ""), state="readonly").grid(row=8, column=i, padx=3, pady=2)

        sbtn = ttk.Frame(parent)
        sbtn.pack(anchor="w", pady=(12, 6))
        ttk.Button(sbtn, text="Compute sextic transforms", command=self._run_sextic).pack(side=tk.LEFT)
        ttk.Button(sbtn, text="Compute harmonic/cubic sextic hierarchy", command=self._run_sextic_hierarchy).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(sbtn, text="Compute sextic H22 diagnostic", command=self._run_sextic_h22_from_fchk).pack(side=tk.LEFT, padx=(8, 0))

        self.s_report = tk.Text(parent, height=10, wrap="word")
        self.s_report.pack(fill=tk.BOTH, expand=True)
        self.vars["s_red_in"].trace_add("write", self._refresh_sextic_labels)
        self.vars["s_red_out"].trace_add("write", self._refresh_sextic_labels)
        self._refresh_sextic_labels()

    def _refresh_quartic_labels(self, *_: object) -> None:
        names = QUARTIC_A_NAMES if _norm_reduction(self.vars["q_red"].get()) == "A" else QUARTIC_S_NAMES
        for lbl, nm in zip(self.q_in_labels, names):
            lbl.configure(text=nm)
        for lbl, nm in zip(self.q_o1_labels, names):
            lbl.configure(text=nm)
        for lbl, nm in zip(self.q_o2_labels, names):
            lbl.configure(text=nm)

    def _refresh_sextic_labels(self, *_: object) -> None:
        in_names = SEXTIC_A_NAMES if _norm_reduction(self.vars["s_red_in"].get()) == "A" else SEXTIC_S_NAMES
        out_names = SEXTIC_A_NAMES if _norm_reduction(self.vars["s_red_out"].get()) == "A" else SEXTIC_S_NAMES
        for lbl, nm in zip(self.s_in_labels, in_names):
            lbl.configure(text=nm)
        for lbl, nm in zip(self.s_o1_labels, out_names):
            lbl.configure(text=nm)
        for lbl, nm in zip(self.s_o2_labels, out_names):
            lbl.configure(text=nm)

    def _browse_h22_fchk(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select formatted checkpoint for H22 diagnostic",
            filetypes=[("Gaussian fchk", "*.fchk"), ("All files", "*.*")],
        )
        if pth:
            self.vars["q_h22_fchk"].set(pth)

    def _browse_q_symm_xyz(self) -> None:
        pth = filedialog.askopenfilename(title="Select XYZ geometry for point-group assignment", filetypes=[("XYZ", "*.xyz"), ("All files", "*.*")])
        if pth:
            self.vars["q_symm_xyz"].set(pth)

    def _browse_h22_xyz(self) -> None:
        pth = filedialog.askopenfilename(title="Select XYZ geometry for H22 diagnostic", filetypes=[("XYZ", "*.xyz"), ("All files", "*.*")])
        if pth:
            self.vars["q_h22_xyz"].set(pth)

    def _browse_h22_hessian(self) -> None:
        pth = filedialog.askopenfilename(title="Select Cartesian Hessian for H22 diagnostic", filetypes=[("Text", "*.txt *.dat *.hess"), ("All files", "*.*")])
        if pth:
            self.vars["q_h22_hessian"].set(pth)

    def _browse_q_alpha_log(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select Gaussian anharmonic log for alpha parsing",
            filetypes=[("Gaussian log", "*.log *.out"), ("All files", "*.*")],
        )
        if pth:
            self.vars["q_alpha_log"].set(pth)

    def _browse_q_alpha_fchk(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select formatted checkpoint for planned alpha route",
            filetypes=[("Gaussian fchk", "*.fchk"), ("All files", "*.*")],
        )
        if pth:
            self.vars["q_alpha_fchk"].set(pth)

    def _browse_q_alpha_xyz(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select XYZ geometry for planned alpha route",
            filetypes=[("XYZ", "*.xyz"), ("All files", "*.*")],
        )
        if pth:
            self.vars["q_alpha_xyz"].set(pth)

    def _browse_q_alpha_hessian(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select Cartesian Hessian for planned alpha route",
            filetypes=[("Text", "*.txt *.dat *.hess"), ("All files", "*.*")],
        )
        if pth:
            self.vars["q_alpha_hessian"].set(pth)

    def _browse_q_alpha_cubic_2idx(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select 2-index cubic matrix for planned alpha route",
            filetypes=[("Text", "*.txt *.dat *.csv"), ("All files", "*.*")],
        )
        if pth:
            self.vars["q_alpha_cubic_2idx"].set(pth)

    def _browse_s_h22_fchk(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select formatted checkpoint for sextic H22 diagnostic",
            filetypes=[("Gaussian fchk", "*.fchk"), ("All files", "*.*")],
        )
        if pth:
            self.vars["s_h22_fchk"].set(pth)

    def _browse_s_symm_xyz(self) -> None:
        pth = filedialog.askopenfilename(title="Select XYZ geometry for point-group assignment", filetypes=[("XYZ", "*.xyz"), ("All files", "*.*")])
        if pth:
            self.vars["s_symm_xyz"].set(pth)

    def _browse_s_h22_xyz(self) -> None:
        pth = filedialog.askopenfilename(title="Select XYZ geometry for sextic H22 diagnostic", filetypes=[("XYZ", "*.xyz"), ("All files", "*.*")])
        if pth:
            self.vars["s_h22_xyz"].set(pth)

    def _browse_s_h22_hessian(self) -> None:
        pth = filedialog.askopenfilename(title="Select Cartesian Hessian for sextic H22 diagnostic", filetypes=[("Text", "*.txt *.dat *.hess"), ("All files", "*.*")])
        if pth:
            self.vars["s_h22_hessian"].set(pth)

    def _browse_s_cubic_log(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select Gaussian anharmonic log for sextic cubic analysis",
            filetypes=[("Gaussian log", "*.log *.out"), ("All files", "*.*")],
        )
        if pth:
            self.vars["s_cubic_log"].set(pth)

    def _browse_s_cubic_2idx(self) -> None:
        pth = filedialog.askopenfilename(
            title="Select 2-index cubic matrix for XYZ/Hessian sextic analysis",
            filetypes=[("Text", "*.txt *.dat *.csv"), ("All files", "*.*")],
        )
        if pth:
            self.vars["s_cubic_2idx"].set(pth)

    def _read_abc(self) -> tuple[float, float, float]:
        return _as_float(self.vars["A"].get()), _as_float(self.vars["B"].get()), _as_float(self.vars["C"].get())

    def _validate_units_quartic(self, A: float, B: float, C: float, D: np.ndarray) -> None:
        if min(abs(A), abs(B), abs(C)) < 1e-9:
            raise ValueError("Rotational constants A, B, C must be non-zero and in MHz.")
        if np.max(np.abs(D)) > 1.0e5:
            raise ValueError("Quartic constants look too large for kHz input. Please check units.")

    def _validate_units_sextic(self, A: float, B: float, C: float, H: np.ndarray) -> None:
        if min(abs(A), abs(B), abs(C)) < 1e-9:
            raise ValueError("Rotational constants A, B, C must be non-zero and in MHz.")
        if np.max(np.abs(H)) > 1.0e6:
            raise ValueError("Sextic constants look too large for kHz input. Please check units.")

    def _quartic_input(self) -> np.ndarray:
        return np.array([_as_float(self.vars[f"q_in_{nm}"].get()) for nm in QUARTIC_A_NAMES], dtype=float)

    def _sextic_input(self) -> np.ndarray:
        return np.array([_as_float(self.vars[f"s_in_{nm}"].get()) for nm in SEXTIC_KEYS], dtype=float)

    def _set_values(self, prefix: str, names: tuple[str, ...], values: np.ndarray) -> None:
        for nm, vv in zip(names, values):
            self.vars[f"{prefix}_{nm}"].set(f"{float(vv):.10g}")

    def _export_quartic_csv(self) -> None:
        if self.last_quartic_result is None:
            messagebox.showerror("Export error", "Run a quartic transform first.")
            return
        pth = filedialog.asksaveasfilename(
            title="Export quartic results (CSV)",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not pth:
            return
        r = self.last_quartic_result
        names = QUARTIC_A_NAMES if r["red"] == "A" else QUARTIC_S_NAMES
        lines = [
            "field,value",
            f"method,{r['method']}",
            f"input_rep,{r['rep_in']}",
            f"reduction,{r['red']}",
            f"A_MHz,{r['A']}",
            f"B_MHz,{r['B']}",
            f"C_MHz,{r['C']}",
        ]
        for nm, v in zip(names, r["D_in"]):
            lines.append(f"in_{nm},{float(v):.12g}")
        for nm, v in zip(names, r["D_out1"]):
            lines.append(f"{r['rep_out1']}_{nm},{float(v):.12g}")
        for nm, v in zip(names, r["D_out2"]):
            lines.append(f"{r['rep_out2']}_{nm},{float(v):.12g}")
        lines.extend(
            [
                f"{r['rep_in']}_to_{r['rep_out1']}_cond2,{r['m1']['cond2']:.6e}",
                f"{r['rep_in']}_to_{r['rep_out2']}_cond2,{r['m2']['cond2']:.6e}",
                f"{r['rep_out1']}_s111,{r['s111_out1']:.6e}",
                f"{r['rep_out2']}_s111,{r['s111_out2']:.6e}",
                f"{r['rep_out1']}_T_over_B,{r['tb_out1']:.6e}",
                f"{r['rep_out2']}_T_over_B,{r['tb_out2']:.6e}",
            ]
        )
        for tag in ("in", "out1", "out2"):
            spec = r[f"spec_{tag}"]
            red5 = r[f"red5_{tag}"]
            rep = r["rep_in"] if tag == "in" else r[f"rep_{tag}"]
            taup = spec["tauprime"]
            lines.extend(
                [
                    f"{rep}_lambda1,{spec['lambda'][0]:.12g}",
                    f"{rep}_lambda2,{spec['lambda'][1]:.12g}",
                    f"{rep}_lambda3,{spec['lambda'][2]:.12g}",
                    f"{rep}_I1_tauprime,{spec['I1']:.12g}",
                    f"{rep}_I2_tauprime,{spec['I2']:.12g}",
                    f"{rep}_I3_tauprime,{spec['I3']:.12g}",
                    f"{rep}_tauprime_aa,{taup[0,0]:.12g}",
                    f"{rep}_tauprime_bb,{taup[1,1]:.12g}",
                    f"{rep}_tauprime_cc,{taup[2,2]:.12g}",
                    f"{rep}_tauprime_ab,{taup[0,1]:.12g}",
                    f"{rep}_tauprime_ac,{taup[0,2]:.12g}",
                    f"{rep}_tauprime_bc,{taup[1,2]:.12g}",
                    f"{rep}_alpha_zyz,{red5['alpha_zyz']:.12g}",
                    f"{rep}_beta_zyz,{red5['beta_zyz']:.12g}",
                    f"{rep}_gamma_zyz_recovered,{red5['gamma_zyz_recovered']:.12g}",
                    f"{rep}_constraint_F_recovered,{red5['constraint_F_recovered']:.12g}",
                ]
            )
        with open(pth, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        messagebox.showinfo("Export", f"CSV exported:\n{pth}")

    def _export_quartic_latex(self) -> None:
        if self.last_quartic_result is None:
            messagebox.showerror("Export error", "Run a quartic transform first.")
            return
        pth = filedialog.asksaveasfilename(
            title="Export quartic results (LaTeX)",
            defaultextension=".tex",
            filetypes=[("LaTeX", "*.tex"), ("All files", "*.*")],
        )
        if not pth:
            return
        r = self.last_quartic_result
        names = QUARTIC_A_NAMES if r["red"] == "A" else QUARTIC_S_NAMES
        rows = []
        for i, nm in enumerate(names):
            rows.append(
                f"{nm} & {float(r['D_in'][i]):.6g} & {float(r['D_out1'][i]):.6g} & {float(r['D_out2'][i]):.6g} \\\\"
            )
        rows.append(f"s111 & -- & {r['s111_out1']:.3e} & {r['s111_out2']:.3e} \\\\")
        rows.append(f"T/B & -- & {r['tb_out1']:.3e} & {r['tb_out2']:.3e} \\\\")
        rows.append(f"$\\lambda_1(\\Tau')$ & {r['spec_in']['lambda'][0]:.6g} & {r['spec_out1']['lambda'][0]:.6g} & {r['spec_out2']['lambda'][0]:.6g} \\\\")
        rows.append(f"$\\lambda_2(\\Tau')$ & {r['spec_in']['lambda'][1]:.6g} & {r['spec_out1']['lambda'][1]:.6g} & {r['spec_out2']['lambda'][1]:.6g} \\\\")
        rows.append(f"$\\lambda_3(\\Tau')$ & {r['spec_in']['lambda'][2]:.6g} & {r['spec_out1']['lambda'][2]:.6g} & {r['spec_out2']['lambda'][2]:.6g} \\\\")
        rows.append(f"$\\alpha_{{ZYZ}}$ & {r['red5_in']['alpha_zyz']:.6g} & {r['red5_out1']['alpha_zyz']:.6g} & {r['red5_out2']['alpha_zyz']:.6g} \\\\")
        rows.append(f"$\\beta_{{ZYZ}}$ & {r['red5_in']['beta_zyz']:.6g} & {r['red5_out1']['beta_zyz']:.6g} & {r['red5_out2']['beta_zyz']:.6g} \\\\")
        rows.append(f"$\\gamma_{{rec}}$ & {r['red5_in']['gamma_zyz_recovered']:.6g} & {r['red5_out1']['gamma_zyz_recovered']:.6g} & {r['red5_out2']['gamma_zyz_recovered']:.6g} \\\\")
        tex = "\n".join(
            [
                r"\begin{table}[h]",
                r"\centering",
                rf"\caption{{Quartic transform ({r['rep_in']}\to {r['rep_out1']},{r['rep_out2']}; reduction {r['red']}; method {r['method']}).}}",  # noqa: E501
                r"\begin{tabular}{lccc}",
                r"\hline",
                rf"Constant & {r['rep_in']} & {r['rep_out1']} & {r['rep_out2']} \\",
                r"\hline",
                *rows,
                r"\hline",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
        with open(pth, "w", encoding="utf-8") as f:
            f.write(tex)
        messagebox.showinfo("Export", f"LaTeX exported:\n{pth}")

    def _run_quartic(self) -> None:
        try:
            A, B, C = self._read_abc()
            rotor_limit = classify_rotor_limit(np.array([A, B, C], dtype=float))
            if rotor_limit["is_special_limit"]:
                raise ValueError(
                    "Exact symmetric-top and linear limits do not use the asymmetric-top quartic representation transform. "
                    "Use harmonic input to compute the symmetry-adapted quartic constants directly."
                )
            rep_in = _norm_rep(self.vars["q_rep_in"].get())
            red = _norm_reduction(self.vars["q_red"].get())
            D_in = self._quartic_input()
            self._validate_units_quartic(A, B, C, D_in)

            rep_outs = [r for r in REPRESENTATIONS if r != rep_in]
            self.vars["q_out_rep1"].set(rep_outs[0])
            self.vars["q_out_rep2"].set(rep_outs[1])

            M_in = _M4(A, B, C, red)
            tau_in = np.linalg.pinv(M_in) @ D_in
            D1 = transform_quartic_tensor(D_in, A, B, C, rep_in, rep_outs[0], red)
            D2 = transform_quartic_tensor(D_in, A, B, C, rep_in, rep_outs[1], red)
            m_used = "tensor"
            A1, B1, C1 = _rotate_abc(A, B, C, rep_in, rep_outs[0])
            A2, B2, C2 = _rotate_abc(A, B, C, rep_in, rep_outs[1])
            M_out1 = _M4(A1, B1, C1, red)
            M_out2 = _M4(A2, B2, C2, red)
            tau1 = np.linalg.pinv(M_out1) @ D1
            tau2 = np.linalg.pinv(M_out2) @ D2
            spec_in = _quartic_spectral_invariants_from_tau(tau_in)
            spec1 = _quartic_spectral_invariants_from_tau(tau1)
            spec2 = _quartic_spectral_invariants_from_tau(tau2)
            red5_in = _quartic_reduced_3plus2_from_tau(tau_in, A, B, C)
            red5_1 = _quartic_reduced_3plus2_from_tau(tau1, A1, B1, C1)
            red5_2 = _quartic_reduced_3plus2_from_tau(tau2, A2, B2, C2)

            self._set_values("q_out1", QUARTIC_A_NAMES, D1)
            self._set_values("q_out2", QUARTIC_A_NAMES, D2)

            self.q_report.delete("1.0", tk.END)
            self.q_report.insert(tk.END, "Quartic transform completed.\n\n")
            self.q_report.insert(tk.END, f"Input: rep={rep_in}, reduction={red}, method={m_used}\n")
            self.q_report.insert(tk.END, f"Output #1: rep={rep_outs[0]}, reduction={red}\n")
            self.q_report.insert(tk.END, f"Output #2: rep={rep_outs[1]}, reduction={red}\n")
            xyz_symm = self.vars["q_symm_xyz"].get().strip()
            if xyz_symm:
                meta = _point_group_from_xyz_file(xyz_symm)
                self.q_report.insert(
                    tk.END,
                    "symmetry from XYZ: "
                    f"point group={meta['point_group']}, "
                    f"sigma={meta['rotational_symmetry_number']}, "
                    f"rotor class={meta['rotor_type_for_symmetry']}\n",
                )
            self.q_report.insert(
                tk.END,
                "Quartic transform uses the corrected pseudoinverse tensor route "
                "tau -> tau' -> T -> Watson constants.\n",
            )

            # Stability diagnostics for each linear transform.
            T1 = quartic_transform_matrix(A, B, C, rep_in, rep_outs[0], red, m_used)
            T2 = quartic_transform_matrix(A, B, C, rep_in, rep_outs[1], red, m_used)
            m1 = stability_metrics(T1, A, B, C)
            m2 = stability_metrics(T2, A, B, C)
            s1 = compute_s111(*_rotate_abc(A, B, C, rep_in, rep_outs[0]), D1, red)
            s2 = compute_s111(*_rotate_abc(A, B, C, rep_in, rep_outs[1]), D2, red)
            tb1 = compute_T_over_B(_rotate_abc(A, B, C, rep_in, rep_outs[0])[1], D1)
            tb2 = compute_T_over_B(_rotate_abc(A, B, C, rep_in, rep_outs[1])[1], D2)
            self.q_report.insert(tk.END, "\nStability diagnostics\n")
            self.q_report.insert(tk.END, f"{rep_in}->{rep_outs[0]}: cond2={m1['cond2']:.3e}, cond1={m1['cond1']:.3e}, condinf={m1['condinf']:.3e}\n")
            self.q_report.insert(tk.END, f"                sigma_min={m1['sigma_min']:.3e}, sigma_max={m1['sigma_max']:.3e}, cond2*eps={m1['amp_eps']:.3e}\n")
            self.q_report.insert(tk.END, f"                warning={stability_warning(m1['cond2'])}, s111={s1:.3e}, T/B={tb1:.3e}\n")
            self.q_report.insert(tk.END, f"{rep_in}->{rep_outs[1]}: cond2={m2['cond2']:.3e}, cond1={m2['cond1']:.3e}, condinf={m2['condinf']:.3e}\n")
            self.q_report.insert(tk.END, f"                sigma_min={m2['sigma_min']:.3e}, sigma_max={m2['sigma_max']:.3e}, cond2*eps={m2['amp_eps']:.3e}\n")
            self.q_report.insert(tk.END, f"                warning={stability_warning(m2['cond2'])}, s111={s2:.3e}, T/B={tb2:.3e}\n")
            self.q_report.insert(tk.END, f"Gaps (MHz): A-B={m1['A_minus_B']:.6f}, B-C={m1['B_minus_C']:.6f}, A-C={m1['A_minus_C']:.6f}\n")
            self.q_report.insert(tk.END, "\nTau' spectral invariants\n")
            for label, spec in (
                (rep_in, spec_in),
                (rep_outs[0], spec1),
                (rep_outs[1], spec2),
            ):
                self.q_report.insert(
                    tk.END,
                    f"{label}: lambda=({spec['lambda'][0]:.6g}, {spec['lambda'][1]:.6g}, {spec['lambda'][2]:.6g}), "
                    f"I1={spec['I1']:.6g}, I2={spec['I2']:.6g}, I3={spec['I3']:.6g}\n",
                )
                tp = spec["tauprime"]
                self.q_report.insert(
                    tk.END,
                    f"     Tau'=[[{tp[0,0]:.6g}, {tp[0,1]:.6g}, {tp[0,2]:.6g}], "
                    f"[{tp[0,1]:.6g}, {tp[1,1]:.6g}, {tp[1,2]:.6g}], "
                    f"[{tp[0,2]:.6g}, {tp[1,2]:.6g}, {tp[2,2]:.6g}]]\n",
                )
            self.q_report.insert(tk.END, "\nReduced 3+2 coordinates on pseudoinverse slice\n")
            for label, red5 in (
                (rep_in, red5_in),
                (rep_outs[0], red5_1),
                (rep_outs[1], red5_2),
            ):
                self.q_report.insert(
                    tk.END,
                    f"{label}: lambda=({red5['lambda1']:.6g}, {red5['lambda2']:.6g}, {red5['lambda3']:.6g}), "
                    f"alpha={red5['alpha_zyz']:.6g}, beta={red5['beta_zyz']:.6g}, "
                    f"gamma_rec={red5['gamma_zyz_recovered']:.6g}, F={red5['constraint_F_recovered']:.3e}\n",
                )

            # Internal round-trip verification of the exact tensor transform.
            D1_back = transform_quartic_tensor(D1, A1, B1, C1, rep_outs[0], rep_in, red)
            D2_back = transform_quartic_tensor(D2, A2, B2, C2, rep_outs[1], rep_in, red)
            rt1 = float(np.max(np.abs(D1_back - D_in)))
            rt2 = float(np.max(np.abs(D2_back - D_in)))
            self.q_report.insert(
                tk.END,
                f"Tensor round-trip max diff: {rt1:.3e} via {rep_outs[0]}, {rt2:.3e} via {rep_outs[1]}\n",
            )

            self.last_quartic_result = {
                "A": A,
                "B": B,
                "C": C,
                "rep_in": rep_in,
                "red": red,
                "method": m_used,
                "rep_out1": rep_outs[0],
                "rep_out2": rep_outs[1],
                "D_in": D_in.copy(),
                "D_out1": D1.copy(),
                "D_out2": D2.copy(),
                "m1": m1,
                "m2": m2,
                "s111_out1": s1,
                "s111_out2": s2,
                "tb_out1": tb1,
                "tb_out2": tb2,
                "tau_in": tau_in.copy(),
                "tau_out1": tau1.copy(),
                "tau_out2": tau2.copy(),
                "spec_in": spec_in,
                "spec_out1": spec1,
                "spec_out2": spec2,
                "red5_in": red5_in,
                "red5_out1": red5_1,
                "red5_out2": red5_2,
            }

        except Exception as exc:
            messagebox.showerror("Quartic transform error", str(exc))

    def _run_quartic_legacy_check(self) -> None:
        try:
            A, B, C = self._read_abc()
            rotor_limit = classify_rotor_limit(np.array([A, B, C], dtype=float))
            if rotor_limit["is_special_limit"]:
                raise ValueError("Legacy asymmetric-top quartic transforms are not defined for symmetric-top or linear limits.")
            rep_in = _norm_rep(self.vars["q_rep_in"].get())
            red = _norm_reduction(self.vars["q_red"].get())
            D_in = self._quartic_input()
            self._validate_units_quartic(A, B, C, D_in)

            rep_outs = [r for r in REPRESENTATIONS if r != rep_in]
            D1 = transform_quartic_tensor(D_in, A, B, C, rep_in, rep_outs[0], red)
            D2 = transform_quartic_tensor(D_in, A, B, C, rep_in, rep_outs[1], red)
            R1 = transform_quartic_reference(D_in, A, B, C, rep_in, rep_outs[0], red)
            R2 = transform_quartic_reference(D_in, A, B, C, rep_in, rep_outs[1], red)

            self.q_report.insert(tk.END, "\nLegacy quartic comparison\n")
            if red == "A":
                e1 = float(np.max(np.abs(D1[:4] - R1[:4])))
                e2 = float(np.max(np.abs(D2[:4] - R2[:4])))
                dk1 = float(D1[4] - R1[4])
                dk2 = float(D2[4] - R2[4])
                self.q_report.insert(
                    tk.END,
                    f"Tensor-legacy max diff on AJ/AJK/AK/dJ: {e1:.3e} (out1), {e2:.3e} (out2)\n",
                )
                self.q_report.insert(
                    tk.END,
                    f"Tensor-legacy dK difference: {dk1:.6e} (out1), {dk2:.6e} (out2)\n",
                )
                self.q_report.insert(
                    tk.END,
                    "Interpretation: quartic representation transform is invariant on the first four "
                    "constants; any residual discrepancy is confined to dK/deltaK.\n",
                )
            else:
                e1 = float(np.max(np.abs(D1 - R1)))
                e2 = float(np.max(np.abs(D2 - R2)))
                self.q_report.insert(
                    tk.END,
                    f"S-reduction legacy backend is not independent in this app; comparison reuses the tensor route "
                    f"({e1:.3e}, {e2:.3e}).\n",
                )
        except Exception as exc:
            messagebox.showerror("Quartic legacy check error", str(exc))

    def _run_h22_from_fchk(self) -> None:
        try:
            rep = _norm_rep(self.vars["q_rep_in"].get())
            red = _norm_reduction(self.vars["q_red"].get())
            model, source = _build_harmonic_model_from_inputs(
                rep,
                fchk_path=self.vars["q_h22_fchk"].get().strip(),
                xyz_path=self.vars["q_h22_xyz"].get().strip(),
                hessian_path=self.vars["q_h22_hessian"].get().strip(),
            )
            from_path = source
            fchk_tmp = self.vars["q_h22_fchk"].get().strip()
            if fchk_tmp:
                res = _h22_from_fchk(fchk_tmp, rep, red)
            else:
                omega_au = tuple(sp.Float(abs(x) / AU_FREQ_TO_CMINV) for x in model.vib_freq_cm)
                n_modes = model.dInv_au.shape[2]
                mu1 = sp.MutableDenseNDimArray(
                    [sp.Float(model.dInv_au[a, b, k]) for a in range(3) for b in range(3) for k in range(n_modes)],
                    (3, 3, n_modes),
                )
                mu2 = _sympy_modepair_tensor(np.asarray(model.d2Inv_au, dtype=float))
                intrinsic = _sympy_modepair_tensor(np.asarray(model.d2Inv_intrinsic_au, dtype=float))
                inertia0 = _sympy_rank2_tensor(np.asarray(model.i_tensor_au, dtype=float))
                tau_h22 = channel_h22(mu2, omega_au, sp.Integer(1))
                h22_decomp = channel_h22_from_mu1_intrinsic(mu1, intrinsic, inertia0, omega_au, sp.Integer(1))
                sigma, sigma1 = _sigma_values_quartic(*[float(x) for x in model.abc_mhz])
                taup = _gaussian_tauprime_from_compressed_tau(tau_h22)
                tmat = gaussian_t_from_tauprime(taup)
                watson_a = {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in gaussian_asymmetric_a_from_t(tmat, sp.Float(sigma)).items()}
                watson_s = {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in gaussian_symmetric_from_t(tmat, sp.Float(sigma1)).items()}
                decomp_khz: dict[str, dict[str, float]] = {}
                decomp_tau_khz: dict[str, dict[str, float]] = {}
                for name, tau in h22_decomp.items():
                    taup_piece = _gaussian_tauprime_from_compressed_tau(tau)
                    tmat_piece = gaussian_t_from_tauprime(taup_piece)
                    piece = gaussian_asymmetric_a_from_t(tmat_piece, sp.Float(sigma)) if red == "A" else gaussian_symmetric_from_t(tmat_piece, sp.Float(sigma1))
                    decomp_khz[name] = {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in piece.items()}
                    decomp_tau_khz[name] = {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in tau.items()}
                total = watson_a if red == "A" else watson_s
                res = {
                    "representation": rep,
                    "reduction": red,
                    "abc_mhz": tuple(float(x) for x in model.abc_mhz),
                    "h22_total_khz": total,
                    "h22_decomposition_khz": decomp_khz,
                    "h22_tau_total_khz": {k: float(CMINV_TO_MHZ * 1000.0 * sp.N(v)) for k, v in tau_h22.items()},
                    "h22_tau_decomposition_khz": decomp_tau_khz,
                    "maxabs_total_khz": float(max(abs(v) for v in total.values())),
                }
            names = ("DJ", "DJK", "DK", "dJ", "dK") if red == "A" else ("DJ", "DJK", "DK", "d1", "d2")

            self.q_report.insert(tk.END, "\nH22 diagnostic from harmonic input\n")
            self.q_report.insert(tk.END, f"source={from_path}\n")
            abc = res["abc_mhz"]
            self.vars["A"].set(f"{abc[0]:.10g}")
            self.vars["B"].set(f"{abc[1]:.10g}")
            self.vars["C"].set(f"{abc[2]:.10g}")
            self.q_report.insert(
                tk.END,
                f"model ABC (MHz)=({abc[0]:.6f}, {abc[1]:.6f}, {abc[2]:.6f}), rep={res['representation']}, reduction={res['reduction']}\n",
            )
            _append_point_group_report(self.q_report, model)
            self.q_report.insert(
                tk.END,
                f"harmonic frequencies (cm^-1): {_format_freqs_cm(model.vib_freq_cm)}\n",
            )
            self.q_report.insert(
                tk.END,
                "geometry and Hessian were reoriented internally to the selected input representation before building the harmonic model.\n",
            )
            self.q_report.insert(tk.END, f"max|H22|={res['maxabs_total_khz']:.6g} kHz\n")
            total = res["h22_total_khz"]
            rotor_limit = classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
            if rotor_limit["is_special_limit"]:
                tau_total = res.get("h22_tau_total_khz")
                special = None if tau_total is None else project_special_quartic_constants(
                    tau_total,
                    np.asarray(model.abc_mhz, dtype=float),
                    np.asarray(model.moments_amu_a2, dtype=float),
                )
                self.q_report.insert(tk.END, f"H22 symmetry-adapted projection for {rotor_limit['kind']}\n")
                if special is not None:
                    self.q_report.insert(
                        tk.END,
                        ", ".join(f"{nm}={float(val):.6g}" for nm, val in special["quartic_mhz"].items()) + " kHz\n",
                    )
                    if rotor_limit["kind"] == "linear":
                        ltype = linear_ltype_terms(model, quartic_special=special)
                        _append_linear_ltype_report(
                            self.q_report,
                            ltype,
                            title="Linear-molecule l-type doubling from H22 quartic projection",
                        )
                else:
                    self.q_report.insert(tk.END, "Direct special-limit H22 projection is available from the geometry+Hessian branch.\n")
            else:
                self.q_report.insert(
                    tk.END,
                    "H22 Watson contribution (kHz): "
                    + ", ".join(f"{nm}={float(total[nm]):.6g}" for nm in names)
                    + "\n",
                )
                self.q_report.insert(tk.END, "H22 decomposition in Watson coordinates (kHz)\n")
                for key in ("bilinear_bilinear", "cross", "intrinsic_intrinsic", "total"):
                    part = res["h22_decomposition_khz"][key]
                    self.q_report.insert(
                        tk.END,
                        f"  {key}: " + ", ".join(f"{nm}={float(part[nm]):.6g}" for nm in names) + "\n",
                    )
            self.q_report.insert(
                tk.END,
                "Interpretation: this is the first quartic post-standard diagnostic from the harmonic backend.\n",
            )
        except Exception as exc:
            messagebox.showerror("H22 diagnostic error", str(exc))

    def _run_alpha_parser(self) -> None:
        try:
            log_path = self.vars["q_alpha_log"].get().strip()
            if not log_path:
                raise ValueError("Select a Gaussian anharmonic log first.")
            bench = parse_gaussian_alpha_data(log_path)
            excluded = _parse_mode_selection(self.vars["q_alpha_excluded"].get())
            mode_indices = np.asarray(bench.mode_indices, dtype=int)
            n_modes = mode_indices.size
            invalid = sorted(idx for idx in excluded if idx < 1 or idx > n_modes)
            if invalid:
                raise ValueError(f"Excluded mode indices out of range: {invalid}")

            keep_mask = np.array([idx not in excluded for idx in mode_indices], dtype=bool)
            total_mhz = np.sum(np.asarray(bench.alpha_mhz, dtype=float), axis=0)
            kept_mhz = np.sum(np.asarray(bench.alpha_mhz, dtype=float)[keep_mask], axis=0) if np.any(keep_mask) else np.zeros(3, dtype=float)
            removed_mhz = total_mhz - kept_mhz
            total_cm = np.sum(np.asarray(bench.alpha_cm, dtype=float), axis=0)
            kept_cm = np.sum(np.asarray(bench.alpha_cm, dtype=float)[keep_mask], axis=0) if np.any(keep_mask) else np.zeros(3, dtype=float)
            removed_cm = total_cm - kept_cm

            freqs = np.full(n_modes, np.nan, dtype=float)
            pg = None
            try:
                harmonic = parse_gaussian_harmonic_data(log_path).reordered_to_anharmonic()
                parsed_freqs = np.asarray(harmonic.frequencies_cm, dtype=float)
                if parsed_freqs.size == n_modes:
                    freqs = parsed_freqs
                symbols = symbols_from_atomic_numbers(harmonic.atomic_numbers)
                pg = point_group_from_geometry(
                    np.asarray(harmonic.coords_std_ang, dtype=float),
                    symbols,
                    rotor_type_for_symmetry(harmonic.rot_ghz),
                )
            except Exception:
                harmonic = None

            self.q_report.insert(tk.END, "\nGaussian alpha parser / mode filtering\n")
            self.q_report.insert(tk.END, f"log={log_path}\n")
            if pg is not None:
                self.q_report.insert(
                    tk.END,
                    f"point group={pg['point_group']}, sigma={pg['rotational_symmetry_number']}, "
                    f"rotor class={pg['rotor_type_for_symmetry']}\n",
                )
            else:
                self.q_report.insert(tk.END, "point group: not available from this log block\n")
            self.q_report.insert(
                tk.END,
                "alpha axes="
                f"({bench.axis_labels[0]}, {bench.axis_labels[1]}, {bench.axis_labels[2]})\n",
            )
            self.q_report.insert(
                tk.END,
                "excluded modes="
                + (", ".join(str(x) for x in sorted(excluded)) if excluded else "(none)")
                + "\n",
            )
            self.q_report.insert(
                tk.END,
                f"total alpha sum (MHz)=({total_mhz[0]:.6f}, {total_mhz[1]:.6f}, {total_mhz[2]:.6f})\n",
            )
            self.q_report.insert(
                tk.END,
                f"kept  alpha sum (MHz)=({kept_mhz[0]:.6f}, {kept_mhz[1]:.6f}, {kept_mhz[2]:.6f})\n",
            )
            self.q_report.insert(
                tk.END,
                f"removed contribution (MHz)=({removed_mhz[0]:.6f}, {removed_mhz[1]:.6f}, {removed_mhz[2]:.6f})\n",
            )
            self.q_report.insert(
                tk.END,
                f"total alpha sum (cm^-1)=({total_cm[0]:.6f}, {total_cm[1]:.6f}, {total_cm[2]:.6f})\n",
            )
            self.q_report.insert(
                tk.END,
                f"kept  alpha sum (cm^-1)=({kept_cm[0]:.6f}, {kept_cm[1]:.6f}, {kept_cm[2]:.6f})\n",
            )
            self.q_report.insert(
                tk.END,
                f"removed contribution (cm^-1)=({removed_cm[0]:.6f}, {removed_cm[1]:.6f}, {removed_cm[2]:.6f})\n",
            )
            self.q_report.insert(tk.END, "\nMode-resolved alpha contributions (MHz)\n")
            for i, mode_idx in enumerate(mode_indices):
                status = "excluded" if mode_idx in excluded else "kept"
                freq_text = f"{freqs[i]:.3f}" if np.isfinite(freqs[i]) else "n/a"
                row = bench.alpha_mhz[i]
                self.q_report.insert(
                    tk.END,
                    f"Q({mode_idx:>3d})  w={freq_text:>10} cm^-1  "
                    f"{bench.axis_labels[0]}={row[0]:>12.5f}  "
                    f"{bench.axis_labels[1]}={row[1]:>12.5f}  "
                    f"{bench.axis_labels[2]}={row[2]:>12.5f}  [{status}]\n",
                )
        except Exception as exc:
            messagebox.showerror("Alpha parser error", str(exc))

    def _prepare_alpha_inputs(self) -> None:
        try:
            rep = _norm_rep(self.vars["q_rep_in"].get())
            cubic_2idx = self.vars["q_alpha_cubic_2idx"].get().strip()
            if not cubic_2idx:
                raise ValueError("Provide the semi-diagonal cubic 2-index matrix for alpha.")
            model, source = _build_harmonic_model_from_inputs(
                rep,
                fchk_path=self.vars["q_alpha_fchk"].get().strip(),
                xyz_path=self.vars["q_alpha_xyz"].get().strip(),
                hessian_path=self.vars["q_alpha_hessian"].get().strip(),
            )
            n_modes = int(np.abs(model.vib_freq_cm).size)
            mat = read_cubic_two_index_matrix(cubic_2idx, n_modes)
            excluded = _parse_mode_selection(self.vars["q_alpha_excluded"].get())
            invalid = sorted(idx for idx in excluded if idx < 1 or idx > n_modes)
            if invalid:
                raise ValueError(f"Excluded mode indices out of range: {invalid}")
            keep_modes = [i for i in range(1, n_modes + 1) if i not in excluded]
            alpha = alpha_matrix_from_cubic_two_index_cm(model, mat, excluded_modes=excluded)
            total_cm = np.asarray(alpha["alpha_total_cm_abc"], dtype=float)
            total_mhz = total_cm * CMINV_TO_MHZ
            sum_total_cm = np.sum(total_cm, axis=0)
            sum_total_mhz = np.sum(total_mhz, axis=0)
            sum_cor_cm = np.sum(np.asarray(alpha["alpha_coriolis_cm_abc"], dtype=float), axis=0)
            sum_inertia_cm = np.sum(np.asarray(alpha["alpha_inertia_cm_abc"], dtype=float), axis=0)
            sum_anh_cm = np.sum(np.asarray(alpha["alpha_anharmonic_cm_abc"], dtype=float), axis=0)

            self.q_report.insert(tk.END, "\nAlpha from harmonic input + semi-diagonal cubic data\n")
            self.q_report.insert(tk.END, f"source={source}\n")
            self.q_report.insert(tk.END, f"cubic_2index={cubic_2idx}\n")
            _append_point_group_report(self.q_report, model)
            self.q_report.insert(
                tk.END,
                f"model ABC (MHz)=({model.abc_mhz[0]:.6f}, {model.abc_mhz[1]:.6f}, {model.abc_mhz[2]:.6f}), "
                f"representation={rep}\n",
            )
            rotor_limit = alpha.get("rotor_limit")
            if rotor_limit is not None:
                self.q_report.insert(tk.END, f"rotor limit={rotor_limit['kind']}\n")
            proj_strategy = alpha.get("projection_strategy")
            if proj_strategy:
                self.q_report.insert(tk.END, f"projection strategy={proj_strategy}\n")
            deg_pairs = alpha.get("degenerate_mode_pairs") or []
            if deg_pairs:
                self.q_report.insert(
                    tk.END,
                    "detected near-degenerate pairs="
                    + ", ".join(f"({i + 1},{j + 1})" for i, j in deg_pairs)
                    + "\n",
                )
            deg_meta = alpha.get("degenerate_mode_metadata") or []
            if deg_meta:
                self.q_report.insert(tk.END, "degenerate-mode metadata\n")
                for meta in deg_meta:
                    self.q_report.insert(
                        tk.END,
                        f"  pair={meta['pair_1based']}  w={meta['freq_cm']:.3f} cm^-1  "
                        f"dominant zeta axis={meta['dominant_coriolis_axis']}  "
                        f"|zeta|={meta['dominant_coriolis_abs']:.6f}\n",
                    )
            lsdgnm_like = alpha.get("lsdgnm_like")
            if lsdgnm_like is not None:
                self.q_report.insert(
                    tk.END,
                    f"LsDgNM-like summary: ntotnm={lsdgnm_like['ntotnm']}  ndeg_pairs={lsdgnm_like['ndeg_pairs']}\n",
                )
                for row in lsdgnm_like["mode_table"]:
                    if row["degeneracy_order"] == 1:
                        continue
                    self.q_report.insert(
                        tk.END,
                        f"  mode {row['mode_1based']:>3d}: code={row['lsdgnm_code']:>4d}  "
                        f"role={row['pair_role']:<8s}  ref={row['reference_mode_1based']}  "
                        f"partner={row['partner_mode_1based']}  "
                        f"dom-axis={row['dominant_coriolis_axis']}\n",
                    )
            canonical_pair_data = alpha.get("canonical_pair_data") or []
            if canonical_pair_data:
                self.q_report.insert(tk.END, "canonical pair data\n")
                for meta in canonical_pair_data:
                    sval = np.asarray(meta["feature_singular_values"], dtype=float)
                    self.q_report.insert(
                        tk.END,
                        f"  pair={meta['pair_1based']}  feature singular values=({sval[0]:.6e}, {sval[1]:.6e})\n",
                    )
            self.q_report.insert(
                tk.END,
                f"harmonic frequencies (cm^-1): {_format_freqs_cm(model.vib_freq_cm)}\n",
            )
            self.q_report.insert(
                tk.END,
                f"semi-diagonal cubic matrix size={mat.shape[0]}x{mat.shape[1]}\n",
            )
            self.q_report.insert(
                tk.END,
                "excluded modes="
                + (", ".join(str(x) for x in sorted(excluded)) if excluded else "(none)")
                + "\n",
            )
            self.q_report.insert(
                tk.END,
                "kept modes="
                + (", ".join(str(x) for x in keep_modes) if keep_modes else "(none)")
                + "\n",
            )
            self.q_report.insert(
                tk.END,
                f"sum alpha (cm^-1)=({sum_total_cm[0]:.6f}, {sum_total_cm[1]:.6f}, {sum_total_cm[2]:.6f})\n",
            )
            self.q_report.insert(
                tk.END,
                f"sum alpha (MHz)=({sum_total_mhz[0]:.6f}, {sum_total_mhz[1]:.6f}, {sum_total_mhz[2]:.6f})\n",
            )
            self.q_report.insert(
                tk.END,
                f"component sums (cm^-1): coriolis=({sum_cor_cm[0]:.6f}, {sum_cor_cm[1]:.6f}, {sum_cor_cm[2]:.6f}), "
                f"inertia=({sum_inertia_cm[0]:.6f}, {sum_inertia_cm[1]:.6f}, {sum_inertia_cm[2]:.6f}), "
                f"anharmonic=({sum_anh_cm[0]:.6f}, {sum_anh_cm[1]:.6f}, {sum_anh_cm[2]:.6f})\n",
            )
            if rotor_limit is not None and rotor_limit["is_special_limit"]:
                if rotor_limit["kind"] == "linear" and "alpha_linear_cm" in alpha:
                    alpha_lin_mhz = np.asarray(alpha["alpha_linear_cm"], dtype=float) * CMINV_TO_MHZ
                    self.q_report.insert(
                        tk.END,
                        f"sum symmetry-adapted linear alpha_perp (MHz)={float(np.sum(alpha_lin_mhz)):.6f}\n",
                    )
                elif "alpha_axial_cm" in alpha:
                    axial = alpha["alpha_axial_cm"]
                    par_mhz = np.asarray(axial["parallel"], dtype=float) * CMINV_TO_MHZ
                    perp_mhz = np.asarray(axial["perpendicular"], dtype=float) * CMINV_TO_MHZ
                    self.q_report.insert(
                        tk.END,
                        f"sum symmetry-adapted alpha_parallel/perpendicular (MHz)=({float(np.sum(par_mhz)):.6f}, {float(np.sum(perp_mhz)):.6f})\n",
                    )
            self.q_report.insert(tk.END, "\nMode-resolved alpha contributions from harmonic+cubic route (MHz)\n")
            axis_labels = ("A", "B", "C")
            for i in range(n_modes):
                freq_text = f"{float(model.vib_freq_cm[i]):.3f}"
                status = "excluded" if (i + 1) in excluded else "kept"
                row = total_mhz[i]
                self.q_report.insert(
                    tk.END,
                    f"Q({i + 1:>3d})  w={freq_text:>10} cm^-1  "
                    f"{axis_labels[0]}={row[0]:>12.5f}  "
                    f"{axis_labels[1]}={row[1]:>12.5f}  "
                    f"{axis_labels[2]}={row[2]:>12.5f}  [{status}]\n",
                )

            log_path = self.vars["q_alpha_log"].get().strip()
            if log_path:
                try:
                    bench = parse_gaussian_alpha_data(log_path)
                    bench_sum_mhz = np.sum(np.asarray(bench.alpha_mhz, dtype=float), axis=0)
                    self.q_report.insert(
                        tk.END,
                        "\nGaussian alpha benchmark from log (sum over printed mode rows, MHz)\n"
                        f"({bench_sum_mhz[0]:.6f}, {bench_sum_mhz[1]:.6f}, {bench_sum_mhz[2]:.6f})\n",
                    )
                    self.q_report.insert(
                        tk.END,
                        "difference on summed alpha (MHz)="
                        f"({sum_total_mhz[0] - bench_sum_mhz[0]:.6e}, "
                        f"{sum_total_mhz[1] - bench_sum_mhz[1]:.6e}, "
                        f"{sum_total_mhz[2] - bench_sum_mhz[2]:.6e})\n",
                    )
                except Exception as exc:
                    self.q_report.insert(tk.END, f"\nGaussian alpha comparison unavailable: {exc}\n")
        except Exception as exc:
            messagebox.showerror("Alpha harmonic route error", str(exc))

    def _load_quartics_from_harmonic_input(self) -> None:
        try:
            rep = _norm_rep(self.vars["q_rep_in"].get())
            red = _norm_reduction(self.vars["q_red"].get())
            model, source = _build_harmonic_model_from_inputs(
                rep,
                fchk_path=self.vars["q_h22_fchk"].get().strip(),
                xyz_path=self.vars["q_h22_xyz"].get().strip(),
                hessian_path=self.vars["q_h22_hessian"].get().strip(),
            )
            q2 = compute_order2_quartic(model)
            abc = [float(x) for x in model.abc_mhz]
            self.vars["A"].set(f"{abc[0]:.10g}")
            self.vars["B"].set(f"{abc[1]:.10g}")
            self.vars["C"].set(f"{abc[2]:.10g}")

            self.q_report.insert(tk.END, "\nStandard quartics loaded from harmonic input\n")
            self.q_report.insert(tk.END, f"source={source}\n")
            self.q_report.insert(
                tk.END,
                f"model ABC (MHz)=({abc[0]:.6f}, {abc[1]:.6f}, {abc[2]:.6f}), rep={rep}, reduction={red}\n",
            )
            _append_point_group_report(self.q_report, model)
            self.q_report.insert(tk.END, f"rotor limit={q2['rotor_limit']['kind']}\n")
            self.q_report.insert(
                tk.END,
                f"harmonic frequencies (cm^-1): {_format_freqs_cm(model.vib_freq_cm)}\n",
            )
            if q2["rotor_limit"]["is_special_limit"]:
                special = q2["special_quartic_projection"]
                if special is None:
                    raise ValueError("Could not build the symmetry-adapted quartic projection.")
                self.q_report.insert(
                    tk.END,
                    "Symmetry-adapted quartic constants: "
                    + ", ".join(f"{nm}={1000.0 * float(val):.10g} kHz" for nm, val in special["quartic_mhz"].items())
                    + "\n",
                )
                self.q_report.insert(
                    tk.END,
                    "Interpretation: for exact symmetric-top and linear limits the app reports the direct projection from the quartic tensor, not asymmetric-top Watson constants.\n",
                )
                if q2["rotor_limit"]["kind"] == "linear":
                    _append_linear_ltype_report(
                        self.q_report,
                        q2.get("linear_ltype_terms"),
                        title="Linear-molecule l-type doubling",
                    )
            else:
                if red == "A":
                    data_mhz = q2["watson_a_mhz"]
                    names = ("AJ", "AJK", "AK", "dJ", "dK")
                    source_keys = ("DJ", "DJK", "DK", "dJ", "dK")
                else:
                    data_mhz = q2["watson_s_mhz"]
                    names = ("DJ", "DJK", "DK", "d1", "d2")
                    source_keys = ("DJ", "DJK", "DK", "d1", "d2")

                for nm, sk in zip(names, source_keys):
                    self.vars[f"q_in_{nm}"].set(f"{1000.0 * float(data_mhz[sk]):.10g}")

                self.q_report.insert(
                    tk.END,
                    "Loaded quartic constants (kHz): "
                    + ", ".join(f"{nm}={self.vars[f'q_in_{nm}'].get()}" for nm in names)
                    + "\n",
                )
                self.q_report.insert(
                    tk.END,
                    "Interpretation: these are the standard harmonic H12H12 quartics computed from geometry + Hessian.\n",
                )
        except Exception as exc:
            messagebox.showerror("Load quartics error", str(exc))

    def _run_sextic(self) -> None:
        try:
            A, B, C = self._read_abc()
            rotor_limit = classify_rotor_limit(np.array([A, B, C], dtype=float))
            if rotor_limit["is_special_limit"]:
                raise ValueError(
                    "Exact symmetric-top and linear limits do not use the asymmetric-top sextic representation transform. "
                    "Use harmonic input to compute the symmetry-adapted sextic constants directly."
                )
            rep_in = _norm_rep(self.vars["s_rep_in"].get())
            red_in = _norm_reduction(self.vars["s_red_in"].get())
            red_out = _norm_reduction(self.vars["s_red_out"].get())
            H_in = self._sextic_input()
            self._validate_units_sextic(A, B, C, H_in)

            rep_outs = [r for r in REPRESENTATIONS if r != rep_in]
            self.vars["s_out_rep1"].set(rep_outs[0])
            self.vars["s_out_rep2"].set(rep_outs[1])

            H1 = transform_sextic_tensor(H_in, A, B, C, rep_in, rep_outs[0], red_in, red_out)
            H2 = transform_sextic_tensor(H_in, A, B, C, rep_in, rep_outs[1], red_in, red_out)

            self._set_values("s_out1", SEXTIC_KEYS, H1)
            self._set_values("s_out2", SEXTIC_KEYS, H2)

            A1, B1, C1 = _rotate_abc(A, B, C, rep_in, rep_outs[0])
            A2, B2, C2 = _rotate_abc(A, B, C, rep_in, rep_outs[1])
            H_back1 = transform_sextic_tensor(H1, A1, B1, C1, rep_outs[0], rep_in, red_out, red_in)
            H_back2 = transform_sextic_tensor(H2, A2, B2, C2, rep_outs[1], rep_in, red_out, red_in)
            err1 = float(np.max(np.abs(H_back1 - H_in)))
            err2 = float(np.max(np.abs(H_back2 - H_in)))
            phys_in = _sextic_physical_subspace_residual(H_in, rep_in, red_in, A, B, C)
            phys1 = _sextic_physical_subspace_residual(H1, rep_outs[0], red_out, A1, B1, C1)
            phys2 = _sextic_physical_subspace_residual(H2, rep_outs[1], red_out, A2, B2, C2)
            dec_in = _sextic_decomposition(H_in, rep_in, red_in, A, B, C)
            dec1 = _sextic_decomposition(H1, rep_outs[0], red_out, A1, B1, C1)
            dec2 = _sextic_decomposition(H2, rep_outs[1], red_out, A2, B2, C2)
            cm1 = _sextic_condition_metrics(rep_in, rep_outs[0])
            cm2 = _sextic_condition_metrics(rep_in, rep_outs[1])

            self.s_report.delete("1.0", tk.END)
            self.s_report.insert(tk.END, "Sextic tensor transform completed.\n\n")
            self.s_report.insert(tk.END, f"Input: rep={rep_in}, reduction={red_in}\n")
            self.s_report.insert(tk.END, f"Output #1: rep={rep_outs[0]}, reduction={red_out}\n")
            self.s_report.insert(tk.END, f"Output #2: rep={rep_outs[1]}, reduction={red_out}\n")
            xyz_symm = self.vars["s_symm_xyz"].get().strip()
            if xyz_symm:
                meta = _point_group_from_xyz_file(xyz_symm)
                self.s_report.insert(
                    tk.END,
                    "symmetry from XYZ: "
                    f"point group={meta['point_group']}, "
                    f"sigma={meta['rotational_symmetry_number']}, "
                    f"rotor class={meta['rotor_type_for_symmetry']}\n",
                )
            self.s_report.insert(
                tk.END,
                "Sextic transform uses the validated 5D invariant-subspace route with explicit A<->S conversion.\n"
                "The canonical decomposition is H_S = B_rep sigma + r, with five physical coordinates sigma "
                "and a two-dimensional residual r in the orthogonal complement. "
                "The same physical sector may also be viewed as harmonic 1+4 and, "
                "for cyclic representation changes, as 1+1+1+2.\n",
            )
            self.s_report.insert(tk.END, f"\nPhysical-subspace residuals: in={phys_in:.3e}, out1={phys1:.3e}, out2={phys2:.3e}\n")
            self.s_report.insert(
                tk.END,
                f"Round-trip check ({rep_in}->{rep_outs[0]}->{rep_in}) max error: {err1:.3e}\n",
            )
            self.s_report.insert(
                tk.END,
                f"Round-trip check ({rep_in}->{rep_outs[1]}->{rep_in}) max error: {err2:.3e}\n",
            )
            self.s_report.insert(
                tk.END,
                "For physically consistent sextic constants the round-trip closes at numerical precision; "
                "large residuals indicate that the input does not lie on the modeled sextic physical subspace.\n",
            )
            self.s_report.insert(tk.END, "\nSextic conditioning diagnostics on the physical subspace\n")
            for rep_to, cm in ((rep_outs[0], cm1), (rep_outs[1], cm2)):
                self.s_report.insert(
                    tk.END,
                    f"{rep_in}->{rep_to}: cond2(T_phys)={cm['T_phys_cond2_nz']:.3e}, "
                    f"sigma_min={cm['T_phys_sigma_min_nz']:.3e}, sigma_max={cm['T_phys_sigma_max_nz']:.3e}, "
                    f"cond2*eps={cm['T_phys_amp_eps_nz']:.3e}\n",
                )
                self.s_report.insert(
                    tk.END,
                    f"              cond2(B_{rep_in})={cm['B_from_cond2_nz']:.3e}, "
                    f"cond2(B_{rep_to})={cm['B_to_cond2_nz']:.3e}, "
                    f"warning={stability_warning(cm['T_phys_cond2_nz'])}\n",
                )
            self.s_report.insert(tk.END, "\nSextic 5+2 decomposition in canonical S-subspace\n")
            for label, dec in (
                (rep_in, dec_in),
                (rep_outs[0], dec1),
                (rep_outs[1], dec2),
            ):
                sigma = dec["sigma"]
                gauge = dec["gauge_coords"]
                self.s_report.insert(
                    tk.END,
                    f"{label}: sigma=("
                    f"{sigma[0]:.6g}, {sigma[1]:.6g}, {sigma[2]:.6g}, {sigma[3]:.6g}, {sigma[4]:.6g}), "
                    f"g=({gauge[0]:.6g}, {gauge[1]:.6g}), "
                    f"||r||2={dec['residual_norm2']:.3e}, max|r|={dec['residual_maxabs']:.3e}\n",
                )

        except Exception as exc:
            messagebox.showerror("Sextic transform error", str(exc))

    def _run_sextic_h22_from_fchk(self) -> None:
        try:
            rep = _norm_rep(self.vars["s_rep_in"].get())
            model, source = _build_harmonic_model_from_inputs(
                rep,
                fchk_path=self.vars["s_h22_fchk"].get().strip(),
                xyz_path=self.vars["s_h22_xyz"].get().strip(),
                hessian_path=self.vars["s_h22_hessian"].get().strip(),
            )
            axes = {"a": 0, "b": 1, "c": 2}
            cand = sextic_h22_linear_candidate_hz(model, axes)
            comps = ("aaa", "aab", "aac", "abb", "abc", "acc", "bbb", "bbc", "bcc", "ccc")
            max_key = max(comps, key=lambda k: abs(cand[k]["linear_hz"]))
            rotor_limit = classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
            self.s_report.insert(tk.END, "\nSextic H22-linear diagnostic from harmonic input\n")
            self.s_report.insert(tk.END, f"source={source}\n")
            self.s_report.insert(
                tk.END,
                f"model ABC (MHz)=({float(model.abc_mhz[0]):.6f}, {float(model.abc_mhz[1]):.6f}, {float(model.abc_mhz[2]):.6f}), rep={rep}\n",
            )
            _append_point_group_report(self.s_report, model)
            self.vars["A"].set(f"{float(model.abc_mhz[0]):.10g}")
            self.vars["B"].set(f"{float(model.abc_mhz[1]):.10g}")
            self.vars["C"].set(f"{float(model.abc_mhz[2]):.10g}")
            self.s_report.insert(
                tk.END,
                f"harmonic frequencies (cm^-1): {_format_freqs_cm(model.vib_freq_cm)}\n",
            )
            self.s_report.insert(
                tk.END,
                "geometry and Hessian were reoriented internally to the selected input representation before building the harmonic model.\n",
            )
            self.s_report.insert(
                tk.END,
                f"largest linear component: {max_key} = {cand[max_key]['linear_hz']:.6g} Hz\n",
            )
            self.s_report.insert(tk.END, "component-wise linear response (Hz)\n")
            for key in comps:
                self.s_report.insert(
                    tk.END,
                    f"  {key}: base={cand[key]['base_hz']:.6g}, linear={cand[key]['linear_hz']:.6g}, "
                    f"quadratic={cand[key]['quadratic_hz']:.6g}, delta={cand[key]['delta_total_hz']:.6g}\n",
                )
            if rotor_limit["is_special_limit"]:
                linear_cart = {key: float(cand[key]["linear_hz"]) for key in comps}
                special = project_special_sextic_constants(linear_cart, np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
                if special is not None:
                    self.s_report.insert(
                        tk.END,
                        "Symmetry-adapted sextic H22-linear projection: "
                        + ", ".join(f"{nm}={val:.6g} Hz" for nm, val in special["sextic_hz"].items())
                        + "\n",
                    )
                if rotor_limit["kind"] == "linear":
                    h_special = special
                    q2 = compute_order2_quartic(model)
                    ltype = linear_ltype_terms(
                        model,
                        quartic_special=q2.get("special_quartic_projection"),
                        sextic_special=h_special,
                    )
                    _append_linear_ltype_report(
                        self.s_report,
                        ltype,
                        title="Linear-molecule l-type doubling from H22-linear sextic response",
                    )
            self.s_report.insert(
                tk.END,
                "Interpretation: this is the first sextic post-standard diagnostic induced linearly by tau(H22).\n",
            )
        except Exception as exc:
            messagebox.showerror("Sextic H22 diagnostic error", str(exc))

    def _run_sextic_hierarchy(self) -> None:
        try:
            rep = _norm_rep(self.vars["s_rep_in"].get())
            model, source = _build_harmonic_model_from_inputs(
                rep,
                fchk_path=self.vars["s_h22_fchk"].get().strip(),
                xyz_path=self.vars["s_h22_xyz"].get().strip(),
                hessian_path=self.vars["s_h22_hessian"].get().strip(),
            )
            axes = {"a": 0, "b": 1, "c": 2}
            cubic_log = self.vars["s_cubic_log"].get().strip()
            cubic_2idx = self.vars["s_cubic_2idx"].get().strip()
            phi3 = None
            cubic_source = "none (geometry-only sextic level)"
            if cubic_2idx:
                mat = read_cubic_two_index_matrix(cubic_2idx, np.abs(model.vib_freq_cm).size)
                phi3 = expand_cubic_two_index_matrix(mat)
                cubic_source = cubic_2idx + " (2-index only)"
            if cubic_log:
                anh = parse_gaussian_anharmonic_force_data(cubic_log)
                _mapping, phi3_full, _raw = align_gaussian_cubic_force_constants(anh, np.abs(model.vib_freq_cm))
                if phi3 is None:
                    phi3 = phi3_full
                    cubic_source = cubic_log + " (full cubic)"
                else:
                    # Complete the 2-index input with the genuine 3-index remainder from the full log.
                    _phi3_sd_full, phi3_3ind = split_cubic_force_constants(phi3_full)
                    phi3 = phi3 + phi3_3ind
                    cubic_source = f"{cubic_2idx} + 3-index remainder from {cubic_log}"

            levels = sextic_cubic_hierarchy_hz(model, phi3, axes)
            comps = ("aaa", "aab", "aac", "abb", "abc", "acc", "bbb", "bbc", "bcc", "ccc")
            max_geom = max(comps, key=lambda k: abs(levels[k]["geometry_hz"]))
            max_full = max(comps, key=lambda k: abs(levels[k]["total_full_hz"]))
            rotor_limit = classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))

            self.s_report.insert(tk.END, "\nSextic harmonic/cubic hierarchy\n")
            self.s_report.insert(tk.END, f"harmonic source={source}\n")
            self.s_report.insert(tk.END, f"cubic source={cubic_source}\n")
            self.s_report.insert(
                tk.END,
                f"model ABC (MHz)=({float(model.abc_mhz[0]):.6f}, {float(model.abc_mhz[1]):.6f}, {float(model.abc_mhz[2]):.6f}), rep={rep}\n",
            )
            _append_point_group_report(self.s_report, model)
            self.vars["A"].set(f"{float(model.abc_mhz[0]):.10g}")
            self.vars["B"].set(f"{float(model.abc_mhz[1]):.10g}")
            self.vars["C"].set(f"{float(model.abc_mhz[2]):.10g}")
            self.s_report.insert(tk.END, f"harmonic frequencies (cm^-1): {_format_freqs_cm(model.vib_freq_cm)}\n")
            self.s_report.insert(
                tk.END,
                "geometry and Hessian were reoriented internally to the selected input representation before building the harmonic model.\n",
            )
            self.s_report.insert(
                tk.END,
                f"largest geometry-only component: {max_geom} = {levels[max_geom]['geometry_hz']:.6g} Hz\n",
            )
            self.s_report.insert(
                tk.END,
                f"largest full component: {max_full} = {levels[max_full]['total_full_hz']:.6g} Hz\n",
            )
            self.s_report.insert(
                tk.END,
                "component-wise sextic hierarchy (Hz): geometry, cubic(2-index), cubic(3-index), total(2-index), total(full)\n",
            )
            for key in comps:
                vals = levels[key]
                line = (
                    f"  {key}: geom={vals['geometry_hz']:.6g}, "
                    f"cubic_2ind={vals['cubic_sd_hz']:.6g}, "
                    f"cubic_3ind={vals['cubic_3ind_hz']:.6g}, "
                    f"total_2ind={vals['total_sd_hz']:.6g}, "
                    f"total_full={vals['total_full_hz']:.6g}"
                )
                if phi3 is not None:
                    line += (
                        f", %3ind/cubic={vals['pct_3ind_of_cubic']:.3g}%, "
                        f"%3ind/total={vals['pct_3ind_of_total']:.3g}%"
                    )
                self.s_report.insert(tk.END, line + "\n")
            if rotor_limit["is_special_limit"]:
                geom_cart = {key: float(levels[key]["geometry_hz"]) for key in comps}
                full_cart = {key: float(levels[key]["total_full_hz"]) for key in comps}
                geom_special = project_special_sextic_constants(geom_cart, np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
                full_special = project_special_sextic_constants(full_cart, np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
                self.s_report.insert(tk.END, "Symmetry-adapted sextic constants\n")
                if geom_special is not None:
                    self.s_report.insert(
                        tk.END,
                        "  geometry-only: " + ", ".join(f"{nm}={val:.6g} Hz" for nm, val in geom_special["sextic_hz"].items()) + "\n",
                    )
                if full_special is not None:
                    self.s_report.insert(
                        tk.END,
                        "  full: " + ", ".join(f"{nm}={val:.6g} Hz" for nm, val in full_special["sextic_hz"].items()) + "\n",
                    )
                if rotor_limit["kind"] == "linear":
                    q2 = compute_order2_quartic(model)
                    ltype = linear_ltype_terms(
                        model,
                        quartic_special=q2.get("special_quartic_projection"),
                        sextic_special=full_special,
                    )
                    _append_linear_ltype_report(
                        self.s_report,
                        ltype,
                        title="Linear-molecule l-type doubling",
                    )
            self.s_report.insert(
                tk.END,
                "Interpretation: with harmonic input only, the app returns the sextic geometry level; "
                "with a 2-index cubic matrix, it adds the semi-diagonal cubic sector; "
                "with a full anharmonic log, it can also separate the genuine 3-index cubic remainder.\n",
            )
            self.s_report.insert(
                tk.END,
                "Computational note: in finite-difference schemes based on analytic gradients, "
                "the semi-diagonal cubic sector scales linearly with the number of modes, "
                "whereas the genuine 3-index sector scales quadratically.\n",
            )
        except Exception as exc:
            messagebox.showerror("Sextic hierarchy error", str(exc))


if __name__ == "__main__":
    App().mainloop()
