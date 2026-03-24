#!/usr/bin/env python3
"""Vibration-rotation alpha corrections in the CeDiTT tensor language.

This module provides the dedicated public API for the `alpha` branch:

- reading semi-diagonal cubic 2-index data,
- expanding it to the symmetric reduced cubic tensor,
- computing Gaussian-style VPT2 alpha corrections from the harmonic model,
- symmetry-adapted handling of asymmetric, symmetric, and linear limits.

The low-level harmonic/tensor helpers are reused from the validated
CeDiTT/benchmark infrastructure in `compare_gaussian_sextic.py`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from compare_gaussian_sextic import (
    FACTG,
    AMU_KG,
    CLIGHT_CM,
    PLANCK,
    _c1_from_mu1,
    _classify_rotor_limit,
    _representation_axis_values,
    _special_alpha_context,
    _apply_special_alpha_projection,
    _canonicalize_alpha_rows,
    _zeta_xyz,
)


PICH12 = math.pi * math.sqrt(CLIGHT_CM * AMU_KG / PLANCK / (1.0e10**2))


def read_cubic_two_index_matrix(path: str | Path, n_modes: int) -> np.ndarray:
    """Read an ``N x N`` semi-diagonal cubic matrix from text."""
    raw = Path(path).read_text(encoding="utf-8")
    vals = np.fromstring(raw.replace(",", " "), sep=" ", dtype=float)
    if vals.size != n_modes * n_modes:
        raise ValueError(f"Unexpected cubic 2-index matrix size: {vals.size}. Expected {n_modes*n_modes}.")
    return vals.reshape((n_modes, n_modes))


def expand_cubic_two_index_matrix(two_index_cm: np.ndarray) -> np.ndarray:
    """Expand an ``N x N`` semi-diagonal cubic matrix to the full symmetric tensor."""
    mat = np.asarray(two_index_cm, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected a square 2-index cubic matrix, got {mat.shape}")
    n_modes = mat.shape[0]
    out = np.zeros((n_modes, n_modes, n_modes), dtype=float)
    for i in range(n_modes):
        out[i, i, i] = mat[i, i]
        for j in range(n_modes):
            if i == j:
                continue
            val = mat[i, j]
            out[i, i, j] = val
            out[i, j, i] = val
            out[j, i, i] = val
    return out


def alpha_matrix_from_harmonic_and_cubic_cm(
    model,
    phi3_reduced_cm: np.ndarray | None = None,
    *,
    excluded_modes: set[int] | None = None,
    disabled_semidiagonal_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    effective_frequencies_cm: np.ndarray | None = None,
    resonance_threshold_cm: float = 20.0,
) -> dict[str, np.ndarray]:
    """Return Gaussian-style VPT2 alpha contributions in cm^-1."""
    pmom, rot_cm = _representation_axis_values(model)
    freq_cm = np.abs(np.asarray(model.vib_freq_cm, dtype=float))
    anh_freq_cm = freq_cm.copy()
    if effective_frequencies_cm is not None:
        eff = np.asarray(effective_frequencies_cm, dtype=float)
        if eff.shape != freq_cm.shape:
            raise ValueError(f"Unexpected effective frequency shape: {eff.shape}, expected {freq_cm.shape}")
        if np.any(eff <= 1.0e-14):
            raise ValueError("Effective frequencies must be strictly positive.")
        anh_freq_cm = eff.copy()
    zeta = _zeta_xyz(model)
    c1 = _c1_from_mu1(model)
    n_modes = freq_cm.size
    excluded = set() if excluded_modes is None else set(int(x) for x in excluded_modes)
    keep = np.array([(i + 1) not in excluded for i in range(n_modes)], dtype=bool)
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))

    aa = 2.0 * rot_cm**2
    alpha_cor = np.zeros((n_modes, 3), dtype=float)
    alpha_inertia = np.zeros((n_modes, 3), dtype=float)
    alpha_anh = np.zeros((n_modes, 3), dtype=float)
    axis_map = {label: i for i, label in enumerate(model.xyz_to_abc)}
    linear_skip_axis = None
    if rotor_limit["kind"] == "linear":
        linear_skip_axis = axis_map[str(rotor_limit["symmetry_axis"])]

    for ix in range(3):
        if linear_skip_axis is not None and ix == linear_skip_axis:
            continue
        if (not np.isfinite(rot_cm[ix])) or rot_cm[ix] <= 1.0e-14:
            continue
        for i in range(n_modes):
            if not keep[i]:
                continue
            frq_i = freq_cm[i]
            if frq_i <= 1.0e-14:
                continue
            f2ii = frq_i * frq_i
            ai = aa[ix] / frq_i
            for j in range(i):
                if not keep[j]:
                    continue
                zz = float(zeta[ix, i, j])
                if abs(zz) <= 1.0e-14:
                    continue
                frq_j = freq_cm[j]
                if frq_j <= 1.0e-14:
                    continue
                f2jj = frq_j * frq_j
                aj = aa[ix] / frq_j
                dfrq = frq_i - frq_j
                if abs(dfrq) < resonance_threshold_cm:
                    sfrq = frq_i + frq_j
                    cfrq = 0.0 if abs(sfrq) <= 1.0e-14 else dfrq * dfrq / (2.0 * sfrq)
                    cfrq_i = -cfrq / frq_j
                    cfrq_j = -cfrq / frq_i
                else:
                    cfrq_i = (3.0 * f2ii + f2jj) / (f2ii - f2jj)
                    cfrq_j = (3.0 * f2jj + f2ii) / (f2jj - f2ii)
                alpha_cor[i, ix] += ai * zz * zz * cfrq_i
                alpha_cor[j, ix] += aj * zz * zz * cfrq_j

    for ix in range(3):
        if linear_skip_axis is not None and ix == linear_skip_axis:
            continue
        for i in range(n_modes):
            if not keep[i]:
                continue
            frq_i = freq_cm[i]
            if frq_i <= 1.0e-14:
                continue
            ai = aa[ix] / frq_i
            x_i = math.sqrt((FACTG * frq_i) ** 3)
            acc = 0.0
            for jx in range(3):
                if abs(pmom[jx]) <= 1.0e-30:
                    continue
                didq_ixj = 2.0 * pmom[ix] * pmom[jx] * x_i * c1[i, ix, jx]
                acc += didq_ixj**2 / pmom[jx]
            alpha_inertia[i, ix] = 0.75 * ai * acc

    if phi3_reduced_cm is not None:
        phi3 = np.asarray(phi3_reduced_cm, dtype=float)
        if phi3.shape != (n_modes, n_modes, n_modes):
            raise ValueError(f"Unexpected reduced cubic tensor shape: {phi3.shape}, expected {(n_modes, n_modes, n_modes)}")
        if disabled_semidiagonal_pairs:
            phi3 = phi3.copy()
            for pair in disabled_semidiagonal_pairs:
                if len(pair) != 2:
                    raise ValueError(f"Invalid disabled semidiagonal pair: {pair}")
                i, j = int(pair[0]), int(pair[1])
                if i == j:
                    continue
                if not (1 <= i <= n_modes and 1 <= j <= n_modes):
                    raise ValueError(f"Disabled semidiagonal pair {pair} outside mode range 1..{n_modes}.")
                ia = i - 1
                ja = j - 1
                phi3[ia, ia, ja] = 0.0
                phi3[ia, ja, ia] = 0.0
                phi3[ja, ia, ia] = 0.0
                phi3[ja, ja, ia] = 0.0
                phi3[ja, ia, ja] = 0.0
                phi3[ia, ja, ja] = 0.0
        for ix in range(3):
            if linear_skip_axis is not None and ix == linear_skip_axis:
                continue
            for i in range(n_modes):
                if not keep[i]:
                    continue
                frq_i_harm = freq_cm[i]
                frq_i_denom = anh_freq_cm[i]
                if frq_i_harm <= 1.0e-14 or frq_i_denom <= 1.0e-14:
                    continue
                ai = aa[ix] / frq_i_denom
                acc = 0.0
                for j in range(n_modes):
                    if not keep[j]:
                        continue
                    frq_j_harm = freq_cm[j]
                    frq_j_denom = anh_freq_cm[j]
                    if frq_j_harm <= 1.0e-14 or frq_j_denom <= 1.0e-14:
                        continue
                    x_j = math.sqrt((FACTG * frq_j_harm) ** 3)
                    didq_iix_j = 2.0 * pmom[ix] * pmom[ix] * x_j * c1[j, ix, ix]
                    f3_term = phi3[i, i, j] * frq_i_harm * math.sqrt(frq_j_harm) / (frq_j_denom * frq_j_denom)
                    acc += didq_iix_j * f3_term
                alpha_anh[i, ix] = PICH12 * ai * acc

    alpha_cor = -alpha_cor
    alpha_inertia = -alpha_inertia
    alpha_anh = -alpha_anh

    deg_pairs: list[tuple[int, int]] = []
    deg_meta: list[dict[str, object]] = []
    lsdgnm_like: dict[str, object] | None = None
    canonical_pair_data: list[dict[str, object]] = []
    projection_strategy = "identity"
    special_ctx = None
    if rotor_limit["is_special_limit"]:
        special_ctx = _special_alpha_context(model, rotor_limit, excluded_keep_mask=keep, c1=c1)
        alpha_cor = _apply_special_alpha_projection(alpha_cor, special_ctx)
        alpha_inertia = _apply_special_alpha_projection(alpha_inertia, special_ctx)
        alpha_anh = _apply_special_alpha_projection(alpha_anh, special_ctx)
        deg_pairs = list(special_ctx["degenerate_mode_pairs"])
        deg_meta = list(special_ctx["degenerate_mode_metadata"])
        lsdgnm_like = dict(special_ctx["lsdgnm_like"])
        canonical_pair_data = list(special_ctx.get("canonical_pair_data", ()))
        projection_strategy = str(special_ctx["strategy"])

    total = alpha_cor + alpha_inertia + alpha_anh
    abc_order = np.array([model.xyz_to_abc.index(lbl) for lbl in ("a", "b", "c")], dtype=int)
    out = {
        "alpha_total_cm": total,
        "alpha_coriolis_cm": alpha_cor,
        "alpha_inertia_cm": alpha_inertia,
        "alpha_anharmonic_cm": alpha_anh,
        "alpha_total_cm_abc": total[:, abc_order],
        "alpha_coriolis_cm_abc": alpha_cor[:, abc_order],
        "alpha_inertia_cm_abc": alpha_inertia[:, abc_order],
        "alpha_anharmonic_cm_abc": alpha_anh[:, abc_order],
        "abc_order_from_xyz": abc_order,
        "kept_mask": keep,
        "rotor_limit": rotor_limit,
        "degenerate_mode_pairs": deg_pairs,
        "degenerate_mode_metadata": deg_meta,
        "lsdgnm_like": lsdgnm_like,
        "canonical_pair_data": canonical_pair_data,
        "projection_strategy": projection_strategy,
        "anharmonic_reference_frequencies_cm": anh_freq_cm.copy(),
    }
    if rotor_limit["is_special_limit"] and canonical_pair_data and special_ctx is not None:
        alpha_total_canonical_cm = _canonicalize_alpha_rows(total, special_ctx)
        out["alpha_total_canonical_cm"] = alpha_total_canonical_cm
        out["alpha_total_canonical_cm_abc"] = alpha_total_canonical_cm[:, abc_order]
    if rotor_limit["is_special_limit"]:
        if rotor_limit["kind"] == "linear":
            out["alpha_linear_cm"] = 0.5 * (out["alpha_total_cm_abc"][:, 1] + out["alpha_total_cm_abc"][:, 2])
        else:
            label_to_idx = {"a": 0, "b": 1, "c": 2}
            sym_axis = str(rotor_limit["symmetry_axis"])
            deg_axes = tuple(str(x) for x in rotor_limit["degenerate_axes"])
            sym_idx = label_to_idx[sym_axis]
            perp_indices = [label_to_idx[lbl] for lbl in deg_axes]
            out["alpha_axial_cm"] = {
                "parallel": out["alpha_total_cm_abc"][:, sym_idx],
                "perpendicular": 0.5 * (
                    out["alpha_total_cm_abc"][:, perp_indices[0]] + out["alpha_total_cm_abc"][:, perp_indices[1]]
                ),
            }
    return out


def alpha_matrix_from_cubic_two_index_cm(
    model,
    two_index_cm: np.ndarray,
    *,
    excluded_modes: set[int] | None = None,
    disabled_semidiagonal_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    effective_frequencies_cm: np.ndarray | None = None,
    resonance_threshold_cm: float = 20.0,
) -> dict[str, np.ndarray]:
    """Convenience wrapper using the semi-diagonal reduced cubic matrix ``phi_iij``."""
    mat = np.asarray(two_index_cm, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2-index cubic matrix, got {mat.shape}")
    phi3_sd = expand_cubic_two_index_matrix(mat)
    return alpha_matrix_from_harmonic_and_cubic_cm(
        model,
        phi3_sd,
        excluded_modes=excluded_modes,
        disabled_semidiagonal_pairs=disabled_semidiagonal_pairs,
        effective_frequencies_cm=effective_frequencies_cm,
        resonance_threshold_cm=resonance_threshold_cm,
    )
