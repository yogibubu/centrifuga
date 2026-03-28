#!/usr/bin/env python3
"""Utility workflows exposing a minimal subset of order-2 quartic utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp

from compare_gaussian_sextic import FAC3AU, FACTG, DIDQ_AU_PER_AMU_SQRT_ANG, _degenerate_mode_metadata
from quartic_channels import channel_h12h12, tau_to_watson_a
from symmetry_metadata import assign_normal_mode_irreps

EH_TO_MHZ = 6.57968392061e9
AU_FREQ_TO_CMINV = 219474.6313705
CMINV_TO_HZ = 2.99792458e10
CMINV_TO_MHZ = CMINV_TO_HZ / 1.0e6


def _representation_axis_values(model) -> tuple[np.ndarray, np.ndarray]:
    abc_to_idx = {"a": 0, "b": 1, "c": 2}
    perm = np.array([abc_to_idx[label] for label in model.xyz_to_abc], dtype=int)
    moments_xyz = np.asarray(model.moments_amu_a2, dtype=float)[perm]
    rot_xyz_cm = (np.asarray(model.abc_mhz, dtype=float) / CMINV_TO_MHZ)[perm]
    return moments_xyz, rot_xyz_cm


def _didq_from_model(model) -> np.ndarray:
    """Replicate Gaussian ``dIdQ`` in the model convention."""
    n_atoms = model.masses_amu.size
    modes = model.vib_vecs_mw_pa.reshape(n_atoms, 3, -1)
    didq = np.zeros((modes.shape[2], 6), dtype=float)
    ijx = 0
    for ix in range(3):
        for jx in range(ix + 1):
            for mode in range(modes.shape[2]):
                acc = 0.0
                for atom in range(n_atoms):
                    mass = np.sqrt(model.masses_amu[atom])
                    acc -= mass * model.coords_pa_ang[atom, ix] * modes[atom, jx, mode]
                    if ix == jx:
                        for kx in range(3):
                            acc += mass * model.coords_pa_ang[atom, kx] * modes[atom, kx, mode]
                didq[mode, ijx] = 2.0 * acc
            ijx += 1
    return didq


def _linear_gaussian_source_qe_hz(model, pair: tuple[int, int]) -> float:
    """Return the non-resonant Gaussian q^e source formula in Hz.

    This follows the explicit implementation in `dinautil.F`:
      q_t^e = - sum_{ix,j(nondeg)} [2 B_ix^2 / w_t] * A_ij * zeta(ix,t,j)^2
    with the same near-degeneracy regularization, but using the current
    harmonic model as input.
    """
    freq = np.asarray(model.vib_freq_cm, dtype=float).reshape(-1)
    zeta = np.asarray(model.coriolis_zeta_pairs_xyz, dtype=float)
    _moments_xyz, rot_xyz_cm = _representation_axis_values(model)
    rot_xyz_cm = np.where(np.isfinite(rot_xyz_cm), rot_xyz_cm, 0.0)

    i0 = int(pair[0])
    wi = float(freq[i0])
    if abs(wi) <= 1.0e-12:
        return 0.0
    fii = wi * wi
    q_cm = 0.0
    deg_pair = {int(pair[0]), int(pair[1])}
    for ix in range(3):
        bix = float(rot_xyz_cm[ix])
        if abs(bix) <= 1.0e-12:
            continue
        pref = 2.0 * bix * bix / wi
        for j in range(freq.size):
            if j in deg_pair:
                continue
            wj = float(freq[j])
            fjj = wj * wj
            dfrq = wi - wj
            if abs(dfrq) <= 1.0e-8:
                sfrq = wi + wj
                if abs(wj) <= 1.0e-12 or abs(sfrq) <= 1.0e-12:
                    continue
                aij = -(dfrq * dfrq) / (2.0 * wj * sfrq)
            else:
                den = fii - fjj
                if abs(den) <= 1.0e-12:
                    continue
                aij = (3.0 * fii + fjj) / den
            zij = float(zeta[ix, i0, j])
            if abs(zij) <= 1.0e-12:
                continue
            q_cm -= pref * aij * zij * zij
    return float(q_cm * CMINV_TO_HZ)


def _linear_gaussian_source_aux_data(model, gaussian_log_path: str):
    from gaussian_vpt_parser import (
        frequency_reorder_map,
        parse_gaussian_alpha_data,
        parse_gaussian_anharmonic_force_data,
    )

    freq = np.asarray(model.vib_freq_cm, dtype=float).reshape(-1)
    alpha = parse_gaussian_alpha_data(gaussian_log_path)
    anh = parse_gaussian_anharmonic_force_data(gaussian_log_path)
    mapping = frequency_reorder_map(anh.frequencies_cm, freq)
    alpha_target = np.zeros((freq.size, 3), dtype=float)
    raw_target = np.zeros_like(anh.phi3_raw_au)
    for src, tgt in enumerate(mapping):
        alpha_target[tgt] = alpha.alpha_cm[src]
    for i in range(mapping.size):
        for j in range(mapping.size):
            for k in range(mapping.size):
                raw_target[mapping[i], mapping[j], mapping[k]] = anh.phi3_raw_au[i, j, k]
    return alpha_target, raw_target


def _linear_gaussian_exact_source_blocks(gaussian_log_path: str) -> dict[str, object]:
    from gaussian_vpt_parser import (
        _resolve_gaussian_path,
        _find_c1_matrix,
        _find_c2_tensor,
        _find_last_section_index,
        _find_tau_tensor,
        parse_gaussian_alpha_data,
        parse_gaussian_anharmonic_force_data,
        parse_gaussian_linear_ltype_constants,
        parse_gaussian_sextic_benchmark,
    )

    lines = _resolve_gaussian_path(gaussian_log_path).read_text(encoding="utf-8").splitlines()
    alpha = parse_gaussian_alpha_data(gaussian_log_path)
    anh = parse_gaussian_anharmonic_force_data(gaussian_log_path)
    try:
        sextic = parse_gaussian_sextic_benchmark(gaussian_log_path)
        sextic_tau_cm = None if sextic.tau_cm is None else np.asarray(sextic.tau_cm, dtype=float)
        sextic_c1 = None if sextic.c1 is None else np.asarray(sextic.c1, dtype=float)
        sextic_c2 = None if sextic.c2 is None and sextic.c2_reordered_to_print is None else np.asarray(
            sextic.c2_reordered_to_print if sextic.c2_reordered_to_print is not None else sextic.c2,
            dtype=float,
        )
    except Exception:
        try:
            sextic_start = _find_last_section_index(lines, "Dump from SEXTIC")
            tau_start = _find_last_section_index(lines, "Quartic Centrifugal Distortion Constants Tau (in cm^-1)")
            c1_start = _find_last_section_index(lines, "Dimensionless C_i^ab Matrix")
            c2_start = _find_last_section_index(lines, "Dimensionless C_i^abc Matrix")
            if tau_start < sextic_start:
                raise ValueError("Could not locate sextic Tau block after 'Dump from SEXTIC'.")
            sextic_tau_cm = _find_tau_tensor(lines, tau_start)
            sextic_c1 = _find_c1_matrix(lines, c1_start)
            sextic_c2 = _find_c2_tensor(lines, c2_start)
        except Exception:
            sextic_tau_cm = None
            sextic_c1 = None
            sextic_c2 = None
    qconst = parse_gaussian_linear_ltype_constants(gaussian_log_path)

    axis_pat = re.compile(r"\(([xyz])\)", re.IGNORECASE)
    axis_to_col: dict[str, int] = {}
    for col, label in enumerate(alpha.axis_labels):
        m = axis_pat.search(label)
        if m:
            axis_to_col[m.group(1).lower()] = col
    if not {"x", "y", "z"} <= set(axis_to_col):
        raise ValueError("Could not infer Gaussian x/y/z ordering from alpha-matrix header.")
    alpha_xyz_cm = (alpha.alpha_mhz[:, [axis_to_col["x"], axis_to_col["y"], axis_to_col["z"]]] / CMINV_TO_MHZ).copy()

    n_modes = int(alpha.mode_indices.size)
    didq = np.zeros((n_modes, 6), dtype=float)
    didq_header_pat = re.compile(r"Ixx\s+Ixy\s+Iyy\s+Ixz\s+Iyz\s+Izz", re.IGNORECASE)
    qrow_pat = re.compile(r"^\s*Q\(\s*(\d+)\)\s+(.+)$")
    for i, line in enumerate(lines):
        if didq_header_pat.search(line):
            j = i + 1
            while j < len(lines):
                m = qrow_pat.match(lines[j])
                if not m:
                    break
                idx = int(m.group(1)) - 1
                vals = [float(tok) for tok in m.group(2).split()[-6:]]
                didq[idx] = vals
                j += 1
            break

    zeta_xyz = np.zeros((3, n_modes, n_modes), dtype=float)
    zeta_pat = re.compile(r"^\s*([xyz])\s+(\d+)\s+(\d+)\s+([\-0-9.]+)", re.IGNORECASE)
    axis_idx = {"x": 0, "y": 1, "z": 2}
    for line in lines:
        m = zeta_pat.match(line)
        if not m:
            continue
        ax = axis_idx[m.group(1).lower()]
        i = int(m.group(2)) - 1
        j = int(m.group(3)) - 1
        val = float(m.group(4))
        zeta_xyz[ax, i, j] = val
        zeta_xyz[ax, j, i] = val

    rep_indices = sorted(int(k) - 1 for k in qconst.q_e_mhz)
    deg_indices: list[int] = []
    used = set()
    for i, wi in enumerate(anh.frequencies_cm):
        if i in used:
            continue
        group = [j for j, wj in enumerate(anh.frequencies_cm) if abs(float(wj) - float(wi)) <= 1.0e-6]
        if len(group) == 2:
            deg_indices.extend(group)
            used.update(group)
    return {
        "freq_cm": np.asarray(anh.frequencies_cm, dtype=float),
        "alpha_xyz_cm": alpha_xyz_cm,
        "didq_amu_sqrt_ang": didq,
        "zeta_xyz": zeta_xyz,
        "phi3_raw_au": np.asarray(anh.phi3_raw_au, dtype=float),
        "sextic_tau_cm": None if sextic_tau_cm is None else np.asarray(sextic_tau_cm, dtype=float),
        "sextic_c1": None if sextic_c1 is None else np.asarray(sextic_c1, dtype=float),
        "sextic_c2": None if sextic_c2 is None else np.asarray(sextic_c2, dtype=float),
        "qconst": qconst,
        "representative_indices": rep_indices,
        "degenerate_indices": sorted(set(deg_indices)),
    }


def _linear_gaussian_source_qjk_hz(
    model,
    pair: tuple[int, int],
    *,
    d_hz: float | None,
    rotor_limit: dict[str, object],
    pair_meta: list[dict[str, object]],
    alpha_target_cm: np.ndarray,
    raw_cubic_au: np.ndarray,
) -> dict[str, float] | None:
    if d_hz is None:
        return None
    freq = np.asarray(model.vib_freq_cm, dtype=float).reshape(-1)
    zeta = np.asarray(model.coriolis_zeta_pairs_xyz, dtype=float)
    moments_xyz, rot_xyz_cm = _representation_axis_values(model)
    rot_xyz_cm = np.where(np.isfinite(rot_xyz_cm), rot_xyz_cm, 0.0)
    deg_axes = tuple(rotor_limit.get("degenerate_axes", ()))
    if not deg_axes:
        return None
    try:
        perp_axis = tuple(model.xyz_to_abc).index(str(deg_axes[0]))
    except ValueError:
        perp_axis = 1
    rot_iax = float(rot_xyz_cm[perp_axis])
    pmom_iax = float(moments_xyz[perp_axis])
    if abs(rot_iax) <= 1.0e-12 or abs(pmom_iax) <= 1.0e-12:
        return None

    didq_au_sqrt_ang = np.asarray(model.dI_au, dtype=float) / DIDQ_AU_PER_AMU_SQRT_ANG
    v_iscr = np.zeros(freq.size, dtype=float)
    for i in range(freq.size):
        wi = float(freq[i])
        if abs(wi) <= 1.0e-12:
            continue
        x = np.sqrt((FACTG * abs(wi)) ** 3)
        num = float(didq_au_sqrt_ang[perp_axis, perp_axis, i])
        den = 2.0 * x * pmom_iax * pmom_iax
        v_iscr[i] = 0.0 if abs(den) <= 1.0e-30 else num / den

    pair0 = int(pair[0])
    wi = float(freq[pair0])
    fii = wi * wi
    fiii = fii * wi
    q_e_cm = _linear_gaussian_source_qe_hz(model, pair) / CMINV_TO_HZ
    q_j_cm = -4.0 * (float(d_hz) / CMINV_TO_HZ) * q_e_cm / rot_iax
    q_j_cm += 2.0 * (rot_iax**4) / fiii
    q_j_cm -= (rot_iax**2) * (2.0 * float(alpha_target_cm[pair0, perp_axis]) + q_e_cm) / fii

    deg_modes = {int(x) for meta in pair_meta for x in meta["pair"]}
    nondeg_modes = [idx for idx in range(freq.size) if idx not in deg_modes]
    deg_reps = [int(meta["pair"][0]) for meta in pair_meta]

    for j in deg_reps:
        if j == pair0:
            continue
        wj = float(freq[j])
        fjj = wj * wj
        a = 2.0 * wi * (wi + wj) if abs(wi - wj) <= 1.0e-8 else (fii - fjj)
        if abs(a) <= 1.0e-30:
            continue
        for k in nondeg_modes:
            wk = float(freq[k])
            fkk = wk * wk
            b = 2.0 * wk * (wi + wk) if abs(wk - wi) <= 1.0e-8 else (fkk - fii)
            if abs(b) <= 1.0e-30:
                continue
            for ix in range(3):
                zij = float(zeta[ix, pair0, k])
                zjk = float(zeta[ix, j, k])
                if abs(zij) <= 1.0e-12 or abs(zjk) <= 1.0e-12:
                    continue
                q_j_cm -= 32.0 * (rot_xyz_cm[ix] * rot_xyz_cm[ix] * zij * zjk) ** 2 * wi * (3.0 * fii - fjj) / (a * b * b)
                for l in nondeg_modes:
                    fijl = float(raw_cubic_au[pair0, j, l]) * FAC3AU
                    if abs(fijl) <= 1.0e-14:
                        continue
                    c = np.sqrt(freq[l]) ** 3
                    q_j_cm -= 16.0 * (rot_xyz_cm[ix] ** 2) * fijl * zij * zjk * wi / (c * a * b)

    for j in nondeg_modes:
        wj = float(freq[j])
        fjj = wj * wj
        a = 2.0 * wj * (wj + wi) if abs(wj - wi) <= 1.0e-8 else (fjj - fii)
        if abs(a) <= 1.0e-30:
            continue
        for ix in range(3):
            zij = float(zeta[ix, pair0, j])
            if abs(zij) <= 1.0e-12:
                continue
            q_j_cm -= (4.0 * (rot_xyz_cm[ix] ** 2) * q_e_cm - 2.0 * float(alpha_target_cm[pair0, ix])) * zij * zij * (fjj - fii) / (a * a)
            q_j_cm -= 12.0 * rot_xyz_cm[ix] * (v_iscr[j] ** 2) * zij * zij * wi * wj / (a * a)
            for k in nondeg_modes:
                zik = float(zeta[ix, pair0, k])
                if abs(zik) <= 1.0e-12:
                    continue
                q_j_cm -= 8.0 * rot_xyz_cm[ix] * v_iscr[j] * v_iscr[k] * zij * zik * np.sqrt(wj) * wi / (a * (np.sqrt(freq[k]) ** 3))
                b = 2.0 * freq[k] * (freq[k] + wi) if abs(freq[k] - wi) <= 1.0e-8 else (freq[k] ** 2 - fii)
                if abs(b) <= 1.0e-30:
                    continue
                for l in nondeg_modes:
                    fjkl = float(raw_cubic_au[j, k, l]) * FAC3AU
                    if abs(fjkl) <= 1.0e-14:
                        continue
                    q_j_cm -= 8.0 * (rot_xyz_cm[ix] ** 2) * v_iscr[l] * fjkl * zij * zik * wi / (a * b * (np.sqrt(freq[l]) ** 3))

    q_k_cm = 0.0
    for j in nondeg_modes:
        wj = float(freq[j])
        fjj = wj * wj
        a = 2.0 * wj * (wj + wi) if abs(wj - wi) <= 1.0e-8 else (fjj - fii)
        if abs(a) <= 1.0e-30:
            continue
        b = fjj * fjj + 10.0 * fii * fjj + 5.0 * fii * fii
        for ix in range(3):
            zij = float(zeta[ix, pair0, j])
            if abs(zij) <= 1.0e-12:
                continue
            q_k_cm -= (8.0 * (rot_xyz_cm[ix] ** 4)) * zij * zij * b / (wi * (a**3))

    return {
        "q_J_source": float(q_j_cm * CMINV_TO_HZ),
        "q_K_source": float((q_k_cm - q_j_cm) * CMINV_TO_HZ),
    }


def _linear_gaussian_exact_source_qjk_hz(
    model,
    *,
    gaussian_source_blocks: dict[str, object],
    q_index: int,
    d_hz: float | None,
    rotor_limit: dict[str, object],
) -> dict[str, float] | None:
    if d_hz is None:
        return None
    freq = np.asarray(gaussian_source_blocks["freq_cm"], dtype=float)
    alpha_xyz_cm = np.asarray(gaussian_source_blocks["alpha_xyz_cm"], dtype=float)
    didq = np.asarray(gaussian_source_blocks["didq_amu_sqrt_ang"], dtype=float)
    zeta = np.asarray(gaussian_source_blocks["zeta_xyz"], dtype=float)
    raw = np.asarray(gaussian_source_blocks["phi3_raw_au"], dtype=float)
    qconst = gaussian_source_blocks["qconst"]
    rep_indices = [int(x) for x in gaussian_source_blocks["representative_indices"]]
    deg_indices = [int(x) for x in gaussian_source_blocks.get("degenerate_indices", rep_indices)]

    i0 = int(q_index) - 1
    if i0 < 0 or i0 >= freq.size:
        return None
    if i0 not in rep_indices:
        return None

    deg_axes = tuple(rotor_limit.get("degenerate_axes", ()))
    sym_axis = str(rotor_limit.get("symmetry_axis", "a"))
    abc_to_idx = {"a": 0, "b": 1, "c": 2}
    abc_mhz = np.asarray(model.abc_mhz, dtype=float)
    moments = np.asarray(model.moments_amu_a2, dtype=float)
    gauss_xyz_order = [abc_to_idx[str(deg_axes[0])], abc_to_idx[str(deg_axes[1])], abc_to_idx[sym_axis]]
    rot_xyz_cm = np.asarray(
        [
            0.0 if not np.isfinite(abc_mhz[idx]) else float(abc_mhz[idx] / CMINV_TO_MHZ)
            for idx in gauss_xyz_order
        ],
        dtype=float,
    )
    pmom_xyz = np.asarray([float(moments[idx]) for idx in gauss_xyz_order], dtype=float)
    rot_iax = float(rot_xyz_cm[0])
    pmom_iax = float(pmom_xyz[0])
    if abs(rot_iax) <= 1.0e-12 or abs(pmom_iax) <= 1.0e-12:
        return None

    wi = float(freq[i0])
    if abs(wi) <= 1.0e-12:
        return None
    fii = wi * wi
    fiii = fii * wi
    q_e_cm = float(qconst.q_e_mhz[q_index] / CMINV_TO_MHZ)
    q_j_cm = -4.0 * (float(d_hz) / CMINV_TO_HZ) * q_e_cm / rot_iax
    q_j_cm += 2.0 * (rot_iax**4) / fiii
    q_j_cm -= (rot_iax**2) * (2.0 * float(alpha_xyz_cm[i0, 0]) + q_e_cm) / fii

    v_iscr = np.zeros(freq.size, dtype=float)
    for i, w in enumerate(freq):
        x = np.sqrt((FACTG * abs(w)) ** 3) if abs(w) > 1.0e-12 else 0.0
        den = 2.0 * x * pmom_iax * pmom_iax
        v_iscr[i] = 0.0 if abs(den) <= 1.0e-30 else float(didq[i, 0] / den)

    nonrep_indices = [idx for idx in range(freq.size) if idx not in set(rep_indices)]

    for j in deg_indices:
        if j == i0:
            continue
        wj = float(freq[j])
        fjj = wj * wj
        a = 2.0 * wi * (wi + wj) if abs(wi - wj) <= 1.0e-8 else (fii - fjj)
        if abs(a) <= 1.0e-30:
            continue
        for k in nonrep_indices:
            wk = float(freq[k])
            fkk = wk * wk
            b = 2.0 * wk * (wi + wk) if abs(wk - wi) <= 1.0e-8 else (fkk - fii)
            if abs(b) <= 1.0e-30:
                continue
            for ix in range(3):
                zij = float(zeta[ix, i0, k])
                zjk = float(zeta[ix, j, k])
                if abs(zij) <= 1.0e-12 or abs(zjk) <= 1.0e-12:
                    continue
                q_j_cm -= 32.0 * (rot_xyz_cm[ix] * rot_xyz_cm[ix] * zij * zjk) ** 2 * wi * (3.0 * fii - fjj) / (a * b * b)
                for l in nonrep_indices:
                    x = float(raw[i0, j, l])
                    if abs(x) <= 1.0e-14:
                        continue
                    fijl = x * FAC3AU
                    c = np.sqrt(freq[l]) ** 3
                    q_j_cm -= 16.0 * (rot_xyz_cm[ix] ** 2) * fijl * zij * zjk * wi / (c * a * b)

    for j in nonrep_indices:
        wj = float(freq[j])
        fjj = wj * wj
        a = 2.0 * wj * (wj + wi) if abs(wj - wi) <= 1.0e-8 else (fjj - fii)
        if abs(a) <= 1.0e-30:
            continue
        for ix in range(3):
            zij = float(zeta[ix, i0, j])
            if abs(zij) <= 1.0e-12:
                continue
            q_j_cm -= (4.0 * (rot_xyz_cm[ix] ** 2) * q_e_cm - 2.0 * float(alpha_xyz_cm[i0, ix])) * zij * zij * (fjj - fii) / (a * a)
            q_j_cm -= 12.0 * rot_xyz_cm[ix] * (v_iscr[j] ** 2) * zij * zij * wi * wj / (a * a)
            for k in nonrep_indices:
                zik = float(zeta[ix, i0, k])
                if abs(zik) <= 1.0e-12:
                    continue
                q_j_cm -= 8.0 * rot_xyz_cm[ix] * v_iscr[j] * v_iscr[k] * zij * zik * np.sqrt(wj) * wi / (a * (np.sqrt(freq[k]) ** 3))
                b = 2.0 * freq[k] * (freq[k] + wi) if abs(freq[k] - wi) <= 1.0e-8 else (freq[k] ** 2 - fii)
                if abs(b) <= 1.0e-30:
                    continue
                for l in nonrep_indices:
                    x = float(raw[j, k, l])
                    if abs(x) <= 1.0e-14:
                        continue
                    fjkl = x * FAC3AU
                    q_j_cm -= 8.0 * (rot_xyz_cm[ix] ** 2) * v_iscr[l] * fjkl * zij * zik * wi / (a * b * (np.sqrt(freq[l]) ** 3))

    q_k_cm = 0.0
    for j in nonrep_indices:
        wj = float(freq[j])
        fjj = wj * wj
        a = 2.0 * wj * (wj + wi) if abs(wj - wi) <= 1.0e-8 else (fjj - fii)
        if abs(a) <= 1.0e-30:
            continue
        b = fjj * fjj + 10.0 * fii * fjj + 5.0 * fii * fii
        for ix in range(3):
            zij = float(zeta[ix, i0, j])
            if abs(zij) <= 1.0e-12:
                continue
            q_k_cm -= (8.0 * (rot_xyz_cm[ix] ** 4)) * zij * zij * b / (wi * (a**3))

    return {
        "q_J_source": float(q_j_cm * CMINV_TO_HZ),
        "q_K_source": float((q_k_cm - q_j_cm) * CMINV_TO_HZ),
        "source": "reconstructed_from_printed_gaussian_blocks",
    }


def _linear_gaussian_exact_source_h_hz(
    model,
    *,
    gaussian_source_blocks: dict[str, object],
    d_hz: float | None,
    rotor_limit: dict[str, object],
) -> dict[str, object] | None:
    """Reconstruct the linear sextic constant from Gaussian source blocks.

    This follows the ``ITop=4`` branch of ``Sextic`` in ``l717.F``:
    only the transverse diagonal branch survives and

        He = Phi(1,1,1) = X1 - X2 + X3

    with ``X4 = 0`` for linear tops.
    """
    if d_hz is None:
        return None
    deg_axes = tuple(rotor_limit.get("degenerate_axes", ()))
    sym_axis = str(rotor_limit.get("symmetry_axis", "a"))
    if len(deg_axes) != 2:
        return None

    from gaussian_vpt_parser import frequency_reorder_map

    freq_src = np.asarray(gaussian_source_blocks["freq_cm"], dtype=float)
    didq_src = np.asarray(gaussian_source_blocks["didq_amu_sqrt_ang"], dtype=float)
    zeta_src = np.asarray(gaussian_source_blocks["zeta_xyz"], dtype=float)
    phi3_src = np.asarray(gaussian_source_blocks["phi3_raw_au"], dtype=float)
    tau_src = gaussian_source_blocks.get("sextic_tau_cm")
    c1_src = gaussian_source_blocks.get("sextic_c1")
    c2_src = gaussian_source_blocks.get("sextic_c2")

    # Align Gaussian printed source blocks to the current model convention.
    # ``frequency_reorder_map`` returns source -> target; we need the inverse
    # permutation to reorder source-indexed arrays into target order.
    source_to_target = frequency_reorder_map(freq_src, np.abs(np.asarray(model.vib_freq_cm, dtype=float)))
    target_order = np.argsort(np.asarray(source_to_target, dtype=int))
    freq = freq_src[target_order].copy()
    didq = didq_src[target_order].copy()
    zeta = zeta_src[:, target_order][:, :, target_order].copy()
    phi3_raw = phi3_src[np.ix_(target_order, target_order, target_order)].copy()

    model_didq = _didq_from_model(model)
    sign_vec = np.ones(freq.size, dtype=float)
    for i in range(freq.size):
        if np.dot(didq[i], model_didq[i]) < 0.0:
            sign_vec[i] = -1.0
    didq *= sign_vec[:, None]
    zeta *= sign_vec[None, :, None] * sign_vec[None, None, :]
    phi3_raw *= sign_vec[:, None, None] * sign_vec[None, :, None] * sign_vec[None, None, :]
    c1 = None
    if c1_src is not None:
        c1 = np.asarray(c1_src, dtype=float)[target_order].copy()
        c1 *= sign_vec[:, None, None]
    c2 = None
    if c2_src is not None:
        c2 = np.asarray(c2_src, dtype=float)[target_order].copy()
        c2 *= sign_vec[:, None, None, None]
    tau = None if tau_src is None else np.asarray(tau_src, dtype=float).copy()

    moments_xyz, rot_xyz_cm = _representation_axis_values(model)
    abc_to_xyz = {label: i for i, label in enumerate(model.xyz_to_abc)}
    sym_xyz = abc_to_xyz.get(sym_axis, 0)
    perp_xyz = [i for i in range(3) if i != sym_xyz]
    rot_xyz_safe = np.asarray(rot_xyz_cm, dtype=float).copy()
    rot_xyz_safe[~np.isfinite(rot_xyz_safe)] = 0.0

    def _idx_tm(a: int, b: int) -> int:
        pair = (max(a, b), min(a, b))
        return {(0, 0): 0, (1, 0): 1, (1, 1): 2, (2, 0): 3, (2, 1): 4, (2, 2): 5}[pair]

    n_modes = freq.size
    if c1 is None:
        c1 = np.zeros((n_modes, 3, 3), dtype=float)
        for i in range(n_modes):
            fi = abs(float(freq[i]))
            if fi <= 1.0e-12:
                continue
            x = np.sqrt((FACTG * fi) ** 3)
            for ix in range(3):
                for jx in range(3):
                    den = 2.0 * moments_xyz[ix] * moments_xyz[jx] * x
                    c1[i, ix, jx] = 0.0 if abs(den) <= 1.0e-30 else float(didq[i, _idx_tm(ix, jx)] / den)

    if c2 is None:
        c2 = np.zeros((n_modes, 3, 3, 3), dtype=float)
        for i in range(n_modes):
            fi = abs(float(freq[i]))
            if fi <= 1.0e-12:
                continue
            for j in range(n_modes):
                fj = abs(float(freq[j]))
                if fj <= 1.0e-12:
                    continue
                kernel = (2.0 / 3.0) * (fi**2 + 2.0 * fj**2) / np.sqrt(abs(fi**5 * fj))
                for ix in range(3):
                    for jx in range(3):
                        for kx in range(3):
                            c2[i, ix, jx, kx] += kernel * (
                                rot_xyz_safe[ix] * zeta[ix, i, j] * c1[j, jx, kx]
                                + rot_xyz_safe[jx] * zeta[jx, i, j] * c1[j, kx, ix]
                                + rot_xyz_safe[kx] * zeta[kx, i, j] * c1[j, ix, jx]
                            )

    components_hz: dict[str, float] = {}
    pieces_hz: dict[str, dict[str, float]] = {}
    xyz_to_abc = {i: label for i, label in enumerate(model.xyz_to_abc)}
    for ix in perp_xyz:
        x1_cm = 0.0
        if tau is not None:
            for jx in perp_xyz:
                if abs(rot_xyz_safe[jx]) <= 1.0e-30:
                    continue
                x1_cm += float(tau[ix, ix, ix, jx]) ** 2 / float(rot_xyz_safe[jx])
            x1_cm *= 3.0 / 16.0
        elif d_hz is not None:
            d_cm = float(d_hz / CMINV_TO_HZ)
            x1_cm = 3.0 * (4.0 * d_cm) ** 2 / (16.0 * rot_xyz_cm[ix])
        x2_cm = sum(float(freq[i]) * c2[i, ix, ix, ix] ** 2 for i in range(n_modes)) / 2.0
        x3_cm = 0.0
        for i in range(n_modes):
            fi = abs(float(freq[i]))
            if fi <= 1.0e-12:
                continue
            for j in range(n_modes):
                fj = abs(float(freq[j]))
                if fj <= 1.0e-12:
                    continue
                for k in range(n_modes):
                    fk = abs(float(freq[k]))
                    if fk <= 1.0e-12:
                        continue
                    den = np.sqrt(fi * fj * fk)
                    x3_cm += (
                        float(phi3_raw[i, j, k]) * FAC3AU / den * c1[i, ix, ix] * c1[j, ix, ix] * c1[k, ix, ix]
                    )
        x3_cm /= 6.0
        key = xyz_to_abc[ix] * 3
        components_hz[key] = float((x1_cm - x2_cm + x3_cm) * CMINV_TO_HZ)
        pieces_hz[key] = {
            "X1": float(x1_cm * CMINV_TO_HZ),
            "X2": float(x2_cm * CMINV_TO_HZ),
            "X3": float(x3_cm * CMINV_TO_HZ),
        }

    first_key = xyz_to_abc[perp_xyz[0]] * 3 if perp_xyz else None
    vals = list(components_hz.values())
    return {
        "H": float(components_hz[first_key]) if first_key is not None else 0.0,
        "primary_component": first_key,
        "perpendicular_components_hz": components_hz,
        "pieces_hz": pieces_hz,
        "spread_hz": float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0,
        "source": "gaussian_sextic_exact_reconstructed",
        "formula": "He = Phi(1,1,1) = X1 - X2 + X3 from the dedicated linear-top branch of L717/Sextic",
    }


def _watson_dict_to_float(watson: dict[str, sp.Expr]) -> dict[str, float]:
    return {key: float(EH_TO_MHZ * sp.N(value)) for key, value in watson.items()}


def _compressed_tau_to_float(tau: dict[str, sp.Expr]) -> dict[str, float]:
    return {key: float(EH_TO_MHZ * sp.N(value)) for key, value in tau.items()}


def compute_order2_quartic(model, *, linear_log_path: str | None = None) -> dict[str, object]:
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
        "linear_ltype_terms": linear_ltype_terms(model, quartic_special=special_quartic, gaussian_log_path=linear_log_path),
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


def linear_ltype_terms(model, *, quartic_special=None, sextic_special=None, gaussian_log_path: str | None = None):
    """Return an experimental minimal pairwise l-type model for linear molecules.

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
    interactions, and should be treated as an experimental beyond-paper
    extension layered on top of the paper-aligned linear pure-rotational
    sector.
    """
    rotor_limit = classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    if rotor_limit["kind"] != "linear":
        return None

    gaussian_linear = None
    gaussian_source_blocks = None
    gaussian_rotdist = None
    gaussian_source_sextic = None
    if gaussian_log_path:
        from gaussian_vpt_parser import parse_gaussian_linear_ltype_constants, parse_gaussian_linear_rotdist_constants

        gaussian_linear = parse_gaussian_linear_ltype_constants(gaussian_log_path)
        gaussian_rotdist = parse_gaussian_linear_rotdist_constants(gaussian_log_path)
        gaussian_source_blocks = _linear_gaussian_exact_source_blocks(gaussian_log_path)

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
    if gaussian_source_blocks is not None:
        gaussian_source_sextic = _linear_gaussian_exact_source_h_hz(
            model,
            gaussian_source_blocks=gaussian_source_blocks,
            d_hz=d_hz,
            rotor_limit=rotor_limit,
        )
    h_exact_hz = None
    if gaussian_rotdist is not None and gaussian_rotdist.h_mhz is not None:
        h_exact_hz = float(gaussian_rotdist.h_mhz * 1.0e6)
    h_source_hz = None if gaussian_source_sextic is None else float(gaussian_source_sextic["H"])
    h_feed_hz = h_source_hz if h_source_hz is not None else (h_exact_hz if h_exact_hz is not None else h_hz)

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
        q_th = None if h_feed_hz is None else abs(zeta_parallel) * abs(h_feed_hz)
        q_l_leading_hz = 0.0
        q_l_watson_hz = 0.0
        q_e_source_hz = _linear_gaussian_source_qe_hz(model, tuple(int(x) for x in meta["pair"]))
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
            "q_e_source": q_e_source_hz,
            "q_v": None,
            "note": "q_e0 and q_eW are standard spectroscopic estimates; q_e_source follows Gaussian's explicit non-resonant source formula; q_v requires additional vibrational-state corrections not yet included here.",
        }
        if gaussian_source_blocks is not None:
            qjk_source = _linear_gaussian_exact_source_qjk_hz(
                model,
                gaussian_source_blocks=gaussian_source_blocks,
                q_index=int(meta["pair"][0]) + 1,
                d_hz=d_hz,
                rotor_limit=rotor_limit,
            )
            if qjk_source is not None:
                item["gaussian_source_rotational_constants_hz"] = qjk_source
        item["effective_linear_model_hz"] = {
            "primary_basis": ["I_l", "Z_l", "X_l", "Y_l"],
            "model": "H_eff^(lin) = q_e^(src) X_l + q_J^(pair) J^2 X_l + q_H^(pair) (J^2)^2 X_l",
            "constants_hz": {
                "q_e0": q_l_leading_hz,
                "q_eW": q_l_watson_hz,
                "q_e_source": q_e_source_hz,
                "q_J_pair": q_tj_val,
                "q_H_pair": q_th_val,
            },
        }
        item["conventional_linear_model_hz"] = {
            "status": "partial_internal_mapping",
            "source": "internal_nonresonant_pairwise",
            "model": "q_i = q_i^e + (q_i^J) J(J+1) + (q_i^K) K(K±1)^2",
            "constants_hz": {
                "q_e": q_e_source_hz,
                "q_J": q_tj_val,
                "q_K": None,
            },
            "available_terms_hz": {
                "q_e": q_e_source_hz,
                "q_J_pair": q_tj_val,
                "q_H_pair": q_th_val,
            },
            "note": "Internal non-resonant mapping currently closes q_e and a first q_J-like pair feed. A distinct internal q_K mapping is not derived yet; q_H_pair remains available separately as the higher-order pair feed on the minimal carrier.",
        }
        if h_exact_hz is not None:
            item["gaussian_source_exact_sextic_hz"] = {
                "H": h_exact_hz,
                "source": "Gaussian Pickett linear sextic block",
            }
        if gaussian_source_sextic is not None:
            item["gaussian_source_reconstructed_sextic_hz"] = dict(gaussian_source_sextic)
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
                "vector": [0.0, 0.0, q_e_source_hz, 0.0],
                "matrix": [[0.0, q_e_source_hz], [q_e_source_hz, 0.0]],
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
    for item in pairs:
        qeff = item["effective_linear_model_hz"]["constants_hz"]
        item["effective_linear_model_final_hz"] = {
            "source": "internal_nonresonant_minimal",
            "model": item["effective_linear_model_hz"]["model"],
            "constants_hz": {
                "q_e": float(qeff["q_e_source"]),
                "q_J": float(qeff["q_J_pair"]),
                "q_H": float(qeff["q_H_pair"]),
            },
        }

    if gaussian_linear is not None and pairs:
        qe_items = sorted(gaussian_linear.q_e_mhz.items(), key=lambda kv: kv[1])
        qj_items = gaussian_linear.q_j_mhz
        qk_items = gaussian_linear.q_k_mhz
        for item in pairs:
            qspec = item["spectroscopic_linear_constants_hz"]
            target = min(qe_items, key=lambda kv: abs(kv[1] - (float(qspec["q_e_source"]) / 1.0e6)))
            qidx = int(target[0])
            item["gaussian_log_benchmark"] = {
                "Q_index": qidx,
                "q_e_mhz": float(gaussian_linear.q_e_mhz[qidx]),
                "q_J_mhz": float(qj_items.get(qidx, 0.0)),
                "q_K_mhz": float(qk_items.get(qidx, 0.0)),
                "active_dd_22_count": gaussian_linear.active_dd_22_count,
            }
            if gaussian_source_blocks is not None:
                qjk_source_exact = _linear_gaussian_exact_source_qjk_hz(
                    model,
                    gaussian_source_blocks=gaussian_source_blocks,
                    q_index=qidx,
                    d_hz=d_hz,
                    rotor_limit=rotor_limit,
                )
                if qjk_source_exact is not None:
                    item["gaussian_source_rotational_constants_hz"] = qjk_source_exact
                    item["conventional_linear_model_hz"] = {
                        "status": "reconstructed_from_gaussian_source_blocks",
                        "source": str(qjk_source_exact["source"]),
                        "model": "q_i = q_i^e + (q_i^J) J(J+1) + (q_i^K) K(K±1)^2",
                        "constants_hz": {
                            "q_e": float(qspec["q_e_source"]),
                            "q_J": float(qjk_source_exact["q_J_source"]),
                            "q_K": float(qjk_source_exact["q_K_source"]),
                        },
                        "available_terms_hz": {
                            "q_e": float(qspec["q_e_source"]),
                            "q_J_source": float(qjk_source_exact["q_J_source"]),
                            "q_K_source": float(qjk_source_exact["q_K_source"]),
                            "q_J_pair": float(item["effective_linear_model_hz"]["constants_hz"]["q_J_pair"]),
                            "q_H_pair": float(item["effective_linear_model_hz"]["constants_hz"]["q_H_pair"]),
                        },
                        "note": "This conventional-like mapping uses the internal q_e source term together with q_J/q_K reconstructed from Gaussian alpha/cubic source blocks, without relying on the printed RotL2x q^J/q^K constants.",
                    }
            item["gaussian_source_exact_constants_hz"] = {
                "q_e": float(gaussian_linear.q_e_mhz[qidx] * 1.0e6),
                "q_J": float(qj_items.get(qidx, 0.0) * 1.0e6),
                "q_K": float(qk_items.get(qidx, 0.0) * 1.0e6),
                "Q_index": qidx,
                "source": "Gaussian RotL2x printed linear l-type block",
                "active_dd_22_count": gaussian_linear.active_dd_22_count,
            }
            item["effective_linear_model_gaussian_hz"] = {
                "model": "q_i = q_i^e + (q_i^J) J(J+1) + (q_i^K) K(K±1)^2",
                "constants_hz": {
                    "q_e": float(gaussian_linear.q_e_mhz[qidx] * 1.0e6),
                    "q_J": float(qj_items.get(qidx, 0.0) * 1.0e6),
                    "q_K": float(qk_items.get(qidx, 0.0) * 1.0e6),
                },
                "Q_index": qidx,
                "active_dd_22_count": gaussian_linear.active_dd_22_count,
                "note": "This is the exact Gaussian RotL2x source block as printed in the log, before the later resonance-analysis section.",
            }
            item["conventional_linear_model_exact_hz"] = {
                "status": "exact_printed_rotl2x",
                "source": "gaussian_rotl2x_exact",
                "model": item["effective_linear_model_gaussian_hz"]["model"],
                "constants_hz": dict(item["effective_linear_model_gaussian_hz"]["constants_hz"]),
                "Q_index": qidx,
                "active_dd_22_count": gaussian_linear.active_dd_22_count,
                "note": "Exact conventional linear-model constants taken directly from the printed Gaussian RotL2x block.",
            }
            item["effective_linear_model_final_hz"] = {
                "source": "gaussian_rotl2x_exact",
                "model": item["effective_linear_model_gaussian_hz"]["model"],
                "constants_hz": dict(item["effective_linear_model_gaussian_hz"]["constants_hz"]),
                "Q_index": qidx,
                "active_dd_22_count": gaussian_linear.active_dd_22_count,
            }
    pure_rotational_branch = {
        "kind": "linear_pure_rotational",
        "scope_status": "paper_aligned",
        "quartic_special": quartic_special,
        "sextic_special": sextic_special,
        "scalars": {
            "D_mhz": float(d_hz / 1.0e6) if d_hz is not None else None,
            "H_hz": float(h_feed_hz) if h_feed_hz is not None else None,
        },
    }
    if gaussian_source_sextic is not None:
        pure_rotational_branch["gaussian_source_reconstructed_sextic_hz"] = dict(gaussian_source_sextic)
    if h_exact_hz is not None:
        pure_rotational_branch["gaussian_pickett_sextic_hz"] = {
            "H": float(h_exact_hz),
            "source": "Gaussian Pickett linear sextic block",
        }
    pairwise_ltype_branch = {
        "kind": "linear_pairwise_ltype",
        "scope_status": "experimental_beyond_paper",
        "pair_count": len(pairs),
        "active_operator_channel": "X_l",
        "inactive_operator_channels": ["I_l", "Z_l", "Y_l"],
        "driving_scalars": {
            "D_mhz": float(d_hz / 1.0e6) if d_hz is not None else None,
            "H_hz": float(h_feed_hz) if h_feed_hz is not None else None,
        },
        "rotational_feeds": {
            "quartic_feed_operator": "J^2 X_l",
            "sextic_feed_operator": "(J^2)^2 X_l",
            "quartic_feed_present": bool(d_hz is not None),
            "sextic_feed_present": bool(h_feed_hz is not None),
        },
    }
    return {
        "B_linear_cm": b_linear_cm,
        "pairs": pairs,
        "kind": "linear",
        "paper_scope": "pure_rotational_only",
        "implementation_scope": "pure_rotational_plus_experimental_pairwise_ltype",
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
        # In the exact linear limit the perpendicular quartic components
        # collapse onto a single scalar:
        #   tau_yyyy = tau_zzzz = tau_yyzz = D
        # The previous 0.5*tau_yyzz averaging introduced an artificial 5/6
        # factor and underestimated the Gaussian/Pickett linear D constant.
        dval = np.mean([tau["tau_yyyy"], tau["tau_zzzz"], tau["tau_yyzz"]])
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
