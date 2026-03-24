#!/usr/bin/env python3
"""Compare sextic centrifugal distortion constants against Gaussian benchmarks."""

from __future__ import annotations

import argparse
import math
from itertools import product

import numpy as np

from gaussian_vpt_parser import (
    align_gaussian_cubic_force_constants,
    frequency_reorder_map,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
    parse_gaussian_quartic_benchmark,
    parse_gaussian_sextic_benchmark,
)
from rovib_distortion import (
    AMU_TO_AU_MASS,
    ANGSTROM_TO_BOHR,
    AU_FREQ_TO_CMINV,
    CMINV_TO_MHZ,
    coriolis_zeta_tensor,
    harmonic_inertia_model_from_geometry_hessian,
)


ANGSTROM_PER_BOHR = 0.529177210903
CMINV_TO_HZ = 2.99792458e10
AMU_KG = 1.66053906660e-27
PLANCK = 6.62607015e-34
CLIGHT_CM = 2.99792458e10
HARTREE_J = 4.3597447222071e-18
BOHR_ANG = ANGSTROM_PER_BOHR
INV_FACTG = PLANCK / (4.0 * math.pi**2 * CLIGHT_CM) / (AMU_KG * (1.0e-10) ** 2)
FACTG = 1.0 / INV_FACTG
HC_ATTOJ_CM = PLANCK * CLIGHT_CM * 1.0e18
FAC3AU = (HARTREE_J * 1.0e18) / (BOHR_ANG**3 * FACTG * math.sqrt(FACTG) * HC_ATTOJ_CM)
QCENT_CONST1 = INV_FACTG**3
WILSON_TAU_AU_TO_CMINV = 4.0 * AU_FREQ_TO_CMINV
SPECIAL_LIMIT_REL_TOL = 1.0e-6
SPECIAL_LIMIT_ABS_MHZ_TOL = 1.0e-3
DIDQ_AU_PER_AMU_SQRT_ANG = math.sqrt(AMU_TO_AU_MASS) * ANGSTROM_TO_BOHR


def _representation_axis_values(model) -> tuple[np.ndarray, np.ndarray]:
    """Return moments and rotational constants ordered along the model x,y,z axes."""
    abc_to_idx = {"a": 0, "b": 1, "c": 2}
    perm = [abc_to_idx[label] for label in model.xyz_to_abc]
    moments_xyz = model.moments_amu_a2[np.array(perm, dtype=int)]
    rot_xyz_cm = (model.abc_mhz / CMINV_TO_MHZ)[np.array(perm, dtype=int)]
    return moments_xyz, rot_xyz_cm


def _classify_rotor_limit(
    abc_mhz: np.ndarray,
    moments_amu_a2: np.ndarray | None = None,
    *,
    rel_tol: float = SPECIAL_LIMIT_REL_TOL,
    abs_mhz_tol: float = SPECIAL_LIMIT_ABS_MHZ_TOL,
) -> dict[str, object]:
    """Classify the rotor type relevant for symmetry-adapted alpha output."""
    abc = np.asarray(abc_mhz, dtype=float).reshape(3)
    finite = abc[np.isfinite(abc)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0

    def _close(x: float, y: float) -> bool:
        return abs(x - y) <= max(abs_mhz_tol, rel_tol * max(abs(x), abs(y), scale, 1.0))

    if moments_amu_a2 is not None:
        moments = np.asarray(moments_amu_a2, dtype=float).reshape(3)
        if abs(moments[0]) <= max(1.0e-10, rel_tol * max(abs(moments[2]), 1.0)):
            return {
                "kind": "linear",
                "symmetry_axis": "a",
                "degenerate_axes": ("b", "c"),
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


def _detect_near_degenerate_pairs(
    vib_freq_cm: np.ndarray,
    zeta_xyz: np.ndarray | None = None,
    *,
    abs_tol_cm: float = 5.0,
    rel_tol: float = 1.0e-3,
) -> list[tuple[int, int]]:
    """Detect near-degenerate vibrational pairs for symmetry-adapted mode averaging.

    Pairing is driven primarily by frequency proximity. When Coriolis data are
    available, the greedy choice is refined by favoring partners with larger
    inter-mode Coriolis coupling, which is closer to the physically relevant
    degenerate subspace than a pure frequency-only heuristic.
    """
    freqs = np.abs(np.asarray(vib_freq_cm, dtype=float).reshape(-1))
    zeta = None if zeta_xyz is None else np.abs(np.asarray(zeta_xyz, dtype=float))
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(freqs.size):
        if i in used:
            continue
        best_j = None
        best_score = None
        for j in range(i + 1, freqs.size):
            if j in used:
                continue
            diff = abs(freqs[i] - freqs[j])
            tol = max(abs_tol_cm, rel_tol * max(freqs[i], freqs[j], 1.0))
            if diff > tol:
                continue
            if zeta is None:
                score = diff
            else:
                zeta_strength = float(np.max(zeta[:, i, j]))
                score = diff / (1.0 + 10.0 * zeta_strength)
            if best_score is None or score < best_score:
                best_j = j
                best_score = score
        if best_j is not None:
            pairs.append((i, best_j))
            used.add(i)
            used.add(best_j)
    return pairs


def _degenerate_mode_metadata(
    model,
    rotor_limit: dict[str, object],
    *,
    excluded_keep_mask: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Return lightweight Gaussian/Mills-style metadata for near-degenerate pairs.

    The current backend does not reproduce Gaussian's full ``LsDgNM`` machinery,
    but it can expose the essential information needed for symmetry-adapted
    alpha handling:

    - paired mode indices,
    - average harmonic frequency,
    - symmetry axis and degenerate axes,
    - dominant Coriolis-coupling axis for the pair.
    """
    keep = None if excluded_keep_mask is None else np.asarray(excluded_keep_mask, dtype=bool)
    zeta = np.asarray(model.coriolis_zeta_pairs_xyz, dtype=float)
    pairs = _detect_near_degenerate_pairs(model.vib_freq_cm, zeta)
    axis_map = {label: i for i, label in enumerate(model.xyz_to_abc)}
    idx_to_label = {i: lbl for lbl, i in axis_map.items()}
    out: list[dict[str, object]] = []
    for i, j in pairs:
        if keep is not None and (not keep[i] or not keep[j]):
            continue
        z = np.abs(zeta[:, i, j])
        dom_idx = int(np.argmax(z))
        dom_lbl = idx_to_label[dom_idx]
        out.append(
            {
                "pair": (int(i), int(j)),
                "pair_1based": (int(i + 1), int(j + 1)),
                "freq_cm": float(0.5 * (abs(model.vib_freq_cm[i]) + abs(model.vib_freq_cm[j]))),
                "dominant_coriolis_axis": dom_lbl,
                "dominant_coriolis_abs": float(z[dom_idx]),
                "symmetry_axis": rotor_limit.get("symmetry_axis"),
                "degenerate_axes": tuple(rotor_limit.get("degenerate_axes", ())),
                "kind": str(rotor_limit.get("kind")),
            }
        )
    return out


def _lsdgnm_like_metadata(
    model,
    rotor_limit: dict[str, object],
    *,
    excluded_keep_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Return a lightweight CeDiTT-side analogue of Gaussian's ``LsDgNM`` table."""
    keep = None if excluded_keep_mask is None else np.asarray(excluded_keep_mask, dtype=bool)
    pair_meta = _degenerate_mode_metadata(model, rotor_limit, excluded_keep_mask=excluded_keep_mask)
    n_modes = int(np.abs(model.vib_freq_cm).size)
    rows: list[dict[str, object]] = []
    pair_lookup: dict[int, dict[str, object]] = {}
    for meta in pair_meta:
        i, j = meta["pair"]
        pair_lookup[int(i)] = meta
        pair_lookup[int(j)] = meta
    for i in range(n_modes):
        active = True if keep is None else bool(keep[i])
        if i not in pair_lookup:
            rows.append(
                {
                    "mode": int(i),
                    "mode_1based": int(i + 1),
                    "freq_cm": float(abs(model.vib_freq_cm[i])),
                    "active": active,
                    "degeneracy_order": 1,
                    "lsdgnm_code": 1,
                    "pair_role": "single",
                    "reference_mode": int(i),
                    "reference_mode_1based": int(i + 1),
                    "partner_mode": None,
                    "partner_mode_1based": None,
                    "dominant_coriolis_axis": None,
                }
            )
            continue
        meta = pair_lookup[i]
        i0, i1 = meta["pair"]
        is_leader = int(i) == int(i0)
        rows.append(
            {
                "mode": int(i),
                "mode_1based": int(i + 1),
                "freq_cm": float(abs(model.vib_freq_cm[i])),
                "active": active,
                "degeneracy_order": 2,
                "lsdgnm_code": 2 if is_leader else -21,
                "pair_role": "leader" if is_leader else "follower",
                "reference_mode": int(i0),
                "reference_mode_1based": int(i0 + 1),
                "partner_mode": int(i1 if is_leader else i0),
                "partner_mode_1based": int((i1 if is_leader else i0) + 1),
                "dominant_coriolis_axis": meta["dominant_coriolis_axis"],
            }
        )
    return {
        "ntotnm": n_modes,
        "ndeg_pairs": len(pair_meta),
        "pair_table": pair_meta,
        "mode_table": rows,
    }


def _mode_projector_from_lsdgnm_like(lsdgnm_like: dict[str, object], n_modes: int) -> np.ndarray:
    """Return the symmetry projector acting on the mode index."""
    p = np.eye(n_modes, dtype=float)
    if lsdgnm_like is None:
        return p
    used: set[int] = set()
    for row in lsdgnm_like["mode_table"]:
        mode = int(row["mode"])
        if mode in used:
            continue
        if int(row["degeneracy_order"]) <= 1:
            continue
        ref = int(row["reference_mode"])
        partner = row["partner_mode"]
        if partner is None:
            continue
        partner = int(partner)
        block = np.zeros((n_modes, n_modes), dtype=float)
        block[ref, ref] = 0.5
        block[ref, partner] = 0.5
        block[partner, ref] = 0.5
        block[partner, partner] = 0.5
        p[ref, :] = 0.0
        p[partner, :] = 0.0
        p[:, ref] = 0.0
        p[:, partner] = 0.0
        p += block
        used.add(ref)
        used.add(partner)
    return p


def _axis_projector_for_rotor_limit(model, rotor_limit: dict[str, object]) -> np.ndarray:
    """Return the symmetry projector acting on the xyz-axis index."""
    p = np.eye(3, dtype=float)
    axis_map = {label: i for i, label in enumerate(model.xyz_to_abc)}
    if not rotor_limit["is_special_limit"]:
        return p
    deg_axes = tuple(axis_map[label] for label in rotor_limit.get("degenerate_axes", ()))
    sym_axis_label = rotor_limit.get("symmetry_axis")
    sym_idx = axis_map[sym_axis_label] if sym_axis_label is not None else None
    if len(deg_axes) == 2:
        p = np.zeros((3, 3), dtype=float)
        if sym_idx is not None:
            p[sym_idx, sym_idx] = 1.0
        p[deg_axes[0], deg_axes[0]] = 0.5
        p[deg_axes[0], deg_axes[1]] = 0.5
        p[deg_axes[1], deg_axes[0]] = 0.5
        p[deg_axes[1], deg_axes[1]] = 0.5
        if rotor_limit.get("kind") == "linear" and sym_idx is not None:
            p[sym_idx, :] = 0.0
            p[:, sym_idx] = 0.0
    return p


def _degenerate_mode_feature_matrix(c1: np.ndarray, model, rotor_limit: dict[str, object]) -> np.ndarray:
    """Return mode features used to canonicalize near-degenerate two-mode subspaces.

    The features are built from first-level tensor components that distinguish
    the symmetry axis from the degenerate perpendicular plane. They are
    therefore closer to the CeDiTT tensor language than a purely numerical
    comparison of mode vectors.
    """
    n_modes = c1.shape[0]
    feats = np.zeros((n_modes, 4), dtype=float)
    if not rotor_limit["is_special_limit"]:
        return feats
    abc_to_idx = {"a": 0, "b": 1, "c": 2}
    xyz_idx = {lbl: i for i, lbl in enumerate(model.xyz_to_abc)}
    sym_lbl = rotor_limit.get("symmetry_axis")
    deg_lbls = tuple(rotor_limit.get("degenerate_axes", ()))
    if sym_lbl is None or len(deg_lbls) != 2:
        return feats
    s = xyz_idx[str(sym_lbl)]
    p = xyz_idx[str(deg_lbls[0])]
    q = xyz_idx[str(deg_lbls[1])]
    for i in range(n_modes):
        feats[i, 0] = c1[i, s, p]
        feats[i, 1] = c1[i, s, q]
        feats[i, 2] = c1[i, p, p] - c1[i, q, q]
        feats[i, 3] = 2.0 * c1[i, p, q]
    return feats


def _canonicalize_pair_rows(
    rows: np.ndarray,
    feature_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize a two-mode subspace from tensorial feature rows.

    Returns the rotated rows, the orthogonal 2x2 rotation matrix, and the two
    singular values associated with the feature content of the pair.
    """
    gram = np.asarray(feature_rows, dtype=float) @ np.asarray(feature_rows, dtype=float).T
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    rotated = evecs.T @ np.asarray(rows, dtype=float)
    if np.linalg.norm(rotated[0]) < np.linalg.norm(rotated[1]):
        swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        evecs = evecs @ swap
        rotated = swap @ rotated
        evals = evals[::-1]
    # Fix an arbitrary but stable sign convention on the leading row.
    lead = rotated[0]
    nz = np.where(np.abs(lead) > 1.0e-12)[0]
    if nz.size and lead[nz[0]] < 0.0:
        rotated[0] *= -1.0
        evecs[:, 0] *= -1.0
    return rotated, evecs, np.sqrt(np.clip(evals, 0.0, None))


def _special_alpha_context(
    model,
    rotor_limit: dict[str, object],
    *,
    excluded_keep_mask: np.ndarray | None = None,
    c1: np.ndarray | None = None,
) -> dict[str, object]:
    """Return reusable symmetry data for alpha-like observables."""
    n_modes = int(np.abs(model.vib_freq_cm).size)
    lsdgnm_like = _lsdgnm_like_metadata(model, rotor_limit, excluded_keep_mask=excluded_keep_mask)
    deg_meta = list(lsdgnm_like["pair_table"])
    mode_p = _mode_projector_from_lsdgnm_like(lsdgnm_like, n_modes)
    axis_p = _axis_projector_for_rotor_limit(model, rotor_limit)
    canonical_pair_data: list[dict[str, object]] = []
    if c1 is not None and deg_meta:
        features = _degenerate_mode_feature_matrix(c1, model, rotor_limit)
        for meta in deg_meta:
            i, j = meta["pair"]
            feat_pair = np.asarray(features[[i, j], :], dtype=float)
            _, rot, sval = _canonicalize_pair_rows(np.eye(2), feat_pair)
            canonical_pair_data.append(
                {
                    "pair": meta["pair"],
                    "pair_1based": meta["pair_1based"],
                    "rotation_2x2": rot,
                    "feature_singular_values": sval,
                }
            )
    return {
        "mode_projector": mode_p,
        "axis_projector": axis_p,
        "strategy": "symmetry_projector",
        "lsdgnm_like": lsdgnm_like,
        "degenerate_mode_metadata": deg_meta,
        "degenerate_mode_pairs": [tuple(meta["pair"]) for meta in deg_meta],
        "canonical_pair_data": canonical_pair_data,
    }


def _apply_special_alpha_projection(alpha_xyz_cm: np.ndarray, special_ctx: dict[str, object]) -> np.ndarray:
    """Apply a precomputed symmetry projector to an alpha-like mode/axis observable."""
    return np.asarray(special_ctx["mode_projector"], dtype=float) @ np.asarray(alpha_xyz_cm, dtype=float) @ np.asarray(
        special_ctx["axis_projector"], dtype=float
    )


def _canonicalize_alpha_rows(alpha_xyz_cm: np.ndarray, special_ctx: dict[str, object]) -> np.ndarray:
    """Return a mode-canonicalized version of the alpha rows on degenerate pairs."""
    out = np.array(alpha_xyz_cm, dtype=float, copy=True)
    for meta in special_ctx.get("canonical_pair_data", ()):
        i, j = meta["pair"]
        rot = np.asarray(meta["rotation_2x2"], dtype=float)
        out[[i, j], :] = rot.T @ out[[i, j], :]
    return out


def _didq_from_model(model) -> np.ndarray:
    """Replicate Gaussian dPMdQ in the chosen representation."""
    n_atoms = model.masses_amu.size
    modes = model.vib_vecs_mw_pa.reshape(n_atoms, 3, -1)
    didq = np.zeros((6, modes.shape[2]), dtype=float)
    ijx = 0
    for ix in range(3):
        for jx in range(ix + 1):
            for mode in range(modes.shape[2]):
                acc = 0.0
                for atom in range(n_atoms):
                    mass = math.sqrt(model.masses_amu[atom])
                    acc -= mass * model.coords_pa_ang[atom, ix] * modes[atom, jx, mode]
                    if ix == jx:
                        for kx in range(3):
                            acc += mass * model.coords_pa_ang[atom, kx] * modes[atom, kx, mode]
                didq[ijx, mode] = 2.0 * acc
            ijx += 1
    return didq


def _aligned_model_from_gaussian(fchk_path: str, log_path: str):
    """Return a model aligned to Gaussian's dIdQ convention by rep/order/sign search."""
    fchk = parse_gaussian_fchk_harmonic_data(fchk_path)
    quart = parse_gaussian_quartic_benchmark(log_path)
    anh = parse_gaussian_anharmonic_force_data(log_path)
    if quart.didq_amu_sqrt_ang is None:
        raise ValueError("Gaussian dIdQ block is required for sextic alignment.")

    best = None
    for rep in ("I", "II", "III"):
        trial = harmonic_inertia_model_from_geometry_hessian(
            fchk.masses_amu,
            fchk.coords_bohr * ANGSTROM_PER_BOHR,
            fchk.cartesian_force_constants,
            representation=rep,
        )
        order = frequency_reorder_map(anh.frequencies_cm, np.abs(trial.vib_freq_cm))
        for signs in product((1.0, -1.0), repeat=trial.vib_freq_cm.size):
            model = harmonic_inertia_model_from_geometry_hessian(
                fchk.masses_amu,
                fchk.coords_bohr * ANGSTROM_PER_BOHR,
                fchk.cartesian_force_constants,
                representation=rep,
            )
            sign_vec = np.array(signs, dtype=float)
            model.vib_vecs_mw_pa[:] = model.vib_vecs_mw_pa[:, order] * sign_vec
            model.vib_freq_cm[:] = model.vib_freq_cm[order]
            model.coriolis_g_au[:] = model.coriolis_g_au[:, order] * sign_vec[None, :]
            model.coriolis_zeta_pairs_xyz[:] = coriolis_zeta_tensor(model.vib_vecs_mw_pa)
            # Keep all mode-indexed harmonic tensors in the same reordered/phase-fixed convention.
            model.dI_au[:] = model.dI_au[:, :, order] * sign_vec[None, None, :]
            model.dInv_au[:] = model.dInv_au[:, :, order] * sign_vec[None, None, :]
            pair_sign = sign_vec[:, None] * sign_vec[None, :]
            model.d2I_au[:] = model.d2I_au[:, :, order][:, :, :, order] * pair_sign[None, None, :, :]
            model.d2Inv_bilinear_au[:] = (
                model.d2Inv_bilinear_au[:, :, order][:, :, :, order] * pair_sign[None, None, :, :]
            )
            model.d2Inv_intrinsic_au[:] = (
                model.d2Inv_intrinsic_au[:, :, order][:, :, :, order] * pair_sign[None, None, :, :]
            )
            model.d2Inv_au[:] = model.d2Inv_au[:, :, order][:, :, :, order] * pair_sign[None, None, :, :]
            didq = _didq_from_model(model)
            err = float(np.linalg.norm(didq - quart.didq_amu_sqrt_ang))
            if best is None or err < best[0]:
                best = (err, rep, order.copy(), sign_vec.copy(), model)
    assert best is not None
    return best[4], best[1], best[2], best[3], best[0]


def _idx_tm(a: int, b: int) -> int:
    pair = (max(a, b), min(a, b))
    idx_map = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (2, 0): 3,
        (2, 1): 4,
        (2, 2): 5,
    }
    return idx_map[pair]


def _quartic_tau_from_model(model) -> tuple[np.ndarray, np.ndarray]:
    """Return Tau(ix,jx,kx,lx) and Tau' in cm^-1."""
    didq = _didq_from_model(model)
    freq_cm = np.abs(model.vib_freq_cm)
    pmom, _ = _representation_axis_values(model)
    tau = np.zeros((3, 3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            ab = _idx_tm(a, b)
            for c in range(3):
                for d in range(3):
                    cd = _idx_tm(c, d)
                    val = 0.0
                    for k in range(freq_cm.size):
                        denom = freq_cm[k] ** 2 * pmom[a] * pmom[b] * pmom[c] * pmom[d]
                        val += didq[ab, k] * didq[cd, k] / denom
                    tau[a, b, c, d] = -0.5 * QCENT_CONST1 * val

    tau_prime = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            tau_prime[i, j] = tau[i, i, j, j]
            if i != j:
                tau_prime[i, j] += 2.0 * tau[i, j, i, j]
    return tau, tau_prime


def _quartic_tau_from_c1(c1: np.ndarray, freq_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Tau(ix,jx,kx,lx) and Tau' in cm^-1 from the first-level tensor ``c1``.

    Using the current normalization of ``c1``, the quartic tensor is

        tau_abcd = -2 sum_i nu_i c1_i,ab c1_i,cd.
    """
    tau = np.zeros((3, 3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    val = 0.0
                    for i in range(freq_cm.size):
                        val += -2.0 * abs(freq_cm[i]) * c1[i, a, b] * c1[i, c, d]
                    tau[a, b, c, d] = val

    tau_prime = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            tau_prime[i, j] = tau[i, i, j, j]
            if i != j:
                tau_prime[i, j] += 2.0 * tau[i, j, i, j]
    return tau, tau_prime


def _quartic_h22_tau_from_mu2(mu2_au: np.ndarray, vib_freq_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the H22 Wilson quartic tensor and Tau' in cm^-1.

    The direct H22 contraction is naturally obtained from the atomic-unit
    second inverse-inertia derivative. The resulting Wilson tensor is converted
    to the cm^-1 convention used by the Gaussian-style sextic machinery.
    """
    omega_au = np.abs(vib_freq_cm) / AU_FREQ_TO_CMINV
    n_modes = omega_au.size
    tau_au = np.zeros((3, 3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    coeff = 0.0
                    for k in range(n_modes):
                        for l in range(n_modes):
                            weight = 2.0 if k == l else 1.0
                            coeff += 0.03125 * weight * mu2_au[a, b, k, l] * mu2_au[c, d, k, l] / (omega_au[k] * omega_au[l])
                    tau_au[a, b, c, d] = coeff

    tau = WILSON_TAU_AU_TO_CMINV * tau_au
    tau_prime = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            tau_prime[i, j] = tau[i, i, j, j]
            if i != j:
                tau_prime[i, j] += 2.0 * tau[i, j, i, j]
    return tau, tau_prime


def _quartic_h22_tau_from_model(model) -> tuple[np.ndarray, np.ndarray]:
    """Return the H22 Wilson quartic tensor and Tau' in cm^-1 from the harmonic model."""
    return _quartic_h22_tau_from_mu2(np.asarray(model.d2Inv_au, dtype=float), np.asarray(model.vib_freq_cm, dtype=float))


def _zeta_xyz(model) -> np.ndarray:
    """Return Coriolis couplings as zeta[axis, i, j] in the chosen representation."""
    return np.array(model.coriolis_zeta_pairs_xyz, copy=True)


def _c1_matrix(didq: np.ndarray, freq_cm: np.ndarray, pmom: np.ndarray) -> np.ndarray:
    n_modes = freq_cm.size
    out = np.zeros((n_modes, 3, 3), dtype=float)
    for i in range(n_modes):
        x = math.sqrt((FACTG * abs(freq_cm[i])) ** 3)
        for ix in range(3):
            for jx in range(3):
                den = 2.0 * pmom[ix] * pmom[jx] * x
                out[i, ix, jx] = 0.0 if abs(den) <= 1.0e-30 else didq[_idx_tm(ix, jx), i] / den
    return out


def _mu1_mode_matrices(model) -> np.ndarray:
    """Return ``mu1[i,a,b] = d(I^-1)_{ab}/dQ_i`` from the harmonic model."""
    return np.transpose(np.asarray(model.dInv_au, dtype=float), (2, 0, 1))


def _modewise_proportionality_scalars(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Return per-mode least-squares scalars for ``target[i] ~= s_i source[i]``."""
    if target.shape != source.shape or target.ndim < 1:
        raise ValueError(f"target/source shape mismatch: {target.shape} vs {source.shape}")
    n_modes = target.shape[0]
    scales = np.zeros(n_modes, dtype=float)
    for i in range(n_modes):
        src = source[i].reshape(-1)
        tgt = target[i].reshape(-1)
        finite = np.isfinite(src) & np.isfinite(tgt)
        src = src[finite]
        tgt = tgt[finite]
        den = float(np.dot(src, src))
        scales[i] = 0.0 if den <= 1.0e-30 else float(np.dot(tgt, src) / den)
    return scales


def _apply_modewise_scalars(source: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Return ``out[i] = scales[i] * source[i]``."""
    if source.shape[0] != scales.shape[0]:
        raise ValueError(f"mode axis mismatch: {source.shape[0]} vs {scales.shape[0]}")
    return source * scales[:, None, None]


def _c1_from_mu1(model) -> np.ndarray:
    """Return the first-level tensor ``c1`` built from the model ``mu1`` tensor.

    In the present code base, the validated sextic machinery already shows that
    the standard first-level geometry is equivalently carried by ``mu1`` and by
    ``c1`` up to a modewise scalar factor. This helper reconstructs ``c1`` from
    ``mu1`` through that modewise proportionality, so that downstream formulas
    do not need to use ``dI/dQ`` explicitly.
    """
    pmom, _ = _representation_axis_values(model)
    mu1 = _mu1_mode_matrices(model)
    i0 = np.asarray(model.i_tensor_au, dtype=float)
    n_modes = mu1.shape[0]
    didq_like = np.zeros((6, n_modes), dtype=float)
    for k in range(n_modes):
        dI_from_mu1 = -i0 @ mu1[k] @ i0
        for ix in range(3):
            for jx in range(ix + 1):
                didq_like[_idx_tm(ix, jx), k] = dI_from_mu1[ix, jx] / DIDQ_AU_PER_AMU_SQRT_ANG
    return _c1_matrix(didq_like, np.abs(model.vib_freq_cm), pmom)


def _c2_modepair_tensor(c1: np.ndarray, zeta_xyz: np.ndarray, freq_cm: np.ndarray, rot_cm: np.ndarray) -> np.ndarray:
    """Return the unsummed mode-pair tensor underlying ``c2``.

    The current sextic code defines ``c2[i, a, b, c]`` only after summing over
    an internal mode ``j``. This helper exposes the pre-contraction object

        c2_pair[i, j, a, b, c]

    whose sum over ``j`` reproduces ``c2``.
    """
    n_modes = freq_cm.size
    out = np.zeros((n_modes, n_modes, 3, 3, 3), dtype=float)
    for i in range(n_modes):
        frq_i = abs(freq_cm[i])
        if frq_i <= 1.0e-12:
            continue
        for j in range(n_modes):
            frq_j = abs(freq_cm[j])
            if frq_j <= 1.0e-12:
                continue
            kernel = (2.0 / 3.0) * (frq_i**2 + 2.0 * frq_j**2) / math.sqrt(abs(frq_i**5 * frq_j))
            for ix in range(3):
                for jx in range(3):
                    for kx in range(3):
                        out[i, j, ix, jx, kx] = kernel * (
                            rot_cm[ix] * zeta_xyz[ix, i, j] * c1[j, jx, kx]
                            + rot_cm[jx] * zeta_xyz[jx, i, j] * c1[j, kx, ix]
                            + rot_cm[kx] * zeta_xyz[kx, i, j] * c1[j, ix, jx]
                        )
    return out


def _c2_modepair_from_mu1_scaled(
    mu1: np.ndarray,
    mu1_to_c1_scales: np.ndarray,
    zeta_xyz: np.ndarray,
    freq_cm: np.ndarray,
    rot_cm: np.ndarray,
) -> np.ndarray:
    """Build the sextic mode-pair object using mode-scaled ``mu1`` in place of ``c1``."""
    c1_like = _apply_modewise_scalars(mu1, mu1_to_c1_scales)
    return _c2_modepair_tensor(c1_like, zeta_xyz, freq_cm, rot_cm)


def _c2_tensor(c1: np.ndarray, zeta_xyz: np.ndarray, freq_cm: np.ndarray, rot_cm: np.ndarray) -> np.ndarray:
    return np.sum(_c2_modepair_tensor(c1, zeta_xyz, freq_cm, rot_cm), axis=1)


def _f3_reduced(phi3_reduced_cm: np.ndarray, freq_cm: np.ndarray, i: int, j: int, k: int) -> float:
    return phi3_reduced_cm[i, j, k]


def split_cubic_force_constants(phi3_reduced_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split reduced cubic force constants into semi-diagonal and 3-index sectors.

    The semi-diagonal sector collects entries with at least two equal mode
    indices. The remainder contains only terms with three distinct indices.
    """
    phi3 = np.asarray(phi3_reduced_cm, dtype=float)
    if phi3.ndim != 3 or phi3.shape[0] != phi3.shape[1] or phi3.shape[1] != phi3.shape[2]:
        raise ValueError(f"phi3 must be a cubic mode tensor, got shape {phi3.shape}")
    sd = np.zeros_like(phi3)
    tri = np.zeros_like(phi3)
    n_modes = phi3.shape[0]
    for i in range(n_modes):
        for j in range(n_modes):
            for k in range(n_modes):
                if i == j or j == k or i == k:
                    sd[i, j, k] = phi3[i, j, k]
                else:
                    tri[i, j, k] = phi3[i, j, k]
    return sd, tri


def sextic_phi_cartesian_cm(model, phi3_reduced_cm: np.ndarray, axes: dict[str, int] | None = None) -> dict[str, float]:
    """Replicate the non-linear-top L717 sextic Cartesian Phi tensor in cm^-1."""
    freq_cm = np.abs(model.vib_freq_cm)
    pmom, rot_cm = _representation_axis_values(model)
    didq = _didq_from_model(model)
    tau, tau_prime = _quartic_tau_from_model(model)
    zeta = _zeta_xyz(model)
    c1 = _c1_matrix(didq, freq_cm, pmom)
    c2 = _c2_tensor(c1, zeta, freq_cm, rot_cm)

    t = tau_prime / 4.0
    tol = 1.0e-12
    scc = np.zeros((3, 3, 3), dtype=float)

    for ix in range(3):
        x1 = sum(tau[ix, ix, ix, jx] ** 2 / rot_cm[jx] for jx in range(3)) * 3.0 / 16.0
        x2 = 0.5 * sum(freq_cm[i] * c2[i, ix, ix, ix] ** 2 for i in range(freq_cm.size))
        x3 = 0.0
        for i in range(freq_cm.size):
            for j in range(freq_cm.size):
                for k in range(freq_cm.size):
                    x3 += _f3_reduced(phi3_reduced_cm, freq_cm, i, j, k) * c1[i, ix, ix] * c1[j, ix, ix] * c1[k, ix, ix]
        x3 /= 6.0
        x4 = 0.0
        for jx in range(3):
            if jx == ix:
                continue
            den = rot_cm[ix] - rot_cm[jx]
            if abs(den) > tol:
                x4 += tau[jx, ix, ix, ix] ** 2 / den
        x4 /= 4.0
        scc[ix, ix, ix] = x1 - x2 + x3 + x4

    for ix in range(3):
        for jx in range(3):
            if ix == jx:
                continue
            y1 = 0.0
            for kx in range(3):
                x1 = (tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx]) ** 2
                x2 = tau[jx, jx, ix, kx] + 2.0 * tau[ix, jx, jx, kx]
                y1 += (x1 + 2.0 * tau[ix, ix, ix, kx] * x2) / rot_cm[kx]
            y1 *= 3.0 / 32.0

            y2 = 0.0
            y3 = 0.0
            for i in range(freq_cm.size):
                x1 = 3.0 * c2[i, ix, ix, jx] ** 2 + 2.0 * c2[i, ix, ix, ix] * c2[i, ix, jx, jx]
                y2 += freq_cm[i] * x1
                for j in range(freq_cm.size):
                    x1 = c1[i, ix, ix] * c1[j, jx, jx] + 4.0 * c1[i, ix, jx] * c1[j, ix, jx]
                    for k in range(freq_cm.size):
                        y3 += _f3_reduced(phi3_reduced_cm, freq_cm, i, j, k) * c1[k, ix, ix] * x1
            y2 *= 3.0 / 4.0
            y3 /= 4.0

            x = 8.0 * (rot_cm[ix] - rot_cm[jx])
            y4 = 0.0
            if abs(x) > tol:
                x1 = 4.0 * tau[jx, jx, jx, ix] - 3.0 * tau[ix, ix, ix, jx]
                y4 = tau[ix, ix, ix, jx] * x1 / x

            y5 = 0.0
            y6 = 0.0
            for kx in range(3):
                if kx == ix or kx == jx:
                    continue
                den1 = 4.0 * (rot_cm[ix] - rot_cm[kx]) ** 2
                if den1 > tol:
                    x1 = (rot_cm[ix] - rot_cm[kx]) * (tau[jx, jx, ix, kx] + 2.0 * tau[jx, ix, jx, kx])
                    x2 = (rot_cm[ix] - rot_cm[jx]) * (tau[kx, kx, kx, ix] - tau[ix, ix, ix, kx])
                    y5 += tau[ix, ix, ix, kx] * (x1 + x2) / den1
                den2 = 8.0 * (rot_cm[jx] - rot_cm[kx]) ** 2
                if den2 > tol:
                    x1 = (rot_cm[jx] - rot_cm[kx]) * (tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx])
                    x2 = 2.0 * (rot_cm[ix] - rot_cm[jx]) * (tau[jx, jx, jx, kx] - tau[kx, kx, kx, jx])
                    x3 = tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx]
                    y6 += x3 * (x1 + x2) / den2
            y = y1 - y2 + y3 + y4 + y5 + y6
            scc[jx, ix, ix] = y
            scc[ix, jx, ix] = y
            scc[ix, ix, jx] = y

    y1 = 0.0
    for ix in range(3):
        x1 = 2.0 * (tau[ix, 0, 1, 2] + tau[ix, 1, 2, 0] + tau[ix, 2, 0, 1]) ** 2
        x2 = (tau[ix, 0, 1, 1] + 2.0 * tau[ix, 1, 0, 1]) * (tau[ix, 0, 2, 2] + 2.0 * tau[ix, 2, 0, 2])
        x3 = (tau[ix, 1, 2, 2] + 2.0 * tau[ix, 2, 1, 2]) * (tau[ix, 1, 0, 0] + 2.0 * tau[ix, 0, 1, 0])
        x4 = (tau[ix, 2, 0, 0] + 2.0 * tau[ix, 0, 2, 0]) * (tau[ix, 2, 1, 1] + 2.0 * tau[ix, 1, 2, 1])
        y1 += (x1 + x2 + x3 + x4) / rot_cm[ix]
    y1 *= 3.0 / 16.0

    y2 = 0.0
    y3 = 0.0
    for i in range(freq_cm.size):
        x1 = (
            2.0 * c2[i, 0, 1, 2] ** 2
            + c2[i, 1, 1, 0] * c2[i, 2, 2, 0]
            + c2[i, 2, 2, 1] * c2[i, 0, 0, 1]
            + c2[i, 0, 0, 2] * c2[i, 1, 1, 2]
        )
        y2 += freq_cm[i] * x1
        for j in range(freq_cm.size):
            for k in range(freq_cm.size):
                x1 = (
                    c1[i, 0, 0] * c1[j, 1, 1] * c1[k, 2, 2]
                    + 2.0 * c1[i, 0, 0] * c1[j, 1, 2] * c1[k, 1, 2]
                    + 2.0 * c1[i, 1, 1] * c1[j, 2, 0] * c1[k, 2, 0]
                    + 2.0 * c1[i, 2, 2] * c1[j, 0, 1] * c1[k, 0, 1]
                    + 8.0 * c1[i, 1, 2] * c1[j, 2, 0] * c1[k, 0, 1]
                )
                y3 += _f3_reduced(phi3_reduced_cm, freq_cm, i, j, k) * x1
    y2 *= 9.0 / 2.0
    y3 /= 2.0

    y4 = 0.0
    for ix in range(3):
        for jx in range(3):
            if jx == ix:
                continue
            for kx in range(3):
                if kx == ix or kx == jx:
                    continue
                den = 4.0 * (rot_cm[jx] - rot_cm[kx]) ** 2
                if den > tol:
                    x1 = (rot_cm[kx] - rot_cm[ix]) * tau[jx, jx, jx, kx]
                    x2 = (rot_cm[ix] - rot_cm[jx]) * tau[kx, kx, kx, jx]
                    y4 += (tau[jx, jx, jx, kx] - tau[kx, kx, kx, jx]) * (x1 + x2) / den
    y4 *= 3.0 / 2.0
    y = y1 - y2 + y3 + y4
    for p in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        scc[p] = y

    if axes is None:
        axes = {"a": 0, "b": 1, "c": 2}
    ia = axes["a"]
    ib = axes["b"]
    ic = axes["c"]
    return {
        "aaa": scc[ia, ia, ia],
        "aab": scc[ia, ia, ib],
        "aac": scc[ia, ia, ic],
        "abb": scc[ia, ib, ib],
        "abc": scc[ia, ib, ic],
        "acc": scc[ia, ic, ic],
        "bbb": scc[ib, ib, ib],
        "bbc": scc[ib, ib, ic],
        "bcc": scc[ib, ic, ic],
        "ccc": scc[ic, ic, ic],
    }


def sextic_linear_source_formula_hz(
    model,
    phi3_reduced_cm: np.ndarray,
) -> dict[str, object]:
    """Replicate the dedicated linear-top ``Sextic`` source branch.

    In ``l717.F`` the linear case does not use the generic non-linear-top
    ``Phi^aab`` / ``Phi^abc`` scaffolds.  Instead, only the perpendicular
    diagonal branch survives and the final constant is

        He = Phi(1,1,1),

    where ``1`` labels one of the two transverse axes of the linear rotor.
    Numerically the two transverse components should coincide in the exact
    linear limit, so we report both and their average.
    """
    freq_cm = np.abs(model.vib_freq_cm)
    pmom, rot_cm = _representation_axis_values(model)
    didq = _didq_from_model(model)
    tau, _ = _quartic_tau_from_model(model)
    zeta = _zeta_xyz(model)
    c1 = _c1_matrix(didq, freq_cm, pmom)
    c2 = _c2_tensor(c1, zeta, freq_cm, rot_cm)

    abc_to_xyz = {label: i for i, label in enumerate(model.xyz_to_abc)}
    sym_idx = abc_to_xyz.get("a", 0)
    perp = [i for i in range(3) if i != sym_idx]
    idx_to_abc = {i: label for label, i in abc_to_xyz.items()}

    comp_hz: dict[str, float] = {}
    for ix in perp:
        x1 = 0.0
        for jx in perp:
            x1 += tau[ix, ix, ix, jx] ** 2 / rot_cm[jx]
        x1 *= 3.0 / 16.0

        x2 = 0.0
        x3 = 0.0
        for i in range(freq_cm.size):
            x2 += freq_cm[i] * c2[i, ix, ix, ix] ** 2
            for j in range(freq_cm.size):
                for k in range(freq_cm.size):
                    x3 += (
                        phi3_reduced_cm[i, j, k]
                        * c1[i, ix, ix]
                        * c1[j, ix, ix]
                        * c1[k, ix, ix]
                    )
        x2 /= 2.0
        x3 /= 6.0
        comp_hz[idx_to_abc[ix] * 3] = (x1 - x2 + x3) * CMINV_TO_HZ

    values = list(comp_hz.values())
    h_hz = float(np.mean(values)) if values else 0.0
    spread_hz = float(max(values) - min(values)) if len(values) >= 2 else 0.0
    return {
        "H": h_hz,
        "perpendicular_components_hz": comp_hz,
        "spread_hz": spread_hz,
        "formula": "He = Phi(1,1,1) from the dedicated linear-top branch of L717/Sextic",
    }


def sextic_breakdown_hz(model, phi3_reduced_cm: np.ndarray, axes: dict[str, int]) -> dict[str, dict[str, float]]:
    """Return selected sextic-term breakdowns in Hz for debugging."""
    freq_cm = np.abs(model.vib_freq_cm)
    pmom, rot_cm = _representation_axis_values(model)
    didq = _didq_from_model(model)
    zeta = _zeta_xyz(model)
    c1 = _c1_matrix(didq, freq_cm, pmom)
    tau, _ = _quartic_tau_from_c1(c1, freq_cm)
    c2 = _c2_tensor(c1, zeta, freq_cm, rot_cm)
    return sextic_breakdown_from_c1_hz(c1, c2, tau, phi3_reduced_cm, freq_cm, rot_cm, axes)


def sextic_breakdown_from_c1_hz(
    c1: np.ndarray,
    c2: np.ndarray,
    tau: np.ndarray,
    phi3_reduced_cm: np.ndarray,
    freq_cm: np.ndarray,
    rot_cm: np.ndarray,
    axes: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Return selected sextic-term breakdowns from precomputed c1/c2/tau."""
    tol = 1.0e-12

    def component_key(indices: tuple[int, int, int]) -> str:
        counts = {axes["a"]: 0, axes["b"]: 0, axes["c"]: 0}
        for idx in indices:
            counts[idx] += 1
        return "".join(
            label * counts[axes[label]]
            for label in ("a", "b", "c")
            if counts[axes[label]] > 0
        )

    ia = axes["a"]
    ib = axes["b"]
    ic = axes["c"]
    out: dict[str, dict[str, float]] = {}

    for ix in range(3):
        x1 = sum(tau[ix, ix, ix, jx] ** 2 / rot_cm[jx] for jx in range(3)) * 3.0 / 16.0
        x2 = 0.5 * sum(freq_cm[i] * c2[i, ix, ix, ix] ** 2 for i in range(freq_cm.size))
        x3 = 0.0
        for i in range(freq_cm.size):
            for j in range(freq_cm.size):
                for k in range(freq_cm.size):
                    x3 += _f3_reduced(phi3_reduced_cm, freq_cm, i, j, k) * c1[i, ix, ix] * c1[j, ix, ix] * c1[k, ix, ix]
        x3 /= 6.0
        x4 = 0.0
        for jx in range(3):
            if jx == ix:
                continue
            den = rot_cm[ix] - rot_cm[jx]
            if abs(den) > tol:
                x4 += tau[jx, ix, ix, ix] ** 2 / den
        x4 /= 4.0
        out[component_key((ix, ix, ix))] = {
            "X1": x1 * CMINV_TO_HZ,
            "X2": x2 * CMINV_TO_HZ,
            "X3": x3 * CMINV_TO_HZ,
            "X4": x4 * CMINV_TO_HZ,
            "phi": (x1 - x2 + x3 + x4) * CMINV_TO_HZ,
        }

    for ix in range(3):
        for jx in range(3):
            if ix == jx:
                continue
            name = component_key((ix, ix, jx))
            y1 = 0.0
            for kx in range(3):
                t1 = tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx]
                t2 = tau[jx, jx, ix, kx] + 2.0 * tau[ix, jx, jx, kx]
                y1 += (t1**2 + 2.0 * tau[ix, ix, ix, kx] * t2) / rot_cm[kx]
            y1 *= 3.0 / 32.0

            y2 = 0.0
            y3 = 0.0
            for i in range(freq_cm.size):
                y2 += freq_cm[i] * (3.0 * c2[i, ix, ix, jx] ** 2 + 2.0 * c2[i, ix, ix, ix] * c2[i, ix, jx, jx])
                for j in range(freq_cm.size):
                    t1 = c1[i, ix, ix] * c1[j, jx, jx] + 4.0 * c1[i, ix, jx] * c1[j, ix, jx]
                    for k in range(freq_cm.size):
                        y3 += _f3_reduced(phi3_reduced_cm, freq_cm, i, j, k) * c1[k, ix, ix] * t1
            y2 *= 3.0 / 4.0
            y3 /= 4.0

            y4 = 0.0
            den = 8.0 * (rot_cm[ix] - rot_cm[jx])
            if abs(den) > tol:
                y4 = tau[ix, ix, ix, jx] * (4.0 * tau[jx, jx, jx, ix] - 3.0 * tau[ix, ix, ix, jx]) / den

            y5 = 0.0
            y6 = 0.0
            for kx in range(3):
                if kx == ix or kx == jx:
                    continue
                den1 = 4.0 * (rot_cm[ix] - rot_cm[kx]) ** 2
                if den1 > tol:
                    t1 = (rot_cm[ix] - rot_cm[kx]) * (tau[jx, jx, ix, kx] + 2.0 * tau[jx, ix, jx, kx])
                    t2 = (rot_cm[ix] - rot_cm[jx]) * (tau[kx, kx, kx, ix] - tau[ix, ix, ix, kx])
                    y5 += tau[ix, ix, ix, kx] * (t1 + t2) / den1
                den2 = 8.0 * (rot_cm[jx] - rot_cm[kx]) ** 2
                if den2 > tol:
                    t1 = (rot_cm[jx] - rot_cm[kx]) * (tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx])
                    t2 = 2.0 * (rot_cm[ix] - rot_cm[jx]) * (tau[jx, jx, jx, kx] - tau[kx, kx, kx, jx])
                    t3 = tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx]
                    y6 += t3 * (t1 + t2) / den2
            out[name] = {
                "Y1": y1 * CMINV_TO_HZ,
                "Y2": y2 * CMINV_TO_HZ,
                "Y3": y3 * CMINV_TO_HZ,
                "Y4": y4 * CMINV_TO_HZ,
                "Y5": y5 * CMINV_TO_HZ,
                "Y6": y6 * CMINV_TO_HZ,
                "phi": (y1 - y2 + y3 + y4 + y5 + y6) * CMINV_TO_HZ,
            }

    y1 = 0.0
    for ix in range(3):
        t1 = 2.0 * (tau[ix, ia, ib, ic] + tau[ix, ib, ic, ia] + tau[ix, ic, ia, ib]) ** 2
        t2 = (tau[ix, ia, ib, ib] + 2.0 * tau[ix, ib, ia, ib]) * (tau[ix, ia, ic, ic] + 2.0 * tau[ix, ic, ia, ic])
        t3 = (tau[ix, ib, ic, ic] + 2.0 * tau[ix, ic, ib, ic]) * (tau[ix, ib, ia, ia] + 2.0 * tau[ix, ia, ib, ia])
        t4 = (tau[ix, ic, ia, ia] + 2.0 * tau[ix, ia, ic, ia]) * (tau[ix, ic, ib, ib] + 2.0 * tau[ix, ib, ic, ib])
        y1 += (t1 + t2 + t3 + t4) / rot_cm[ix]
    y1 *= 3.0 / 16.0

    y2 = 0.0
    y3 = 0.0
    for i in range(freq_cm.size):
        t1 = (
            2.0 * c2[i, ia, ib, ic] ** 2
            + c2[i, ib, ib, ia] * c2[i, ic, ic, ia]
            + c2[i, ic, ic, ib] * c2[i, ia, ia, ib]
            + c2[i, ia, ia, ic] * c2[i, ib, ib, ic]
        )
        y2 += freq_cm[i] * t1
        for j in range(freq_cm.size):
            for k in range(freq_cm.size):
                t1 = (
                    c1[i, ia, ia] * c1[j, ib, ib] * c1[k, ic, ic]
                    + 2.0 * c1[i, ia, ia] * c1[j, ib, ic] * c1[k, ib, ic]
                    + 2.0 * c1[i, ib, ib] * c1[j, ic, ia] * c1[k, ic, ia]
                    + 2.0 * c1[i, ic, ic] * c1[j, ia, ib] * c1[k, ia, ib]
                    + 8.0 * c1[i, ib, ic] * c1[j, ic, ia] * c1[k, ia, ib]
                )
                y3 += _f3_reduced(phi3_reduced_cm, freq_cm, i, j, k) * t1
    y2 *= 9.0 / 2.0
    y3 /= 2.0

    y4 = 0.0
    for ix in range(3):
        for jx in range(3):
            if jx == ix:
                continue
            for kx in range(3):
                if kx == ix or kx == jx:
                    continue
                den = 4.0 * (rot_cm[jx] - rot_cm[kx]) ** 2
                if den > tol:
                    t1 = (rot_cm[kx] - rot_cm[ix]) * tau[jx, jx, jx, kx]
                    t2 = (rot_cm[ix] - rot_cm[jx]) * tau[kx, kx, kx, jx]
                    y4 += (tau[jx, jx, jx, kx] - tau[kx, kx, kx, jx]) * (t1 + t2) / den
    y4 *= 3.0 / 2.0
    out[component_key((ia, ib, ic))] = {
        "Y1": y1 * CMINV_TO_HZ,
        "Y2": y2 * CMINV_TO_HZ,
        "Y3": y3 * CMINV_TO_HZ,
        "Y4": y4 * CMINV_TO_HZ,
        "phi": (y1 - y2 + y3 + y4) * CMINV_TO_HZ,
    }
    return out


def sextic_tau_geometry_only_hz(
    tau: np.ndarray,
    rot_cm: np.ndarray,
    axes: dict[str, int],
) -> dict[str, float]:
    """Return the sextic geometry terms that depend only on the quartic tensor ``tau``.

    This isolates the part of the standard sextic geometry sector where a
    non-standard quartic correction can enter by the substitution

        tau = tau_std + delta_tau.

    The remaining geometry term ``-Y2``/``-X2`` depends on ``c2`` and therefore
    stays in the standard first-level sector.
    """
    tol = 1.0e-12

    def component_key(indices: tuple[int, int, int]) -> str:
        counts = {axes["a"]: 0, axes["b"]: 0, axes["c"]: 0}
        for idx in indices:
            counts[idx] += 1
        return "".join(
            label * counts[axes[label]]
            for label in ("a", "b", "c")
            if counts[axes[label]] > 0
        )

    ia = axes["a"]
    ib = axes["b"]
    ic = axes["c"]
    out: dict[str, float] = {}

    for ix in range(3):
        x1 = sum(tau[ix, ix, ix, jx] ** 2 / rot_cm[jx] for jx in range(3)) * 3.0 / 16.0
        x4 = 0.0
        for jx in range(3):
            if jx == ix:
                continue
            den = rot_cm[ix] - rot_cm[jx]
            if abs(den) > tol:
                x4 += tau[jx, ix, ix, ix] ** 2 / den
        x4 /= 4.0
        out[component_key((ix, ix, ix))] = (x1 + x4) * CMINV_TO_HZ

    for ix in range(3):
        for jx in range(3):
            if ix == jx:
                continue
            name = component_key((ix, ix, jx))
            y1 = 0.0
            for kx in range(3):
                t1 = tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx]
                t2 = tau[jx, jx, ix, kx] + 2.0 * tau[ix, jx, jx, kx]
                y1 += (t1**2 + 2.0 * tau[ix, ix, ix, kx] * t2) / rot_cm[kx]
            y1 *= 3.0 / 32.0

            y4 = 0.0
            den = 8.0 * (rot_cm[ix] - rot_cm[jx])
            if abs(den) > tol:
                y4 = tau[ix, ix, ix, jx] * (4.0 * tau[jx, jx, jx, ix] - 3.0 * tau[ix, ix, ix, jx]) / den

            y5 = 0.0
            y6 = 0.0
            for kx in range(3):
                if kx == ix or kx == jx:
                    continue
                den1 = 4.0 * (rot_cm[ix] - rot_cm[kx]) ** 2
                if den1 > tol:
                    t1 = (rot_cm[ix] - rot_cm[kx]) * (tau[jx, jx, ix, kx] + 2.0 * tau[jx, ix, jx, kx])
                    t2 = (rot_cm[ix] - rot_cm[jx]) * (tau[kx, kx, kx, ix] - tau[ix, ix, ix, kx])
                    y5 += tau[ix, ix, ix, kx] * (t1 + t2) / den1
                den2 = 8.0 * (rot_cm[jx] - rot_cm[kx]) ** 2
                if den2 > tol:
                    t1 = (rot_cm[jx] - rot_cm[kx]) * (tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx])
                    t2 = 2.0 * (rot_cm[ix] - rot_cm[jx]) * (tau[jx, jx, jx, kx] - tau[kx, kx, kx, jx])
                    t3 = tau[ix, ix, jx, kx] + 2.0 * tau[ix, jx, ix, kx]
                    y6 += t3 * (t1 + t2) / den2
            out[name] = (y1 + y4 + y5 + y6) * CMINV_TO_HZ

    y1 = 0.0
    for ix in range(3):
        t1 = 2.0 * (tau[ix, ia, ib, ic] + tau[ix, ib, ic, ia] + tau[ix, ic, ia, ib]) ** 2
        t2 = (tau[ix, ia, ib, ib] + 2.0 * tau[ix, ib, ia, ib]) * (tau[ix, ia, ic, ic] + 2.0 * tau[ix, ic, ia, ic])
        t3 = (tau[ix, ib, ic, ic] + 2.0 * tau[ix, ic, ib, ic]) * (tau[ix, ib, ia, ia] + 2.0 * tau[ix, ia, ib, ia])
        t4 = (tau[ix, ic, ia, ia] + 2.0 * tau[ix, ia, ic, ia]) * (tau[ix, ic, ib, ib] + 2.0 * tau[ix, ib, ic, ib])
        y1 += (t1 + t2 + t3 + t4) / rot_cm[ix]
    y1 *= 3.0 / 16.0

    y4 = 0.0
    for ix in range(3):
        for jx in range(3):
            if jx == ix:
                continue
            for kx in range(3):
                if kx == ix or kx == jx:
                    continue
                den = 4.0 * (rot_cm[jx] - rot_cm[kx]) ** 2
                if den > tol:
                    t1 = (rot_cm[kx] - rot_cm[ix]) * tau[jx, jx, jx, kx]
                    t2 = (rot_cm[ix] - rot_cm[jx]) * tau[kx, kx, kx, jx]
                    y4 += (tau[jx, jx, jx, kx] - tau[kx, kx, kx, jx]) * (t1 + t2) / den
    y4 *= 3.0 / 2.0
    out[component_key((ia, ib, ic))] = (y1 + y4) * CMINV_TO_HZ
    return out


def sextic_tau_linear_response_hz(
    tau_base: np.ndarray,
    delta_tau: np.ndarray,
    rot_cm: np.ndarray,
    axes: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Return the linearized sextic geometry response to ``delta_tau``.

    The standard sextic geometry sector depends quadratically on the quartic
    tensor ``tau``. Writing

        G_tau(tau) = sextic_tau_geometry_only_hz(tau, ...),

    the first post-standard correction induced by ``delta_tau`` is the cross
    term

        G_lin(tau_base; delta_tau)
          = G_tau(tau_base + delta_tau) - G_tau(tau_base) - G_tau(delta_tau).

    This is the natural sextic analogue of the first quartic breaking channel:
    it is the first term linear in the non-standard quartic insertion, while
    ``G_tau(delta_tau)`` is quadratic in the new quartic piece.
    """
    base = sextic_tau_geometry_only_hz(tau_base, rot_cm, axes)
    quadratic = sextic_tau_geometry_only_hz(delta_tau, rot_cm, axes)
    updated = sextic_tau_geometry_only_hz(tau_base + delta_tau, rot_cm, axes)
    out: dict[str, dict[str, float]] = {}
    for key in base:
        linear = updated[key] - base[key] - quadratic[key]
        out[key] = {
            "base_hz": base[key],
            "linear_hz": linear,
            "quadratic_hz": quadratic[key],
            "updated_hz": updated[key],
            "delta_total_hz": updated[key] - base[key],
        }
    return out


def sextic_h22_linear_candidate_hz(
    model,
    axes: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Return the first sextic candidate induced linearly by the quartic H22 insertion."""
    tau_std, _ = _quartic_tau_from_model(model)
    tau_h22, _ = _quartic_h22_tau_from_model(model)
    _pmom, rot_cm = _representation_axis_values(model)
    return sextic_tau_linear_response_hz(tau_std, tau_h22, rot_cm, axes)


def sextic_structural_breakdown_hz(
    model,
    phi3_reduced_cm: np.ndarray,
    axes: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Group selected sextic terms into geometry and explicit cubic contributions."""
    raw = sextic_breakdown_hz(model, phi3_reduced_cm, axes)
    out: dict[str, dict[str, float]] = {}

    for key, vals in raw.items():
        if "X1" in vals:
            out[key] = {
                "geometry_hz": vals["X1"] - vals["X2"] + vals["X4"],
                "anharmonic_cubic_hz": vals["X3"],
                "phi_hz": vals["phi"],
            }
        elif "Y5" in vals:
            out[key] = {
                "geometry_hz": vals["Y1"] - vals["Y2"] + vals["Y4"] + vals["Y5"] + vals["Y6"],
                "anharmonic_cubic_hz": vals["Y3"],
                "phi_hz": vals["phi"],
            }
        else:
            out[key] = {
                "geometry_hz": vals["Y1"] - vals["Y2"] + vals["Y4"],
                "anharmonic_cubic_hz": vals["Y3"],
                "phi_hz": vals["phi"],
            }
    return out


def sextic_cubic_hierarchy_hz(
    model,
    phi3_reduced_cm: np.ndarray | None,
    axes: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Return sextic geometry/cubic hierarchy in Hz.

    If ``phi3_reduced_cm`` is ``None``, only the harmonic geometry sector is
    returned. Otherwise the cubic part is split into semi-diagonal and genuine
    three-index contributions.
    """
    freq_n = np.abs(model.vib_freq_cm).size
    zero = np.zeros((freq_n, freq_n, freq_n), dtype=float)
    geom = sextic_structural_breakdown_hz(model, zero, axes)

    if phi3_reduced_cm is None:
        out: dict[str, dict[str, float]] = {}
        for key, vals in geom.items():
            out[key] = {
                "geometry_hz": vals["geometry_hz"],
                "cubic_sd_hz": 0.0,
                "cubic_3ind_hz": 0.0,
                "total_sd_hz": vals["geometry_hz"],
                "total_full_hz": vals["geometry_hz"],
            }
        return out

    phi_sd, phi_3ind = split_cubic_force_constants(phi3_reduced_cm)
    sd = sextic_structural_breakdown_hz(model, phi_sd, axes)
    tri = sextic_structural_breakdown_hz(model, phi_3ind, axes)
    full = sextic_structural_breakdown_hz(model, np.asarray(phi3_reduced_cm, dtype=float), axes)

    out: dict[str, dict[str, float]] = {}
    for key in geom:
        geom_hz = geom[key]["geometry_hz"]
        sd_hz = sd[key]["anharmonic_cubic_hz"]
        tri_hz = tri[key]["anharmonic_cubic_hz"]
        total_full = full[key]["phi_hz"]
        total_sd = geom_hz + sd_hz
        cubic_full = sd_hz + tri_hz
        pct_3ind_of_cubic = 0.0 if abs(cubic_full) <= 1.0e-30 else 100.0 * tri_hz / cubic_full
        pct_3ind_of_total = 0.0 if abs(total_full) <= 1.0e-30 else 100.0 * tri_hz / total_full
        out[key] = {
            "geometry_hz": geom_hz,
            "cubic_sd_hz": sd_hz,
            "cubic_3ind_hz": tri_hz,
            "total_sd_hz": total_sd,
            "total_full_hz": total_full,
            "pct_3ind_of_cubic": pct_3ind_of_cubic,
            "pct_3ind_of_total": pct_3ind_of_total,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare sextic centrifugal constants with Gaussian benchmarks.")
    ap.add_argument("log")
    ap.add_argument("--fchk", required=True)
    ap.add_argument("--breakdown", action="store_true", help="Print selected X/Y term breakdowns in Hz.")
    args = ap.parse_args()

    model, rep, order, signs, didq_err = _aligned_model_from_gaussian(args.fchk, args.log)
    anh = parse_gaussian_anharmonic_force_data(args.log)
    quart = parse_gaussian_quartic_benchmark(args.log)
    sext = parse_gaussian_sextic_benchmark(args.log)
    # `_aligned_model_from_gaussian` already reorders the harmonic model into
    # Gaussian's anharmonic/vib-rot mode order. The cubic force constants
    # parsed from the Gaussian log are already in the phase convention used by
    # Gaussian's sextic block, so no additional mode-sign transformation is
    # applied here.
    _, phi3_reduced_cm, _ = align_gaussian_cubic_force_constants(anh, np.abs(model.vib_freq_cm))
    phi_model = sextic_phi_cartesian_cm(model, phi3_reduced_cm, quart.spectroscopic_axes)
    tau_model, _ = _quartic_tau_from_model(model)
    pmom, rot_cm = _representation_axis_values(model)
    didq = _didq_from_model(model)
    c1_model = _c1_matrix(didq, np.abs(model.vib_freq_cm), pmom)
    c2_model = _c2_tensor(c1_model, _zeta_xyz(model), np.abs(model.vib_freq_cm), rot_cm)

    tau_mask = np.zeros_like(tau_model, dtype=bool)
    for ix in range(3):
        for jx in range(ix + 1):
            tau_mask[:, :, jx, ix] = True
    c1_mask = np.zeros_like(c1_model, dtype=bool)
    for i in range(c1_model.shape[0]):
        for ix in range(3):
            for jx in range(ix + 1):
                c1_mask[i, ix, jx] = True
    c2_mask = np.zeros_like(c2_model, dtype=bool)
    for i in range(c2_model.shape[0]):
        for ix in range(3):
            for jx in range(ix + 1):
                for kx in range(jx + 1):
                    c2_mask[i, ix, jx, kx] = True

    print(f"alignment: rep={rep}, order={order.tolist()}, signs={signs.tolist()}, didq_err={didq_err:.6e}")
    if sext.tau_cm is not None:
        print(f"max|tau_model-tau_log| = {np.max(np.abs(tau_model - sext.tau_cm)[tau_mask]):.6e}")
    if sext.c1 is not None:
        print(f"max|c1_model-c1_log| = {np.max(np.abs(c1_model - sext.c1)[c1_mask]):.6e}")
    if sext.c2 is not None:
        print(f"max|c2_model-c2_log| = {np.max(np.abs(c2_model - sext.c2)[c2_mask]):.6e}")
    if sext.c2_reordered_to_print is not None:
        print(
            "max|c2_model-c2_log_fixed_order| = "
            f"{np.max(np.abs(c2_model - sext.c2_reordered_to_print)[c2_mask]):.6e}"
        )
    print("Sextic Cartesian Phi comparison in Hz")
    print(f"{'component':<8} {'model':>18} {'gaussian':>18}")
    for key in ("aaa", "aab", "aac", "abb", "abc", "acc", "bbb", "bbc", "bcc", "ccc"):
        print(f"{key:<8} {phi_model[key] * CMINV_TO_HZ:18.8f} {sext.phi_cart_hz[key]:18.8f}")
    if args.breakdown:
        print("Selected sextic-term breakdowns in Hz")
        for key, terms in sextic_breakdown_hz(model, phi3_reduced_cm, quart.spectroscopic_axes).items():
            print(key, terms)
        print("Selected sextic structural breakdowns in Hz")
        for key, terms in sextic_structural_breakdown_hz(model, phi3_reduced_cm, quart.spectroscopic_axes).items():
            print(key, terms)


if __name__ == "__main__":
    main()
