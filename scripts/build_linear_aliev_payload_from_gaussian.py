#!/usr/bin/env python3
"""Build a reduced linear-Aliev payload from Gaussian data for a linear molecule.

Current scope:
- harmonic model from FCHK via the existing CeDiTT builder
- anharmonic cubic/quartic data from Gaussian log
- reduced quartic branch only (`k4_reduced`)

Future scope:
- full four-index quartic input can be added later without changing the target
  payload format consumed by `linear_aliev_cli.py`.

Current normalization policy:
- Gaussian-derived quantities are converted once into an Aliev-side input layer
- downstream code consumes only Aliev-normalized inputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceditt_gui import _build_harmonic_model_from_inputs
from compare_gaussian_sextic import CMINV_TO_MHZ, _classify_rotor_limit, _degenerate_mode_metadata
from compare_gaussian_sextic import DIDQ_AU_PER_AMU_SQRT_ANG, FACTG
from gaussian_force_constant_units import (
    gaussian_bxx_to_aliev,
    raw_cubic_to_reduced_cm,
    raw_quartic_to_reduced_cm,
    reduced_cubic_to_aliev_k3,
    reduced_quartic_to_aliev_k4,
)
from gaussian_vpt_parser import (
    parse_gaussian_alpha_data,
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_linear_ltype_constants,
    parse_gaussian_linear_rotdist_constants,
)
from symmetry_metadata import assign_normal_mode_irreps


def _abc_axis_columns(alpha_axis_labels: tuple[str, ...]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, label in enumerate(alpha_axis_labels):
        head = label.strip()[0].lower()
        if head in {"a", "b", "c"}:
            out[head] = idx
    if {"a", "b", "c"} <= set(out):
        return out
    raise ValueError("Could not infer A/B/C alpha columns from Gaussian alpha labels.")


def _perpendicular_rotational_constant_cm(model, rotor_limit: dict[str, object]) -> float:
    abc = np.asarray(model.abc_mhz, dtype=float)
    deg_axes = tuple(str(x) for x in rotor_limit.get("degenerate_axes", ()))
    abc_to_idx = {"a": 0, "b": 1, "c": 2}
    vals = [float(abc[abc_to_idx[ax]] / CMINV_TO_MHZ) for ax in deg_axes]
    if not vals:
        raise ValueError("Linear-rotor limit does not expose degenerate perpendicular axes.")
    return float(sum(vals) / len(vals))


def _parallel_mode_indices(n_modes: int, pair_meta: list[dict[str, object]]) -> list[int]:
    deg = set()
    for meta in pair_meta:
        deg.update(int(x) for x in meta["pair"])
    return [idx for idx in range(n_modes) if idx not in deg]


def _representative_perpendicular_pairs(pair_meta: list[dict[str, object]]) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in meta["pair"]) for meta in pair_meta]


def _build_zeta_nt(model, pair_meta: list[dict[str, object]], parallel_indices: list[int]) -> list[list[float]]:
    return _build_zeta_nt_with_reduction(model, pair_meta, parallel_indices, reduction="principal_direction")


def _build_zeta_pair_blocks(
    model,
    pair_meta: list[dict[str, object]],
    parallel_indices: list[int],
) -> list[list[list[list[float]]]]:
    """Return the full 2x2 Coriolis block for each (parallel mode, degenerate pair).

    This is kept for quadratic bending invariants, where reducing the block to a
    scalar loses rotational information inside the degenerate perpendicular
    subspace.
    """

    zeta = np.asarray(model.coriolis_zeta_pairs_xyz, dtype=float)
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    deg_axes = tuple(str(x) for x in rotor_limit.get("degenerate_axes", ()))
    axis_map = {"a": 0, "b": 1, "c": 2}
    deg_axis_indices = [axis_map[ax] for ax in deg_axes]
    if len(deg_axis_indices) < 2:
        raise ValueError("Degenerate-pair Coriolis block construction requires two perpendicular axes.")
    ax0, ax1 = deg_axis_indices[:2]
    out: list[list[list[list[float]]]] = []
    for p in parallel_indices:
        row: list[list[list[float]]] = []
        for meta in pair_meta:
            i, j = (int(x) for x in meta["pair"])
            row.append(
                [
                    [float(zeta[ax0, p, i]), float(zeta[ax0, p, j])],
                    [float(zeta[ax1, p, i]), float(zeta[ax1, p, j])],
                ]
            )
        out.append(row)
    return out


def _build_pair_seed_perpendicular_from_gaussian_qe(
    log_path: str,
    *,
    n_pairs: int,
) -> list[float]:
    """Build a provisional bending seed directly from Gaussian q^e values.

    The linear-Aliev implementation uses

        D_v = D_J - sum beta_k * f_k(v)

    so a positive experimental increment Delta D_t > 0 must correspond to a
    negative beta_t.  Until the true rotational-derivative zeta builder exists,
    the Gaussian q^e bridge is therefore injected with the sign needed to act
    as a placeholder for -beta_t rather than beta_t itself.
    """

    qconst = parse_gaussian_linear_ltype_constants(log_path)
    qe_items = sorted(qconst.q_e_mhz.items())
    if len(qe_items) < n_pairs:
        raise ValueError(f"Gaussian q^e data expose only {len(qe_items)} pairs, expected {n_pairs}.")
    return [float(-val / CMINV_TO_MHZ) for _, val in qe_items[:n_pairs]]


def _build_zeta_nt_component(
    zeta: np.ndarray,
    pair_meta: list[dict[str, object]],
    parallel_indices: list[int],
    deg_axis_indices: list[int],
    *,
    axis_slot: int,
    pair_slot: int,
) -> list[list[float]]:
    if len(deg_axis_indices) < 2:
        raise ValueError("Component-wise zeta construction requires two degenerate rotational axes.")
    if axis_slot not in (0, 1) or pair_slot not in (0, 1):
        raise ValueError("axis_slot and pair_slot must be 0 or 1.")
    axis_idx = deg_axis_indices[axis_slot]
    out: list[list[float]] = []
    for p in parallel_indices:
        row: list[float] = []
        for meta in pair_meta:
            pair = tuple(int(x) for x in meta["pair"])
            row.append(float(zeta[axis_idx, p, pair[pair_slot]]))
        out.append(row)
    return out


def _principal_coriolis_projection(block_2x2: np.ndarray) -> float:
    """Return the signed principal projection of a 2x2 degenerate Coriolis block."""

    block = np.asarray(block_2x2, dtype=float)
    if block.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 block, got shape {block.shape}")
    sym = 0.5 * (block + block.T)
    evals = np.linalg.eigvalsh(sym)
    return float(evals[np.argmax(np.abs(evals))])


def _build_zeta_nt_with_reduction(
    model,
    pair_meta: list[dict[str, object]],
    parallel_indices: list[int],
    *,
    reduction: str,
) -> list[list[float]]:
    zeta = np.asarray(model.coriolis_zeta_pairs_xyz, dtype=float)
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    deg_axes = tuple(str(x) for x in rotor_limit.get("degenerate_axes", ()))
    axis_map = {"a": 0, "b": 1, "c": 2}
    deg_axis_indices = [axis_map[ax] for ax in deg_axes]
    red = str(reduction).strip().lower()
    component_map = {
        "component_ta": (0, 0),
        "component_tb": (0, 1),
        "component_ua": (1, 0),
        "component_ub": (1, 1),
    }
    if red not in {"norm", "maxabs", "pair_offdiag", "pair_diag", "principal_direction", *component_map}:
        raise ValueError(
            "zeta_reduction must be 'norm', 'maxabs', 'pair_offdiag', 'pair_diag', 'principal_direction', "
            "'component_ta', 'component_tb', 'component_ua', or 'component_ub'."
        )
    if red == "principal_direction":
        if len(deg_axis_indices) < 2:
            raise ValueError("Principal-direction zeta construction requires two degenerate rotational axes.")
        out: list[list[float]] = []
        ax0, ax1 = deg_axis_indices[:2]
        for p in parallel_indices:
            row: list[float] = []
            for meta in pair_meta:
                i, j = (int(x) for x in meta["pair"])
                block = np.array(
                    [
                        [float(zeta[ax0, p, i]), float(zeta[ax0, p, j])],
                        [float(zeta[ax1, p, i]), float(zeta[ax1, p, j])],
                    ],
                    dtype=float,
                )
                row.append(_principal_coriolis_projection(block))
            out.append(row)
        return out
    if red in component_map:
        axis_slot, pair_slot = component_map[red]
        return _build_zeta_nt_component(
            zeta,
            pair_meta,
            parallel_indices,
            deg_axis_indices,
            axis_slot=axis_slot,
            pair_slot=pair_slot,
        )
    if red in {"pair_offdiag", "pair_diag"}:
        if len(deg_axis_indices) < 2:
            raise ValueError("Pair-wise zeta reduction requires two degenerate rotational axes.")
        out: list[list[float]] = []
        ax0, ax1 = deg_axis_indices[:2]
        for p in parallel_indices:
            row: list[float] = []
            for meta in pair_meta:
                i, j = (int(x) for x in meta["pair"])
                if red == "pair_offdiag":
                    val = 0.5 * (float(zeta[ax0, p, j]) + float(zeta[ax1, p, i]))
                else:
                    val = 0.5 * (float(zeta[ax0, p, i]) + float(zeta[ax1, p, j]))
                row.append(val)
            out.append(row)
        return out
    out: list[list[float]] = []
    for p in parallel_indices:
        row: list[float] = []
        for meta in pair_meta:
            i, j = (int(x) for x in meta["pair"])
            vals: list[float] = []
            for ax in deg_axis_indices:
                vals.extend([float(zeta[ax, p, i]), float(zeta[ax, p, j])])
            if red == "norm":
                row.append(float(np.sqrt(sum(v * v for v in vals))))
            else:
                row.append(float(max(vals, key=lambda x: abs(x))))
        out.append(row)
    return out


def _build_bxx_parallel_from_alpha(
    alpha_cm: np.ndarray,
    alpha_axis_labels: tuple[str, ...],
    parallel_indices: list[int],
    rotor_limit: dict[str, object],
    *,
    cn_source: str,
) -> list[float]:
    axis_cols = _abc_axis_columns(alpha_axis_labels)
    deg_axes = tuple(str(x) for x in rotor_limit.get("degenerate_axes", ()))
    perp_cols = [axis_cols[ax] for ax in deg_axes]
    if not perp_cols:
        raise ValueError("Could not infer perpendicular rotational axes for alpha-based C_n bootstrap.")
    out: list[float] = []
    for p in parallel_indices:
        alpha_perp = float(np.mean(alpha_cm[p, perp_cols]))
        if cn_source == "alpha_perp_with_Bxx_equals_minus_alpha_perp":
            out.append(float(-alpha_perp))
        elif cn_source == "alpha_perp_with_Bxx_equals_minus_half_alpha_perp":
            out.append(float(-0.5 * alpha_perp))
        else:
            raise ValueError(f"Unsupported C_n bootstrap source {cn_source!r}.")
    return out


def _build_bxx_parallel_from_didq(
    model,
    parallel_indices: list[int],
    rotor_limit: dict[str, object],
) -> list[float]:
    deg_axes = tuple(str(x) for x in rotor_limit.get("degenerate_axes", ()))
    if not deg_axes:
        raise ValueError("Linear-rotor limit does not expose degenerate axes for dI/dQ bootstrap.")
    try:
        perp_axis = tuple(model.xyz_to_abc).index(deg_axes[0])
    except ValueError as exc:
        raise ValueError("Could not map the perpendicular linear axis in xyz_to_abc.") from exc
    moments_xyz = np.asarray(model.moments_amu_a2, dtype=float).reshape(3)
    pmom = float(moments_xyz[perp_axis])
    if abs(pmom) <= 1.0e-30:
        raise ValueError("Perpendicular moment of inertia is too small for dI/dQ bootstrap.")
    didq_au_sqrt_ang = np.asarray(model.dI_au, dtype=float) / DIDQ_AU_PER_AMU_SQRT_ANG
    freq = np.asarray(model.vib_freq_cm, dtype=float).reshape(-1)
    out: list[float] = []
    for idx in parallel_indices:
        wi = float(abs(freq[idx]))
        if wi <= 1.0e-30:
            out.append(0.0)
            continue
        x = np.sqrt((FACTG * wi) ** 3)
        den = 2.0 * x * pmom * pmom
        num = float(didq_au_sqrt_ang[perp_axis, perp_axis, idx])
        out.append(0.0 if abs(den) <= 1.0e-30 else num / den)
    return out


def _normalize_bxx_parallel_to_aliev(
    bxx_gaussian: list[float],
    model,
    parallel_indices: list[int],
) -> list[float]:
    freq_parallel = np.asarray([abs(float(model.vib_freq_cm[i])) for i in parallel_indices], dtype=float)
    return list(gaussian_bxx_to_aliev(np.asarray(bxx_gaussian, dtype=float), freq_parallel))


def _normalize_force_constants_to_aliev(
    phi3_gaussian_cm: np.ndarray,
    phi4_gaussian_cm: np.ndarray,
    frequencies_cm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        reduced_cubic_to_aliev_k3(phi3_gaussian_cm, frequencies_cm),
        reduced_quartic_to_aliev_k4(phi4_gaussian_cm, frequencies_cm),
    )


def _build_k3_parallel(phi3_cm: np.ndarray, parallel_indices: list[int]) -> list[list[list[float]]]:
    return [
        [
            [float(phi3_cm[i, j, k]) for k in parallel_indices]
            for j in parallel_indices
        ]
        for i in parallel_indices
    ]


def _build_k3_perp_pair(phi3_cm: np.ndarray, pair_meta: list[dict[str, object]], parallel_indices: list[int]) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    for meta_t in pair_meta:
        i1, i2 = (int(x) for x in meta_t["pair"])
        row: list[list[float]] = []
        for meta_tp in pair_meta:
            j1, j2 = (int(x) for x in meta_tp["pair"])
            vals: list[float] = []
            for n in parallel_indices:
                avg = 0.25 * (
                    float(phi3_cm[i1, j1, n])
                    + float(phi3_cm[i1, j2, n])
                    + float(phi3_cm[i2, j1, n])
                    + float(phi3_cm[i2, j2, n])
                )
                vals.append(avg)
            row.append(vals)
        out.append(row)
    return out


def _build_k4_reduced(phi4_cm: np.ndarray, parallel_indices: list[int]) -> list[list[list[float]]]:
    size = len(parallel_indices)
    out = [[[0.0 for _ in range(size)] for _ in range(size)] for _ in range(size)]
    for ii, i in enumerate(parallel_indices):
        for jj, j in enumerate(parallel_indices):
            for kk, k in enumerate(parallel_indices):
                a, b = sorted((j, k))
                out[ii][jj][kk] = float(phi4_cm[i, i, a, b])
    return out


def build_payload(
    *,
    fchk_path: str,
    log_path: str,
    cn_source: str = "alpha_perp_with_Bxx_equals_minus_alpha_perp",
    force_constant_source: str = "reduced",
    zeta_reduction: str = "principal_direction",
    pair_seed_source: str = "gaussian_qe_source",
) -> dict[str, object]:
    model, _ = _build_harmonic_model_from_inputs("I", fchk_path=fchk_path)
    rotor_limit = _classify_rotor_limit(np.asarray(model.abc_mhz, dtype=float), np.asarray(model.moments_amu_a2, dtype=float))
    if rotor_limit.get("kind") != "linear":
        raise ValueError("This bootstrap currently supports only linear molecules.")

    pair_meta = _degenerate_mode_metadata(model, rotor_limit)
    parallel_indices = _parallel_mode_indices(int(len(model.vib_freq_cm)), pair_meta)
    mode_irreps = assign_normal_mode_irreps(model)
    anh = parse_gaussian_anharmonic_force_data(log_path)
    alpha = parse_gaussian_alpha_data(log_path)
    rotdist = parse_gaussian_linear_rotdist_constants(log_path)
    if rotdist.d_mhz is None:
        raise ValueError("Gaussian log does not expose a linear D constant.")
    pair_seed_src = str(pair_seed_source).strip().lower()
    if pair_seed_src not in {"gaussian_qe_source", "formula"}:
        raise ValueError("pair_seed_source must be 'gaussian_qe_source' or 'formula'.")
    fc_source = str(force_constant_source).strip().lower()
    if fc_source not in {"reduced", "raw_au_reconverted"}:
        raise ValueError("force_constant_source must be 'reduced' or 'raw_au_reconverted'.")
    if fc_source == "reduced":
        phi3_gaussian = anh.phi3_reduced_cm
        phi4_gaussian = anh.phi4_reduced_cm
    else:
        phi3_gaussian = raw_cubic_to_reduced_cm(anh.phi3_raw_au, anh.frequencies_cm)
        phi4_gaussian = raw_quartic_to_reduced_cm(anh.phi4_raw_au, anh.frequencies_cm)
    phi3_for_bootstrap, phi4_for_bootstrap = _normalize_force_constants_to_aliev(
        phi3_gaussian,
        phi4_gaussian,
        np.asarray(anh.frequencies_cm, dtype=float),
    )
    if cn_source in {
        "alpha_perp_with_Bxx_equals_minus_alpha_perp",
        "alpha_perp_with_Bxx_equals_minus_half_alpha_perp",
    }:
        bxx_parallel_gaussian = _build_bxx_parallel_from_alpha(
            alpha.alpha_cm,
            alpha.axis_labels,
            parallel_indices,
            rotor_limit,
            cn_source=cn_source,
        )
    elif cn_source == "didq_linear_v_iscr":
        bxx_parallel_gaussian = _build_bxx_parallel_from_didq(model, parallel_indices, rotor_limit)
    else:
        raise ValueError(f"Unsupported C_n bootstrap source {cn_source!r}.")
    bxx_parallel = _normalize_bxx_parallel_to_aliev(bxx_parallel_gaussian, model, parallel_indices)
    pair_seed_perpendicular = None
    if pair_seed_src == "gaussian_qe_source":
        pair_seed_perpendicular = _build_pair_seed_perpendicular_from_gaussian_qe(
            log_path,
            n_pairs=len(pair_meta),
        )
    payload = {
        "B": _perpendicular_rotational_constant_cm(model, rotor_limit),
        "D_J": float(rotdist.d_mhz / CMINV_TO_MHZ),
        "omega_parallel": [float(abs(model.vib_freq_cm[i])) for i in parallel_indices],
        "omega_perpendicular": [float(meta["freq_cm"]) for meta in pair_meta],
        "zeta_nt": _build_zeta_nt_with_reduction(model, pair_meta, parallel_indices, reduction=zeta_reduction),
        "zeta_pair_blocks": _build_zeta_pair_blocks(model, pair_meta, parallel_indices),
        "bxx_parallel": [float(x) for x in bxx_parallel],
        "pair_seed_perpendicular": None if pair_seed_perpendicular is None else [float(x) for x in pair_seed_perpendicular],
        "k3_parallel": _build_k3_parallel(phi3_for_bootstrap, parallel_indices),
        "k3_perp_pair": _build_k3_perp_pair(phi3_for_bootstrap, pair_meta, parallel_indices),
        "k4_reduced": _build_k4_reduced(phi4_for_bootstrap, parallel_indices),
        "metadata": {
            "source": "gaussian_to_aliev_linear_bootstrap",
            "quartic_mode": "reduced",
            "coordinate_normalization": "aliev",
            "unit_audit_status": "representation_consistent_but_not_yet_quantitatively_validated",
            "force_constant_source": fc_source,
            "zeta_reduction": zeta_reduction,
            "pair_seed_source": pair_seed_src,
            "pair_seed_status": "non_physical_bridge" if pair_seed_src == "gaussian_qe_source" else "diagnostic_formula_path",
            "pair_seed_physical_role": "pairwise_l_type_J0_driver_proxy" if pair_seed_src == "gaussian_qe_source" else "diagnostic_aliev_formula_seed",
            "bending_comparison_status": "qualitative_proxy_only" if pair_seed_src == "gaussian_qe_source" else "diagnostic_only",
            "pair_seed_invariant": "frobenius_dot_of_full_2x2_degenerate_coriolis_blocks",
            "parallel_mode_indices_0based": parallel_indices,
            "parallel_mode_indices_1based": [int(i + 1) for i in parallel_indices],
            "parallel_mode_irreps": [str(mode_irreps[i]) for i in parallel_indices],
            "perpendicular_pairs_0based": _representative_perpendicular_pairs(pair_meta),
            "perpendicular_pairs_1based": [[int(i + 1), int(j + 1)] for i, j in _representative_perpendicular_pairs(pair_meta)],
            "perpendicular_pair_irreps": [str(mode_irreps[int(meta['pair'][0])]) for meta in pair_meta],
            "cn_source": cn_source,
            "cn_recommended_source": "alpha_perp_with_Bxx_equals_minus_alpha_perp",
            "cn_rationale": "Primary bootstrap treats the Gaussian-side B_n^(xx) estimate as a Gaussian-normal-coordinate quantity and converts it once to Aliev form through B_n^(xx)(Aliev)=B_n^(xx)(Gaussian)/sqrt(omega_n). Alternative sources remain diagnostic only.",
            "zeta_rationale": "Scalar zeta_nt remains a diagnostic bridge for the non-quadratic Aliev terms. It is not treated as the physical object required for the bending seed.",
            "known_unit_risks": [
                "Gaussian Coriolis tensors are not assumed equivalent to the physical zeta objects required by the Aliev bending seed.",
                "B_n^(xx) is now normalized to Aliev coordinates, but the Gaussian-side source used to estimate B_n^(xx)(Gaussian) still needs analytic validation.",
                "Force constants are now normalized once into Aliev coordinates before entering the downstream equations; remaining mismatch would point to a deeper convention issue, not to hidden downstream rescaling.",
                "The default bending seed is a non-physical bridge from Gaussian q^e until a rotational-derivative zeta builder is implemented.",
                "Under the Gaussian q^e bridge, the v4/v5 comparison is qualitative only: q^e drives the pairwise l-type J0 layer and is not a quantitative Delta D_t target.",
            ],
        },
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fchk", required=True, help="Gaussian FCHK for the harmonic model.")
    ap.add_argument("--log", required=True, help="Gaussian anharmonic log.")
    ap.add_argument("--output", help="Output JSON payload path.")
    ap.add_argument(
        "--cn-source",
        choices=(
            "alpha_perp_with_Bxx_equals_minus_alpha_perp",
            "alpha_perp_with_Bxx_equals_minus_half_alpha_perp",
            "didq_linear_v_iscr",
        ),
        default="alpha_perp_with_Bxx_equals_minus_alpha_perp",
        help="Gaussian-side source used to estimate B_n^(xx)(Gaussian) before a single Gaussian->Aliev normalization step.",
    )
    ap.add_argument(
        "--force-constant-source",
        choices=("reduced", "raw_au_reconverted"),
        default="reduced",
        help="Use Gaussian reduced constants directly or reconstruct them from raw atomic-unit force constants.",
    )
    ap.add_argument(
        "--zeta-reduction",
        choices=("principal_direction", "norm", "maxabs", "pair_offdiag", "pair_diag", "component_ta", "component_tb", "component_ua", "component_ub"),
        default="principal_direction",
        help="How the degenerate-pair Coriolis block is converted into the scalar zeta_nt supplied to the Aliev equations.",
    )
    ap.add_argument(
        "--pair-seed-source",
        choices=("gaussian_qe_source", "formula"),
        default="gaussian_qe_source",
        help="Default is a non-physical Gaussian q^e bridge; use 'formula' only for diagnostic comparison.",
    )
    args = ap.parse_args()

    payload = build_payload(
        fchk_path=args.fchk,
        log_path=args.log,
        cn_source=args.cn_source,
        force_constant_source=args.force_constant_source,
        zeta_reduction=args.zeta_reduction,
        pair_seed_source=args.pair_seed_source,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
        print(args.output)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
