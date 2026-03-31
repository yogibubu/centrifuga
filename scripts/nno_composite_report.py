#!/usr/bin/env python3
"""Report a dual-level NNO composite example.

Current scope:
- harmonic/order-2 quartic branch from each harmonic model;
- order-3 alpha and sextic diagnostics for pure low, pure high, and
  high-harmonic + low-anharmonic composite under the standard same-normal-modes
  approximation;
- explicit mode-overlap diagnostics used to validate that approximation;
- order-4 linear Aliev branch for pure low, pure high, and composite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LINEAR_CASES = REPO_ROOT / "data/gaussian/linear_cases"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ceditt_gui import _mode_overlap_mapping_from_fchks, _raw_cubic_to_reduced_with_target_freqs
from channel_contributions import parse_model
from compare_gaussian_sextic import CMINV_TO_HZ, sextic_cubic_hierarchy_hz
from distortion_workflow import (
    _linear_gaussian_exact_source_blocks,
    _linear_gaussian_exact_source_h_hz,
    classify_rotor_limit,
    compute_order2_quartic,
)
from gaussian_vpt_parser import (
    align_gaussian_cubic_force_constants,
    apply_mode_signs_to_cubic_force_constants,
    frequency_reorder_map,
    parse_gaussian_anharmonic_force_data,
    reorder_cubic_force_constants,
)
from linear_dv_aliev_terms import (
    build_explicit_aliev_L_model,
    build_explicit_aliev_betas,
    build_explicit_aliev_dv_compactness_audit,
    build_explicit_aliev_dv_general_model,
    make_linear_aliev_explicit_inputs_from_mapping,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload, build_payload_same_modes_approx
from vibrot_alpha import alpha_matrix_from_harmonic_and_cubic_cm


def _dominant_component(levels: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    key = max(levels, key=lambda item: abs(levels[item]["total_full_hz"]))
    return key, levels[key]


def _alpha_summary(alpha: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        "alpha_linear_cm": [float(x) for x in np.asarray(alpha["alpha_linear_cm"], dtype=float)],
        "alpha_total_sum_abc_cm": [float(x) for x in np.asarray(alpha["alpha_total_cm_abc"], dtype=float).sum(axis=0)],
        "alpha_coriolis_sum_abc_cm": [float(x) for x in np.asarray(alpha["alpha_coriolis_cm_abc"], dtype=float).sum(axis=0)],
        "alpha_inertia_sum_abc_cm": [float(x) for x in np.asarray(alpha["alpha_inertia_cm_abc"], dtype=float).sum(axis=0)],
        "alpha_anharmonic_sum_abc_cm": [float(x) for x in np.asarray(alpha["alpha_anharmonic_cm_abc"], dtype=float).sum(axis=0)],
    }


def _zero_state(mode_kinds: tuple[str, ...]) -> tuple[int, ...]:
    return (0,) * len(mode_kinds)


def _same_modes_composite_h_hz(
    model_hi,
    *,
    high_log: str | Path,
    low_phi3_raw_target: np.ndarray,
    d_hz: float,
) -> dict[str, object] | None:
    """Evaluate the linear sextic H on the high-level harmonic model with low-level cubic field.

    This is the standard same-normal-modes composite approximation:
    high-level harmonic objects are retained, and only the cubic tensor is
    injected from the low level after mode label/sign alignment.
    """

    high_blocks = _linear_gaussian_exact_source_blocks(str(high_log))
    source_to_target = np.asarray(
        frequency_reorder_map(np.asarray(high_blocks["freq_cm"], dtype=float), np.abs(np.asarray(model_hi.vib_freq_cm, dtype=float))),
        dtype=int,
    )
    low_phi3_raw_high_source = np.asarray(low_phi3_raw_target, dtype=float)[
        np.ix_(source_to_target, source_to_target, source_to_target)
    ]
    composite_blocks = dict(high_blocks)
    composite_blocks["phi3_raw_au"] = low_phi3_raw_high_source
    rotor_limit = classify_rotor_limit(np.asarray(model_hi.abc_mhz, dtype=float), np.asarray(model_hi.moments_amu_a2, dtype=float))
    return _linear_gaussian_exact_source_h_hz(
        model_hi,
        gaussian_source_blocks=composite_blocks,
        d_hz=float(d_hz),
        rotor_limit=rotor_limit,
    )


def build_nno_composite_report(
    low_fchk: str | Path,
    low_log: str | Path,
    high_fchk: str | Path,
    high_log: str | Path,
) -> str:
    low_fchk = Path(low_fchk)
    low_log = Path(low_log)
    high_fchk = Path(high_fchk)
    high_log = Path(high_log)

    model_lo, _, meta_lo = parse_model(low_fchk, low_log)
    model_hi, _, meta_hi = parse_model(high_fchk, high_log)

    anh_lo = parse_gaussian_anharmonic_force_data(low_log)
    _, _, phi3raw_lo = align_gaussian_cubic_force_constants(anh_lo, np.abs(model_lo.vib_freq_cm))
    transfer = _mode_overlap_mapping_from_fchks(low_fchk, high_fchk)
    raw_target = reorder_cubic_force_constants(phi3raw_lo, transfer["mapping"])
    raw_target = apply_mode_signs_to_cubic_force_constants(raw_target, transfer["target_signs"])
    phi3_dual = _raw_cubic_to_reduced_with_target_freqs(raw_target, model_hi.vib_freq_cm)
    payload_lo = build_payload(
        fchk_path=str(low_fchk),
        log_path=str(low_log),
        force_constant_source="raw_au_reconverted",
    )
    payload_hi = build_payload(
        fchk_path=str(high_fchk),
        log_path=str(high_log),
        force_constant_source="raw_au_reconverted",
    )
    payload_dual = build_payload_same_modes_approx(
        harmonic_fchk_path=str(high_fchk),
        anharmonic_log_path=str(low_log),
        anharmonic_fchk_path=str(low_fchk),
        force_constant_source="raw_au_reconverted",
        align_permutation_and_sign=True,
    )
    inputs_lo = make_linear_aliev_explicit_inputs_from_mapping(payload_lo, quartic_mode="reduced")
    inputs_hi = make_linear_aliev_explicit_inputs_from_mapping(payload_hi, quartic_mode="reduced")
    inputs_dual = make_linear_aliev_explicit_inputs_from_mapping(payload_dual, quartic_mode="reduced")

    order2_lo = compute_order2_quartic(model_lo, linear_log_path=str(low_log))
    order2_hi = compute_order2_quartic(model_hi, linear_log_path=str(high_log))

    sextic_lo = sextic_cubic_hierarchy_hz(model_lo, np.asarray(meta_lo["phi3_reduced_cm"], dtype=float), {"a": 0, "b": 1, "c": 2})
    sextic_hi = sextic_cubic_hierarchy_hz(model_hi, np.asarray(meta_hi["phi3_reduced_cm"], dtype=float), {"a": 0, "b": 1, "c": 2})
    sextic_dual = sextic_cubic_hierarchy_hz(model_hi, phi3_dual, {"a": 0, "b": 1, "c": 2})

    h_lo = order2_lo["linear_ltype_terms"]["pure_rotational_branch"]["gaussian_source_reconstructed_sextic_hz"]
    h_hi = order2_hi["linear_ltype_terms"]["pure_rotational_branch"]["gaussian_source_reconstructed_sextic_hz"]
    h_dual = _same_modes_composite_h_hz(
        model_hi,
        high_log=high_log,
        low_phi3_raw_target=raw_target,
        d_hz=order2_hi["special_quartic_projection"]["quartic_mhz"]["D"] * 1.0e6,
    )

    alpha_lo = alpha_matrix_from_harmonic_and_cubic_cm(model_lo, np.asarray(meta_lo["phi3_reduced_cm"], dtype=float))
    alpha_hi = alpha_matrix_from_harmonic_and_cubic_cm(model_hi, np.asarray(meta_hi["phi3_reduced_cm"], dtype=float))
    alpha_dual = alpha_matrix_from_harmonic_and_cubic_cm(model_hi, phi3_dual)
    betas_lo = build_explicit_aliev_betas(inputs_lo)
    betas_hi = build_explicit_aliev_betas(inputs_hi)
    betas_dual = build_explicit_aliev_betas(inputs_dual)
    dv_lo = build_explicit_aliev_dv_general_model(inputs_lo)
    dv_hi = build_explicit_aliev_dv_general_model(inputs_hi)
    dv_dual = build_explicit_aliev_dv_general_model(inputs_dual)
    l_lo = build_explicit_aliev_L_model(inputs_lo)
    l_hi = build_explicit_aliev_L_model(inputs_hi)
    l_dual = build_explicit_aliev_L_model(inputs_dual)
    audit_lo = build_explicit_aliev_dv_compactness_audit(inputs_lo)
    audit_hi = build_explicit_aliev_dv_compactness_audit(inputs_hi)
    audit_dual = build_explicit_aliev_dv_compactness_audit(inputs_dual)

    dom_lo_key, dom_lo = _dominant_component(sextic_lo)
    dom_hi_key, dom_hi = _dominant_component(sextic_hi)
    dom_dual_key, dom_dual = _dominant_component(sextic_dual)

    molecule_label = high_fchk.stem.split("_")[0].upper()
    h_dual_line = (
        f"`H = {float(h_dual['H']):+.12e}` Hz"
        if h_dual is not None and h_dual.get("H") is not None
        else "`Dual-level sextic H could not be reconstructed.`"
    )

    lines = [
        f"# {molecule_label} composite force-field example",
        "",
        f"Low level:  `{low_fchk.name}` + `{low_log.name}`",
        f"High level: `{high_fchk.name}` + `{high_log.name}`",
        "",
        "## Harmonic / order-2 branch",
        "",
        f"- `D` from low harmonic model  = {order2_lo['special_quartic_projection']['quartic_mhz']['D']:.12f} MHz",
        f"- `D` from high harmonic model = {order2_hi['special_quartic_projection']['quartic_mhz']['D']:.12f} MHz",
        "- In the present implementation the composite order-2 branch coincides with the high-harmonic model, because it depends only on geometry + Hessian.",
        "",
        "## Cubic transfer diagnostics",
        "",
        f"- `mapping = {tuple(int(x) for x in transfer['mapping'])}`",
        f"- `target_signs = {tuple(int(np.sign(x)) if abs(x) > 0 else 1 for x in transfer['target_signs'])}`",
        f"- `min|overlap| = {float(transfer['min_abs_overlap']):.6f}`",
        f"- `max offdiag |overlap| = {float(transfer['max_offdiag_abs_overlap']):.6f}`",
        "- Composite branch below uses the standard same-normal-modes approximation with only label/sign alignment, not a dense tensor re-expansion.",
        "",
        "## Sextic linear source branch",
        "",
        f"- low  : `H = {float(h_lo['H']):+.12e}` Hz",
        f"- high : `H = {float(h_hi['H']):+.12e}` Hz",
        f"- comp : {h_dual_line}",
        "",
        "## Dominant sextic hierarchy component",
        "",
        f"- low  : `{dom_lo_key}` geometry={float(dom_lo['geometry_hz']):+.12e} Hz, cubic_sd={float(dom_lo['cubic_sd_hz']):+.12e} Hz, total={float(dom_lo['total_full_hz']):+.12e} Hz",
        f"- high : `{dom_hi_key}` geometry={float(dom_hi['geometry_hz']):+.12e} Hz, cubic_sd={float(dom_hi['cubic_sd_hz']):+.12e} Hz, total={float(dom_hi['total_full_hz']):+.12e} Hz",
        f"- comp : `{dom_dual_key}` geometry={float(dom_dual['geometry_hz']):+.12e} Hz, cubic_sd={float(dom_dual['cubic_sd_hz']):+.12e} Hz, total={float(dom_dual['total_full_hz']):+.12e} Hz",
        "",
        "## Linear order-4 branch",
        "",
        f"- low  : `D_v(0) = {float(dv_lo.value_for_state(_zero_state(dv_lo.mode_kinds))):+.12e}` cm^-1, `beta_parallel = {[float(v) for _, v in sorted(betas_lo.beta_parallel.items())]}`, `beta_perpendicular = {[float(v) for _, v in sorted(betas_lo.beta_perpendicular.items())]}`, `L = {float(l_lo.value):+.12e}` cm^-1 (`{float(l_lo.value * CMINV_TO_HZ):+.12e}` Hz), `pair/mode = {float(audit_lo.pair_to_mode_ratio):+.6e}`",
        f"- high : `D_v(0) = {float(dv_hi.value_for_state(_zero_state(dv_hi.mode_kinds))):+.12e}` cm^-1, `beta_parallel = {[float(v) for _, v in sorted(betas_hi.beta_parallel.items())]}`, `beta_perpendicular = {[float(v) for _, v in sorted(betas_hi.beta_perpendicular.items())]}`, `L = {float(l_hi.value):+.12e}` cm^-1 (`{float(l_hi.value * CMINV_TO_HZ):+.12e}` Hz), `pair/mode = {float(audit_hi.pair_to_mode_ratio):+.6e}`",
        f"- comp : `D_v(0) = {float(dv_dual.value_for_state(_zero_state(dv_dual.mode_kinds))):+.12e}` cm^-1, `beta_parallel = {[float(v) for _, v in sorted(betas_dual.beta_parallel.items())]}`, `beta_perpendicular = {[float(v) for _, v in sorted(betas_dual.beta_perpendicular.items())]}`, `L = {float(l_dual.value):+.12e}` cm^-1 (`{float(l_dual.value * CMINV_TO_HZ):+.12e}` Hz), `pair/mode = {float(audit_dual.pair_to_mode_ratio):+.6e}`",
        "",
        "## Alpha branch (cm^-1)",
        "",
    ]

    for label, alpha in (("low", alpha_lo), ("high", alpha_hi), ("comp", alpha_dual)):
        summary = _alpha_summary(alpha)
        lines.extend(
            [
                f"- {label} `alpha_linear_cm = {summary['alpha_linear_cm']}`",
                f"  total(a,b,c)   = {summary['alpha_total_sum_abc_cm']}",
                f"  coriolis(a,b,c)= {summary['alpha_coriolis_sum_abc_cm']}",
                f"  inertia(a,b,c) = {summary['alpha_inertia_sum_abc_cm']}",
                f"  anharm(a,b,c)  = {summary['alpha_anharmonic_sum_abc_cm']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The harmonic/order-2 branch follows the chosen harmonic model, so the composite value is the high-level one.",
            "- The order-3 alpha and sextic branches respond strongly to the injected low-level cubic field.",
            "- The order-4 linear branch is evaluated in the same-normal-modes approximation: high-level harmonic model, low-level cubic/quartic tensors, no dense re-expansion between mode bases.",
            f"- For {molecule_label} the overlap audit is strong enough to make this approximation credible, but not trivial enough to make the example uninformative.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-fchk", default=str(LINEAR_CASES / "nno_HPCS2.fchk"))
    parser.add_argument("--low-log", default=str(REPO_ROOT / "nno_HPCS2.log"))
    parser.add_argument("--high-fchk", default=str(LINEAR_CASES / "nno_DPCS3.fchk"))
    parser.add_argument("--high-log", default=str(REPO_ROOT / "nno_DPCS3.log"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = build_nno_composite_report(
        args.low_fchk,
        args.low_log,
        args.high_fchk,
        args.high_log,
    )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
