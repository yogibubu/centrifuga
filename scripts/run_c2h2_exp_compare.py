#!/usr/bin/env python3
"""Run an experimental C2H2 comparison against the current linear-Aliev outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_pipeline import (
    align_c2h2_linear_aliev_modes,
    build_comparison_table,
    build_exp_quantities,
    format_comparison_report,
)
from exp_data_c2h2 import get_c2h2_data
from linear_dv_aliev_terms import build_explicit_aliev_L_model, build_explicit_aliev_dv_model, make_linear_aliev_explicit_inputs_from_mapping
from scripts.build_linear_aliev_payload_from_gaussian import build_payload


def run_compare(
    *,
    fchk_path: str,
    log_path: str,
    quartic_mode: str = "reduced",
    cn_source: str = "alpha_perp_with_Bxx_equals_minus_alpha_perp",
    zeta_reduction: str = "principal_direction",
    pair_seed_source: str = "gaussian_qe_source",
) -> str:
    payload = build_payload(
        fchk_path=fchk_path,
        log_path=log_path,
        cn_source=cn_source,
        zeta_reduction=zeta_reduction,
        pair_seed_source=pair_seed_source,
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode=quartic_mode)
    dv = build_explicit_aliev_dv_model(inputs)
    l_model = build_explicit_aliev_L_model(inputs)

    aligned = align_c2h2_linear_aliev_modes(
        payload=payload,
        beta_parallel=[float(dv.beta_parallel[i]) for i in range(len(dv.beta_parallel))],
        beta_perpendicular=[float(dv.beta_perpendicular[i]) for i in range(len(dv.beta_perpendicular))],
        L_mode_calc=[float(x) for x in l_model.mode_contributions_with_shared_offset],
        L_total_calc=float(l_model.value),
    )
    exp = build_exp_quantities(get_c2h2_data())
    beta_table = build_comparison_table(aligned.modes, aligned.beta_calc, exp.beta_exp)
    dH_table = None if aligned.L_mode_calc is None else build_comparison_table(aligned.modes, aligned.L_mode_calc, exp.dH_exp)
    pair_seed_status = str(payload.get("metadata", {}).get("pair_seed_status", ""))
    beta_note = None
    if pair_seed_status == "non_physical_bridge":
        beta_note = (
            "Bending rows currently use a non-physical Gaussian q^e bridge into the pairwise l-type J0 layer. "
            "Those v4/v5 comparisons are qualitative only and should not be read as quantitative Delta D_t validation."
        )
    elif pair_seed_status == "diagnostic_rotder_seed":
        beta_note = (
            "Bending rows currently use the diagnostic rotational-derivative seed-Gram branch. "
            "This is the first geometry/mode-based zeta scaffold, but it is not yet a validated physical Aliev seed."
        )
    return format_comparison_report(
        beta_table,
        beta_note=beta_note,
        dH_table=dH_table,
        L_total_calc=aligned.L_total_calc,
        dH_note="Modewise optical comparison uses the distributed-shared-offset partition of total L; this is a provisional bookkeeping partition, but units are cm^-1 on both sides.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fchk", default="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk")
    ap.add_argument("--log", default="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log")
    ap.add_argument("--quartic-mode", default="reduced", choices=("full", "reduced", "none", "auto"))
    ap.add_argument(
        "--cn-source",
        default="alpha_perp_with_Bxx_equals_minus_alpha_perp",
        choices=(
            "alpha_perp_with_Bxx_equals_minus_alpha_perp",
            "alpha_perp_with_Bxx_equals_minus_half_alpha_perp",
            "didq_linear_v_iscr",
        ),
    )
    ap.add_argument(
        "--zeta-reduction",
        default="principal_direction",
        choices=("principal_direction", "norm", "maxabs", "pair_offdiag", "pair_diag", "component_ta", "component_tb", "component_ua", "component_ub"),
    )
    ap.add_argument("--pair-seed-source", default="gaussian_qe_source", choices=("gaussian_qe_source", "rotder_seed_gram"))
    args = ap.parse_args()
    print(
        run_compare(
            fchk_path=args.fchk,
            log_path=args.log,
            quartic_mode=args.quartic_mode,
            cn_source=args.cn_source,
            zeta_reduction=args.zeta_reduction,
            pair_seed_source=args.pair_seed_source,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
