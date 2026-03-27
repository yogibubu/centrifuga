#!/usr/bin/env python3
"""Audit the residual sensitivity of the operational small doublet branch.

This report is intentionally narrow:
- fix the Coriolis branch to the operational small-branch default
- compare only the remaining sensitivity to the `B_n^(xx)` bootstrap

It is meant to separate the degenerate-doublet convention issue from the
geometric/force-field bridge issue.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import (
    build_explicit_aliev_L_model,
    build_explicit_aliev_beta_breakdown,
    build_explicit_aliev_dv_legacy_compact_model,
    make_linear_aliev_explicit_inputs_from_mapping,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload


OPERATIONAL_BRANCH = "principal_direction"
CN_SOURCES = (
    "alpha_perp_with_Bxx_equals_minus_alpha_perp",
    "alpha_perp_with_Bxx_equals_minus_half_alpha_perp",
    "didq_linear_v_iscr",
)


def _state_zero(n_parallel: int, n_perp: int) -> tuple[int, ...]:
    return (0,) * (n_parallel + n_perp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fchk", default="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk")
    ap.add_argument("--log", default="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log")
    ap.add_argument("--pair-seed-source", default="gaussian_qe_source", choices=("gaussian_qe_source",))
    args = ap.parse_args()

    print("=== Small-Branch Residual Audit ===")
    print(f"fchk = {args.fchk}")
    print(f"log  = {args.log}")
    print(f"pair_seed_source = {args.pair_seed_source}")
    print(f"branch = {OPERATIONAL_BRANCH}")
    print("")
    print("cn_source                               max|beta_n|       max|beta_t|       max|uv_n|         max|uv_t|         D_v(0)           L")

    for cn_source in CN_SOURCES:
        payload = build_payload(
            fchk_path=args.fchk,
            log_path=args.log,
            zeta_reduction=OPERATIONAL_BRANCH,
            cn_source=cn_source,
            pair_seed_source=args.pair_seed_source,
        )
        inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced_input")
        dv = build_explicit_aliev_dv_legacy_compact_model(inputs)
        bd = build_explicit_aliev_beta_breakdown(inputs)
        l_model = build_explicit_aliev_L_model(inputs)
        beta_n = [float(dv.beta_parallel[i]) for i in range(len(dv.beta_parallel))]
        beta_t = [float(dv.beta_perpendicular[i]) for i in range(len(dv.beta_perpendicular))]
        uv_n = [float(bd.parallel[i]["uv_block"]) for i in range(len(dv.beta_parallel))]
        uv_t = [float(bd.perpendicular[i]["uv_block"]) for i in range(len(dv.beta_perpendicular))]
        dv0 = float(dv.value_for_state(_state_zero(len(beta_n), len(beta_t))))
        print(
            f"{cn_source:38s} "
            f"{max(abs(x) for x in beta_n):13.6e} "
            f"{max(abs(x) for x in beta_t):13.6e} "
            f"{max(abs(x) for x in uv_n):13.6e} "
            f"{max(abs(x) for x in uv_t):13.6e} "
            f"{dv0:13.6e} "
            f"{float(l_model.value):13.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
