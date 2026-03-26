#!/usr/bin/env python3
"""Scan candidate degenerate-doublet Coriolis component conventions.

This is a diagnostic script for the linear Aliev prototype. It compares the
four explicit component choices suggested by the paper-style definition

    zeta_nt = zeta^x_(n,tb)

against the legacy scalar reductions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import build_explicit_aliev_dv_model, make_linear_aliev_explicit_inputs_from_mapping
from scripts.build_linear_aliev_payload_from_gaussian import build_payload


ZETA_VARIANTS = (
    "component_ta",
    "component_tb",
    "component_ua",
    "component_ub",
    "maxabs",
    "norm",
)


def _state_zero(n_parallel: int, n_perp: int) -> tuple[int, ...]:
    return (0,) * (n_parallel + n_perp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fchk", default="/Users/vincenzobarone/centrifugal/gaussian/c2h2.fchk")
    ap.add_argument("--log", default="/Users/vincenzobarone/centrifugal/gaussian/c2h2.log")
    ap.add_argument(
        "--cn-source",
        default="alpha_perp_with_Bxx_equals_minus_alpha_perp",
        choices=(
            "alpha_perp_with_Bxx_equals_minus_alpha_perp",
            "alpha_perp_with_Bxx_equals_minus_half_alpha_perp",
            "didq_linear_v_iscr",
        ),
    )
    ap.add_argument("--pair-seed-source", default="gaussian_qe_source", choices=("gaussian_qe_source",))
    args = ap.parse_args()

    print("=== Doublet Convention Scan ===")
    print(f"fchk = {args.fchk}")
    print(f"log  = {args.log}")
    print(f"cn_source = {args.cn_source}")
    print(f"pair_seed_source = {args.pair_seed_source}")
    print("")
    print("variant         beta_parallel(maxabs)   beta_perp(maxabs)     D_v(0)          zeta_nt")

    for variant in ZETA_VARIANTS:
        payload = build_payload(
            fchk_path=args.fchk,
            log_path=args.log,
            cn_source=args.cn_source,
            zeta_reduction=variant,
            pair_seed_source=args.pair_seed_source,
        )
        inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced")
        dv = build_explicit_aliev_dv_model(inputs)
        beta_n = [float(dv.beta_parallel[i]) for i in range(len(dv.beta_parallel))]
        beta_t = [float(dv.beta_perpendicular[i]) for i in range(len(dv.beta_perpendicular))]
        dv0 = float(dv.value_for_state(_state_zero(len(beta_n), len(beta_t))))
        print(
            f"{variant:15s} "
            f"{max(abs(x) for x in beta_n):16.6e} "
            f"{max(abs(x) for x in beta_t):16.6e} "
            f"{dv0:16.6e} "
            f"{payload['zeta_nt']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
