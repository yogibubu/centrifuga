#!/usr/bin/env python3
"""Audit the `uv` block in the operational small-branch configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import (  # noqa: E402
    build_explicit_aliev_beta_breakdown,
    build_explicit_aliev_uv_breakdown,
    make_linear_aliev_explicit_inputs_from_mapping,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload  # noqa: E402


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
    ap.add_argument("--pair-seed-source", default="gaussian_qe_source", choices=("gaussian_qe_source", "formula"))
    args = ap.parse_args()

    payload = build_payload(
        fchk_path=args.fchk,
        log_path=args.log,
        zeta_reduction="principal_direction",
        cn_source=args.cn_source,
        pair_seed_source=args.pair_seed_source,
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced")
    beta = build_explicit_aliev_beta_breakdown(inputs)
    uv = build_explicit_aliev_uv_breakdown(inputs)

    print("=== UV Component Audit ===")
    print(f"fchk = {args.fchk}")
    print(f"log  = {args.log}")
    print("branch = principal_direction")
    print(f"cn_source = {args.cn_source}")
    print(f"pair_seed_source = {args.pair_seed_source}")
    print("")

    print("--- parallel modes ---")
    for idx in sorted(uv.parallel):
        block = {k: float(v) for k, v in uv.parallel[idx].items()}
        print(f"mode {idx}: {block}, beta_uv = {float(beta.parallel[idx]['uv_block']):.12e}")

    print("")
    print("--- perpendicular modes ---")
    for idx in sorted(uv.perpendicular):
        block = {k: float(v) for k, v in uv.perpendicular[idx].items()}
        print(f"pair {idx}: {block}, beta_uv = {float(beta.perpendicular[idx]['uv_block']):.12e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
