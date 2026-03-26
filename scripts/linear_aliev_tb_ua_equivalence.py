#!/usr/bin/env python3
"""Compare the two small-branch doublet conventions term by term.

The goal is narrow:
- keep the analysis on the physically acceptable small branch
- compare `component_tb` and `component_ua`
- report whether their effect is just a basis relabeling in the current code
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import (  # noqa: E402
    build_explicit_aliev_L_model,
    build_explicit_aliev_beta_breakdown,
    build_explicit_aliev_dv_model,
    build_explicit_aliev_uv_breakdown,
    make_linear_aliev_explicit_inputs_from_mapping,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload  # noqa: E402


def _state_zero(n_parallel: int, n_perp: int) -> tuple[int, ...]:
    return (0,) * (n_parallel + n_perp)


def _maxabs(values: list[float]) -> float:
    return max((abs(x) for x in values), default=0.0)


def _vector_diff(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _print_vector(label: str, values: list[float]) -> None:
    formatted = ", ".join(f"{v:.6e}" for v in values)
    print(f"{label:20s} [{formatted}]")


def _collect(branch: str, fchk: str, log: str, cn_source: str, pair_seed_source: str) -> dict[str, object]:
    payload = build_payload(
        fchk_path=fchk,
        log_path=log,
        zeta_reduction=branch,
        cn_source=cn_source,
        pair_seed_source=pair_seed_source,
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced")
    dv = build_explicit_aliev_dv_model(inputs)
    bd = build_explicit_aliev_beta_breakdown(inputs)
    uvd = build_explicit_aliev_uv_breakdown(inputs)
    lm = build_explicit_aliev_L_model(inputs)
    beta_n = [float(dv.beta_parallel[i]) for i in range(len(dv.beta_parallel))]
    beta_t = [float(dv.beta_perpendicular[i]) for i in range(len(dv.beta_perpendicular))]
    uv_n = [float(bd.parallel[i]["uv_block"]) for i in range(len(dv.beta_parallel))]
    uv_t = [float(bd.perpendicular[i]["uv_block"]) for i in range(len(dv.beta_perpendicular))]
    zeta = [[float(x) for x in row] for row in payload["zeta_nt"]]
    return {
        "payload": payload,
        "beta_n": beta_n,
        "beta_t": beta_t,
        "uv_n": uv_n,
        "uv_t": uv_t,
        "L": float(lm.value),
        "Dv0": float(dv.value_for_state(_state_zero(len(beta_n), len(beta_t)))),
        "uv_diag_r": [float(uvd.perpendicular[i]["diag_rterm"]) for i in range(len(dv.beta_perpendicular))],
        "uv_diag_b": [float(uvd.perpendicular[i]["diag_bterm"]) for i in range(len(dv.beta_perpendicular))],
        "zeta": zeta,
    }


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

    tb = _collect("component_tb", args.fchk, args.log, args.cn_source, args.pair_seed_source)
    ua = _collect("component_ua", args.fchk, args.log, args.cn_source, args.pair_seed_source)

    print("=== Small-Branch Equivalence Audit ===")
    print(f"fchk = {args.fchk}")
    print(f"log  = {args.log}")
    print(f"cn_source = {args.cn_source}")
    print(f"pair_seed_source = {args.pair_seed_source}")
    print("")

    print("--- component_tb ---")
    _print_vector("beta_parallel", tb["beta_n"])
    _print_vector("beta_perp", tb["beta_t"])
    _print_vector("uv_parallel", tb["uv_n"])
    _print_vector("uv_perp", tb["uv_t"])
    _print_vector("uv_diag_r", tb["uv_diag_r"])
    _print_vector("uv_diag_b", tb["uv_diag_b"])
    print(f"{'D_v(0)':20s} {tb['Dv0']:.12e}")
    print(f"{'L':20s} {tb['L']:.12e}")
    print(f"{'zeta_nt':20s} {tb['zeta']}")
    print("")

    print("--- component_ua ---")
    _print_vector("beta_parallel", ua["beta_n"])
    _print_vector("beta_perp", ua["beta_t"])
    _print_vector("uv_parallel", ua["uv_n"])
    _print_vector("uv_perp", ua["uv_t"])
    _print_vector("uv_diag_r", ua["uv_diag_r"])
    _print_vector("uv_diag_b", ua["uv_diag_b"])
    print(f"{'D_v(0)':20s} {ua['Dv0']:.12e}")
    print(f"{'L':20s} {ua['L']:.12e}")
    print(f"{'zeta_nt':20s} {ua['zeta']}")
    print("")

    print("--- absolute differences (tb - ua) ---")
    for label, key in (
        ("beta_parallel", "beta_n"),
        ("beta_perp", "beta_t"),
        ("uv_parallel", "uv_n"),
        ("uv_perp", "uv_t"),
        ("uv_diag_r", "uv_diag_r"),
        ("uv_diag_b", "uv_diag_b"),
    ):
        diff = _vector_diff(tb[key], ua[key])
        _print_vector(label, diff)
        print(f"{'max|' + label + '|':20s} {_maxabs(diff):.12e}")
    print(f"{'|D_v(0) diff|':20s} {abs(tb['Dv0'] - ua['Dv0']):.12e}")
    print(f"{'|L diff|':20s} {abs(tb['L'] - ua['L']):.12e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
