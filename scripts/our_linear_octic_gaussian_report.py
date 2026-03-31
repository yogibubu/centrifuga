#!/usr/bin/env python3
"""Evaluate the current CeDiTT4-derived linear octic constant from Gaussian data.

This report keeps the exact-linear pure rotational scalar branch separate from
the degenerate perpendicular branch. The latter is not yet operational in the
current code path because the exact linear builder for the S111 sector is still
missing. The reported ``L_our_pure_rotational`` therefore contains only the
fully operational branch derived from the current CeDiTT4 program.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from linear_dv_aliev_terms import (  # noqa: E402
    _quartic_value_if_supported,
    build_explicit_aliev_families,
    build_explicit_aliev_L_model,
    make_linear_aliev_explicit_inputs_from_mapping,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload  # noqa: E402
from scripts.linear_s111_builder import build_linear_s111_builder_report  # noqa: E402


@dataclass(frozen=True)
class OurLinearOcticPureReport:
    species: str
    parallel_mode_indices_0based: tuple[int, ...]
    perpendicular_pairs_0based: tuple[tuple[int, int], ...]
    quartic_block: sp.Expr
    term3_block: sp.Expr
    term5_block: sp.Expr
    total: sp.Expr
    aliev_total: sp.Expr
    difference_vs_aliev: sp.Expr
    degenerate_pair_branch: tuple[dict[str, object], ...]


def _species_paths(gaussian_dir: Path, species: str) -> tuple[Path, Path]:
    lower = species.lower()
    candidates = [
        (gaussian_dir / f"{lower}.fchk", gaussian_dir / f"{lower}.log"),
        (ROOT / f"{lower}.fchk", ROOT / f"{lower}.log"),
        (ROOT / f"{species}.fchk", ROOT / f"{species}.log"),
    ]
    for fchk, log in candidates:
        if fchk.exists() and log.exists():
            return fchk, log
    raise FileNotFoundError(f"Missing Gaussian inputs for {species}.")


def _our_linear_octic_pure_from_inputs(inputs) -> dict[str, sp.Expr]:
    fam = build_explicit_aliev_families(inputs)
    omega_n = inputs.omega_parallel
    n_parallel = len(omega_n)

    quartic_block = sp.Integer(0)
    for n in range(n_parallel):
        for np_ in range(n_parallel):
            for npp in range(n_parallel):
                for nppp in range(n_parallel):
                    q = _quartic_value_if_supported(inputs.k4_parallel, inputs.k4_reduced, n, np_, npp, nppp)
                    if q == 0:
                        continue
                    quartic_block += sp.Rational(1, 24) * q * fam.C_n[n] * fam.C_n[np_] * fam.C_n[npp] * fam.C_n[nppp]
    quartic_block = sp.simplify(quartic_block)

    # Exact true-linear reduction derived in the CeDiTT4-based octic program:
    # r_k = Σ_l C_l r_kl
    term5_block = sp.simplify(
        -sum(fam.r_n[n] ** 2 / (2 * omega_n[n]) for n in range(n_parallel))
    )

    term3_kernel = sp.Integer(0)
    for n in range(n_parallel):
        inner = sp.Integer(0)
        for np_ in range(n_parallel):
            for npp in range(n_parallel):
                inner += fam.C_n[np_] * fam.C_n[npp] * fam.r_nn[(np_, npp)]
        term3_kernel += fam.r_n[n] * inner / omega_n[n]
    term3_block = sp.simplify(-sp.Rational(63, 256) * term3_kernel)

    total = sp.simplify(quartic_block + term3_block + term5_block)

    return {
        "quartic_block": quartic_block,
        "term3_block": term3_block,
        "term5_block": term5_block,
        "total": total,
        "aliev_total": sp.simplify(build_explicit_aliev_L_model(inputs).value),
    }


def build_our_linear_octic_pure_report(
    *,
    species: str,
    gaussian_dir: Path,
    cn_source: str = "alpha_perp_with_Bxx_equals_minus_alpha_perp",
    zeta_reduction: str = "pair_offdiag",
    pair_seed_source: str = "gaussian_qe_source",
    beta_t_xf_cross_sign: int | None = None,
) -> OurLinearOcticPureReport:
    fchk, log = _species_paths(gaussian_dir, species)
    payload = build_payload(
        fchk_path=str(fchk),
        log_path=str(log),
        cn_source=cn_source,
        zeta_reduction=zeta_reduction,
        pair_seed_source=pair_seed_source,
        beta_t_xf_cross_sign=beta_t_xf_cross_sign,
    )
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced_input")
    ours = _our_linear_octic_pure_from_inputs(inputs)
    return OurLinearOcticPureReport(
        species=species,
        parallel_mode_indices_0based=tuple(payload["metadata"]["parallel_mode_indices_0based"]),
        perpendicular_pairs_0based=tuple(tuple(x) for x in payload["metadata"]["perpendicular_pairs_0based"]),
        quartic_block=ours["quartic_block"],
        term3_block=ours["term3_block"],
        term5_block=ours["term5_block"],
        total=ours["total"],
        aliev_total=ours["aliev_total"],
        difference_vs_aliev=sp.simplify(ours["total"] - ours["aliev_total"]),
        degenerate_pair_branch=build_linear_s111_builder_report(
            species=species,
            gaussian_dir=gaussian_dir,
        ).pairs,
    )


def _format_report(report: OurLinearOcticPureReport) -> str:
    lines = [
        f"=== {report.species} ===",
        f"parallel modes (0-based): {list(report.parallel_mode_indices_0based)}",
        f"perpendicular pairs (0-based): {[list(x) for x in report.perpendicular_pairs_0based]}",
        "current CeDiTT4 linear-octic split:",
        f"  quartic_block = {float(sp.N(report.quartic_block)):+.12e} cm^-1",
        f"  term3_block  = {float(sp.N(report.term3_block)):+.12e} cm^-1",
        f"  term5_block  = {float(sp.N(report.term5_block)):+.12e} cm^-1",
        f"  L_our_pure_rotational = {float(sp.N(report.total)):+.12e} cm^-1",
        "separate degenerate/perpendicular branch:",
        "  L_deg^perp = (18/35) S_perp^2 tau_perp",
        "  observable exact-linear replacement:",
    ]
    for pair in report.degenerate_pair_branch:
        lines.extend(
            [
                f"    pair {pair['pair_label']} modes={list(pair['modes'])} freq={pair['freq_cm']:.6f} cm^-1",
                f"      carrier = {pair['carrier']}",
                f"      q_e = {pair['q_e_hz']:+.12e} Hz",
                f"      q_J = {pair['q_J_hz']:+.12e} Hz",
                f"      q_H = {pair['q_H_hz']:+.12e} Hz",
            ]
        )
    lines.extend(
        [
            f"  L_Aliev = {float(sp.N(report.aliev_total)):+.12e} cm^-1",
            f"  Delta(our_pure - Aliev) = {float(sp.N(report.difference_vs_aliev)):+.12e} cm^-1",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hccd", "hcn", "ocs"])
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
        default="pair_offdiag",
        choices=("principal_direction", "norm", "maxabs", "pair_offdiag", "pair_diag", "component_ta", "component_tb", "component_ua", "component_ub"),
    )
    ap.add_argument("--pair-seed-source", default="gaussian_qe_source", choices=("gaussian_qe_source", "rotder_seed_gram"))
    ap.add_argument("--beta-t-xf-cross-sign", default=None, type=int, choices=(-1, 1))
    args = ap.parse_args()

    gaussian_dir = Path(args.gaussian_dir)
    for species in args.species:
        report = build_our_linear_octic_pure_report(
            species=species,
            gaussian_dir=gaussian_dir,
            cn_source=args.cn_source,
            zeta_reduction=args.zeta_reduction,
            pair_seed_source=args.pair_seed_source,
            beta_t_xf_cross_sign=args.beta_t_xf_cross_sign,
        )
        print(_format_report(report))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
