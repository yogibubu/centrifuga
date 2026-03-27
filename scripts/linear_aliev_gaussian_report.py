#!/usr/bin/env python3
"""Compare reduced linear-Aliev Gaussian bootstraps across linear species."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_vpt_parser import parse_gaussian_linear_ltype_constants, parse_gaussian_linear_rotdist_constants
from linear_dv_aliev_terms import (
    build_explicit_aliev_L_model,
    build_explicit_aliev_beta_breakdown,
    build_explicit_aliev_dv_compactness_audit,
    build_explicit_aliev_dv_general_model,
    build_explicit_aliev_dv_legacy_compact_model,
    build_explicit_aliev_dv_model,
    build_explicit_aliev_pair_channel_breakdown,
    build_explicit_aliev_uv_breakdown,
    make_linear_aliev_explicit_inputs_from_mapping,
)
from scripts.build_linear_aliev_payload_from_gaussian import build_payload


CN_SOURCES = (
    "alpha_perp_with_Bxx_equals_minus_alpha_perp",
    "alpha_perp_with_Bxx_equals_minus_half_alpha_perp",
    "didq_linear_v_iscr",
)
ZETA_REDUCTIONS = ("pair_offdiag", "principal_direction", "component_tb", "norm", "maxabs")


def _species_paths(gaussian_dir: Path, species: str) -> tuple[Path, Path]:
    fchk = gaussian_dir / f"{species}.fchk"
    log = gaussian_dir / f"{species}.log"
    if not fchk.exists() or not log.exists():
        alt_fchk = ROOT / f"{species}.fchk"
        alt_log = ROOT / f"{species}.log"
        if alt_fchk.exists() and alt_log.exists():
            return alt_fchk, alt_log
        raise FileNotFoundError(f"Missing Gaussian inputs for {species}: {fchk.name}, {log.name}")
    return fchk, log


def _state_zero(n_parallel: int, n_perp: int) -> tuple[int, ...]:
    return (0,) * (n_parallel + n_perp)


def _report_species(gaussian_dir: Path, species: str, beta_t_xf_cross_sign: int | None) -> None:
    fchk, log = _species_paths(gaussian_dir, species)
    print(f"\n=== {species} ===")
    qconst = parse_gaussian_linear_ltype_constants(log)
    rotdist = parse_gaussian_linear_rotdist_constants(log)
    print("Gaussian linear constants:")
    print(f"  q^e = {dict(sorted(qconst.q_e_mhz.items()))}")
    print(f"  q^J = {dict(sorted(qconst.q_j_mhz.items()))}")
    print(f"  q^K = {dict(sorted(qconst.q_k_mhz.items()))}")
    print(f"  D = {rotdist.d_mhz} MHz")
    print(f"  H = {rotdist.h_mhz} MHz")
    qe_sorted = [float(v) for _, v in sorted(qconst.q_e_mhz.items())]
    baseline = None
    for zeta_reduction in ZETA_REDUCTIONS:
        for source in CN_SOURCES:
            for pair_seed_source in ("gaussian_qe_source", "rotder_seed_gram"):
                payload = build_payload(
                    fchk_path=str(fchk),
                    log_path=str(log),
                    cn_source=source,
                    zeta_reduction=zeta_reduction,
                    pair_seed_source=pair_seed_source,
                    beta_t_xf_cross_sign=beta_t_xf_cross_sign,
                )
                inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced_input")
                dv = build_explicit_aliev_dv_general_model(inputs)
                dv_legacy = build_explicit_aliev_dv_legacy_compact_model(inputs)
                dv_general = build_explicit_aliev_dv_general_model(inputs)
                compactness = build_explicit_aliev_dv_compactness_audit(inputs)
                decomp_pair = build_explicit_aliev_pair_channel_breakdown(inputs)
                l_model = build_explicit_aliev_L_model(inputs)
                breakdown = build_explicit_aliev_beta_breakdown(inputs)
                uv_breakdown = build_explicit_aliev_uv_breakdown(inputs)
                beta_n = [float(dv.beta_parallel_mode_terms[i]) for i in range(len(inputs.omega_parallel))]
                beta_t = [float(dv.beta_perpendicular[i]) for i in range(len(dv.beta_perpendicular))]
                dv0 = float(dv.value_for_state(_state_zero(len(beta_n), len(beta_t))))
                dv0_general = float(dv_general.value_for_state(_state_zero(len(beta_n), len(beta_t))))
                lval = float(l_model.value)
                print(f"zeta_reduction = {zeta_reduction}, cn_source = {source}, pair_seed_source = {pair_seed_source}, beta_t_xf_cross_sign = {payload['metadata']['beta_t_xf_cross_sign']}")
                print(f"  parallel modes: {payload['metadata']['parallel_mode_indices_0based']}")
                print(f"  perpendicular pairs: {payload['metadata']['perpendicular_pairs_0based']}")
                print(f"  bxx_parallel = {payload['bxx_parallel']}")
                print(f"  zeta_reduction(meta) = {payload['metadata']['zeta_reduction']}")
                print(f"  beta_parallel = {beta_n}")
                print(f"  beta_perpendicular = {beta_t}")
                print(f"  D_v(0) general/decompacted = {dv0:.12f} cm^-1")
                print(f"  D_v(0) compact/legacy = {float(dv_legacy.value_for_state(_state_zero(len(beta_n), len(beta_t)))):.12f} cm^-1")
                print(f"  L = {lval:.12e} cm^-1")
                print(f"  general_alias_matches_general = {dv_general == dv}")
                print(
                    "  compactness audit:"
                    f" max_mode = {float(sp.N(compactness.max_parallel_mode_term)):+.12e},"
                    f" max_pair = {float(sp.N(compactness.max_parallel_pair_term)):+.12e},"
                    f" pair/mode = {float(sp.N(compactness.pair_to_mode_ratio)):+.6e}"
                )
                if payload["metadata"].get("pair_seed_status") == "non_physical_bridge":
                    print("  note: bending pair seed is a Gaussian q^e -> pairwise l-type J0 proxy; v4/v5 comparison is qualitative only.")
                print("  dominant beta blocks:")
                for idx in range(len(dv.beta_parallel)):
                    pieces = {k: float(v) for k, v in breakdown.parallel[idx].items() if k != "total"}
                    top = sorted(pieces.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                    print(f"    beta_parallel[{idx}] top = {top}")
                for idx in range(len(dv.beta_perpendicular)):
                    pieces = {k: float(v) for k, v in breakdown.perpendicular[idx].items() if k != "total"}
                    top = sorted(pieces.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                    print(f"    beta_perpendicular[{idx}] top = {top}")
                print("  uv breakdown:")
                for idx in range(len(dv.beta_perpendicular)):
                    block = {k: float(v) for k, v in uv_breakdown.perpendicular[idx].items()}
                    print(f"    pair {idx}: {block}")
                print("  decompacted parallel pair channels:")
                for pair_key in sorted(dv_general.beta_parallel_pair_terms):
                    gamma = float(dv_general.beta_parallel_pair_terms[pair_key])
                    print(f"    gamma{pair_key} = {gamma:+.12e} cm^-1")
                    for t_idx in sorted(decomp_pair.parallel[pair_key]):
                        block = decomp_pair.parallel[pair_key][t_idx]
                        print(
                            "      "
                            f"t={t_idx}: offdiag_n={float(block['offdiag_n']):+.6e}, "
                            f"offdiag_np={float(block['offdiag_np']):+.6e}, "
                            f"u_working={float(block['u_from_working']):+.6e}, "
                            f"v_working={float(block['v_from_working']):+.6e}"
                        )
                print("  pair-seed vs Gaussian q^e:")
                for idx in range(len(dv.beta_perpendicular)):
                    pair_seed_cm = float(breakdown.perpendicular[idx]["pair_seed"])
                    pair_seed_mhz = pair_seed_cm * 29979.2458
                    qe_mhz = qe_sorted[idx] if idx < len(qe_sorted) else float("nan")
                    ratio = pair_seed_mhz / qe_mhz if abs(qe_mhz) > 1.0e-30 else float("nan")
                    print(
                        f"    pair {idx}: pair_seed = {pair_seed_mhz:.6f} MHz, "
                        f"q^e = {qe_mhz:.6f} MHz, ratio = {ratio:.6f}"
                    )
                key = (zeta_reduction, source, pair_seed_source)
                if baseline is None:
                    baseline = {"dv0": dv0, "l": lval, "beta_n": beta_n, "beta_t": beta_t, "key": key}
                else:
                    print(f"  deltas vs baseline {baseline['key']}:")
                    print(f"    ΔD_v(0) = {dv0 - baseline['dv0']:+.12e} cm^-1")
                    print(f"    ΔL = {lval - baseline['l']:+.12e} cm^-1")
                    print(f"    max|Δbeta_parallel| = {max(abs(x-y) for x,y in zip(beta_n, baseline['beta_n'], strict=True)):.12e}")
                    print(f"    max|Δbeta_perpendicular| = {max(abs(x-y) for x,y in zip(beta_t, baseline['beta_t'], strict=True)):.12e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gaussian-dir", default="/Users/vincenzobarone/centrifugal/gaussian")
    ap.add_argument("--species", nargs="+", default=["c2h2", "hcn"])
    ap.add_argument("--beta-t-xf-cross-sign", default=None, type=int, choices=(-1, 1))
    args = ap.parse_args()
    gaussian_dir = Path(args.gaussian_dir)
    for species in args.species:
        _report_species(gaussian_dir, species, args.beta_t_xf_cross_sign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
