#!/usr/bin/env python3
"""Inspect the currently implemented H30H30 terms and their resonance metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_vpt_parser import (
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
)
from h30h30_resonance import enumerate_current_h30h30_terms, summarize_terms_by_metric
from rovib_distortion import ANGSTROM_TO_BOHR, harmonic_inertia_model_from_geometry_hessian


def build_model(fchk_path: Path, log_path: Path):
    harm = parse_gaussian_fchk_harmonic_data(fchk_path)
    anh = parse_gaussian_anharmonic_force_data(log_path)
    coords_ang = harm.coords_bohr * (1.0 / ANGSTROM_TO_BOHR)
    model = harmonic_inertia_model_from_geometry_hessian(harm.masses_amu, coords_ang, harm.cartesian_force_constants)
    return model, anh.phi3_raw_au


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fchk", required=True, help="Gaussian formatted checkpoint file")
    ap.add_argument("--log", required=True, help="Gaussian anharmonic log file")
    ap.add_argument("--top", type=int, default=12, help="Number of terms to print")
    ap.add_argument(
        "--wilson-factor-cm-per-au",
        type=float,
        default=1.0,
        help="Explicit conversion factor for the H30H30 numerator into cm^-1. "
        "Default is 1.0 and should be treated as diagnostic only until the unit mapping is fixed.",
    )
    args = ap.parse_args()

    model, phi3 = build_model(Path(args.fchk), Path(args.log))
    terms = enumerate_current_h30h30_terms(
        model.dInv_au,
        phi3,
        model.vib_freq_cm,
        wilson_factor_cm_per_au=args.wilson_factor_cm_per_au,
    )
    top_terms = summarize_terms_by_metric(terms, top_n=args.top)

    print("Current H30H30 diagnostic")
    print(f"fchk = {args.fchk}")
    print(f"log  = {args.log}")
    print(f"wilson_factor_cm_per_au = {args.wilson_factor_cm_per_au}")
    print("NOTE: this enumerates only the currently coded positive-sum family omega_i + omega_j + 1.")
    print()
    for term in top_terms:
        modes_1based = tuple(idx + 1 for idx in term.modes)
        print(
            f"{term.component:>8}  family={term.family:<20} modes={modes_1based!s:<12} "
            f"diag={str(term.diagonal_modes):<5} numerator_au={term.numerator_au: .6e} "
            f"coupling_cm={term.coupling_cm: .6e} denominator_cm={term.denominator_cm: .6e} "
            f"Martin={term.martin_ratio: .6e}"
        )


if __name__ == "__main__":
    main()
