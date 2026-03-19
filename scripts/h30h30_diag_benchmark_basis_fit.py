#!/usr/bin/env python3
"""Fit the molecular H30H30 targets in the benchmark-derived diagonal basis."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from channel_contributions import to_watson_khz
from gaussian_vpt_parser import (
    parse_gaussian_anharmonic_force_data,
    parse_gaussian_fchk_harmonic_data,
    parse_gaussian_quartic_benchmark,
)
from harmonic_convention import align_cubic_to_harmonic_model, build_default_harmonic_model
from quartic_channels import channel_h30h30_decomposed
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")
VCI_PAPER = {
    "h2o": {"DJ": 34.95 - 34.98, "DJK": -168.92 - (-151.17), "DK": 972.11 - 778.31, "d1": 13.909 - 13.980, "d2": 33.5 - 11.1},
    "h2s": {"DJ": 19.651 - 19.712, "DJK": -73.15 - (-72.16), "DK": 115.76 - 109.90, "d1": -8.248 - (-8.111), "d2": 0.6642 - 0.8368},
    "h2co": {"DJ": 68.746 - 69.622, "DJK": 1.28671 - 1.31233, "DK": 18.39056 - 18.54006, "d1": -10.1007 - (-9.8149), "d2": -2.4484 - (-2.1420)},
    "h2cs": {"DJ": 18.75614 - 18.66391, "DJK": 0.51221 - 0.52080, "DK": 22.34638 - 22.10999, "d1": -1.1693 - (-1.1267), "d2": -0.17351 - (-0.14854)},
}
BASIS_NAMES = (
    "diag_0_iii_iii",
    "diag_1_iii_iii_rotmix_candidate",
    "diag_bench_resid11_candidate",
    "diag_bench_resid13_candidate",
)


def _vec(pieces, model, quart, name: str) -> np.ndarray:
    wat = to_watson_khz(
        pieces[name],
        abc_mhz=model.abc_mhz,
        reduction="S",
        spectroscopic_axes=quart.spectroscopic_axes,
        tau_cm_scale=1.0,
    )
    return np.array([float(wat[key]) for key in WATSON_KEYS], dtype=float)


def main() -> None:
    print("H30H30 molecular fit in the benchmark-derived diagonal basis")
    print("basis =", BASIS_NAMES)
    for species in ("h2o", "h2s", "h2co", "h2cs"):
        fchk = parse_gaussian_fchk_harmonic_data(Path(f"{species}.fchk"))
        anh = parse_gaussian_anharmonic_force_data(Path(f"{species}.log"))
        quart = parse_gaussian_quartic_benchmark(Path(f"{species}.log"))
        model, _ = build_default_harmonic_model(
            fchk.masses_amu,
            fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
            fchk.cartesian_force_constants,
        )
        cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
        pieces = channel_h30h30_decomposed(
            sp.MutableDenseNDimArray(model.dInv_au.tolist()),
            sp.MutableDenseNDimArray(cubic.raw_au.tolist()),
            tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm),
            hbar=sp.Float(1.0),
        )
        basis = np.column_stack([_vec(pieces, model, quart, name) for name in BASIS_NAMES])
        target = np.array([VCI_PAPER[species][key] for key in WATSON_KEYS], dtype=float)
        coeffs, *_ = np.linalg.lstsq(basis, target, rcond=None)
        fit = basis @ coeffs
        print(f"\n{species.upper()}")
        print("  coeffs =", dict(zip(BASIS_NAMES, coeffs)))
        print(f"  ||fit|| / ||target|| = {np.linalg.norm(fit) / np.linalg.norm(target):.6g}")
        print(f"  ||target-fit||       = {np.linalg.norm(target - fit):.6g}")


if __name__ == "__main__":
    main()
