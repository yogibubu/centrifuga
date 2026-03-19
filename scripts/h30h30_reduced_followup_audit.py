#!/usr/bin/env python3
"""Audit the reduced iii,iij / iij,iij pivots against the water-like residual."""

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
}
DIAG_BASIS = (
    "diag_0_iii_iii",
    "diag_1_iii_iii_rotmix_candidate",
    "diag_bench_resid11_candidate",
    "diag_bench_resid13_candidate",
)
REDUCED = (
    "waterlike_reduced_iii_iij_candidate",
    "waterlike_reduced_iij_iij_candidate",
)


def main() -> None:
    print("H30H30 reduced follow-up audit")
    for species in ("h2o", "h2s"):
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

        def vec(name: str) -> np.ndarray:
            wat = to_watson_khz(
                pieces[name],
                abc_mhz=model.abc_mhz,
                reduction="S",
                spectroscopic_axes=quart.spectroscopic_axes,
                tau_cm_scale=1.0,
            )
            return np.array([float(wat[key]) for key in WATSON_KEYS], dtype=float)

        target = np.array([VCI_PAPER[species][key] for key in WATSON_KEYS], dtype=float)
        diag = np.column_stack([vec(name) for name in DIAG_BASIS])
        coeffs_diag, *_ = np.linalg.lstsq(diag, target, rcond=None)
        residual = target - diag @ coeffs_diag
        red = np.column_stack([vec(name) for name in REDUCED])
        coeffs_red, *_ = np.linalg.lstsq(red, residual, rcond=None)
        post = residual - red @ coeffs_red
        print(f"\n{species.upper()}")
        print(f"  ||residual|| = {np.linalg.norm(residual):.6g}")
        print(f"  ||post||     = {np.linalg.norm(post):.6g}")
        print(f"  coeffs       = {dict(zip(REDUCED, coeffs_red))}")


if __name__ == "__main__":
    main()
