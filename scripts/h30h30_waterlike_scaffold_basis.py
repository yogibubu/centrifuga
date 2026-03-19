#!/usr/bin/env python3
"""Extract the water-like residual basis directly in follow-up scaffold space."""

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
FOLLOWUP_CANDIDATES = (
    "diag_1_iii_iij_0",
    "diag_0_iii_iij_1_candidate",
    "diag_0_iii_iij_2_resonance_candidate",
    "diag_0_iii_iij_2_regularized_preview",
    "diag_0_iij_iij_1_candidate",
    "diag_0_iij_iij_2_candidate",
)


def main() -> None:
    coeff_rows = []
    print("H30H30 water-like basis in scaffold coefficient space")
    print("follow-up candidates =", FOLLOWUP_CANDIDATES)
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

        diag = np.column_stack([vec(name) for name in DIAG_BASIS])
        target = np.array([VCI_PAPER[species][key] for key in WATSON_KEYS], dtype=float)
        coeffs_diag, *_ = np.linalg.lstsq(diag, target, rcond=None)
        residual = target - diag @ coeffs_diag
        follow = np.column_stack([vec(name) for name in FOLLOWUP_CANDIDATES])
        coeffs_follow, *_ = np.linalg.lstsq(follow, residual, rcond=None)
        coeff_rows.append(coeffs_follow)
        print(f"\n{species.upper()}")
        print("  follow-up coeffs =", dict(zip(FOLLOWUP_CANDIDATES, coeffs_follow)))

    coeff_mat = np.array(coeff_rows, dtype=float)
    u, s, vt = np.linalg.svd(coeff_mat, full_matrices=False)
    print()
    print(f"coefficient-matrix rank = {np.linalg.matrix_rank(coeff_mat)}")
    print(f"singular values = {s}")
    print("scaffold-space basis vectors:")
    for i, row in enumerate(vt):
        print(f"  scaffold_basis_{i+1} = {dict(zip(FOLLOWUP_CANDIDATES, row))}")


if __name__ == "__main__":
    main()
