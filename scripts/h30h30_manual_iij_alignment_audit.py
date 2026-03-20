#!/usr/bin/env python3
"""Measure alignment of the post-trusted residual with the manual H30H30 iij axis."""

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
    "h2o": {
        "PFIT_RVCI_E": {"DJ": 34.95, "DJK": -168.92, "DK": 972.11, "d1": 13.909, "d2": 33.5},
        "VPT": {"DJ": 34.98, "DJK": -151.17, "DK": 778.31, "d1": 13.980, "d2": 11.1},
    },
    "h2s": {
        "PFIT_RVCI_E": {"DJ": 19.651, "DJK": -73.15, "DK": 115.76, "d1": -8.248, "d2": 0.6642},
        "VPT": {"DJ": 19.712, "DJK": -72.16, "DK": 109.90, "d1": -8.111, "d2": 0.8368},
    },
    "h2co": {
        "PFIT_RVCI_E": {"DJ": 68.746, "DJK": 1.28671, "DK": 18.39056, "d1": -10.1007, "d2": -2.4484},
        "VPT": {"DJ": 69.622, "DJK": 1.31233, "DK": 18.54006, "d1": -9.8149, "d2": -2.1420},
    },
    "h2cs": {
        "PFIT_RVCI_E": {"DJ": 18.75614, "DJK": 0.51221, "DK": 22.34638, "d1": -1.1693, "d2": -0.17351},
        "VPT": {"DJ": 18.66391, "DJK": 0.52080, "DK": 22.10999, "d1": -1.1267, "d2": -0.14854},
    },
}


def _watson_vector(species: str, piece_name: str) -> np.ndarray:
    fchk = parse_gaussian_fchk_harmonic_data(Path(f"{species}.fchk"))
    anh = parse_gaussian_anharmonic_force_data(Path(f"{species}.log"))
    quart = parse_gaussian_quartic_benchmark(Path(f"{species}.log"))
    model, _ = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
    mu1 = sp.MutableDenseNDimArray(model.dInv_au.tolist())
    omega = tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm)
    pieces = channel_h30h30_decomposed(
        mu1,
        sp.MutableDenseNDimArray(cubic.raw_au.tolist()),
        omega,
        hbar=sp.Float(1.0),
    )
    wat = to_watson_khz(
        pieces[piece_name],
        abc_mhz=model.abc_mhz,
        reduction="S",
        spectroscopic_axes=quart.spectroscopic_axes,
        tau_cm_scale=1.0,
    )
    return np.array([float(wat[key]) for key in WATSON_KEYS], dtype=float)


def main() -> None:
    for species, refs in VCI_PAPER.items():
        delta = np.array([refs["PFIT_RVCI_E"][k] - refs["VPT"][k] for k in WATSON_KEYS], dtype=float)
        trusted = _watson_vector(species, "trusted_total")
        pivot = _watson_vector(species, "manual_iij_pivot_candidate")
        residual = delta - trusted
        cos = float(np.dot(residual, pivot) / max(np.linalg.norm(residual) * np.linalg.norm(pivot), 1.0e-30))
        alpha = float(np.dot(pivot, residual) / max(np.dot(pivot, pivot), 1.0e-30))
        captured = abs(alpha) * np.linalg.norm(pivot) / max(np.linalg.norm(residual), 1.0e-30)
        print(
            f"{species:>4}  cos(res,pivot)={cos: .6e}  "
            f"|alpha|*||pivot||/||res||={captured: .6e}  "
            f"alpha={alpha: .6e}"
        )


if __name__ == "__main__":
    main()
