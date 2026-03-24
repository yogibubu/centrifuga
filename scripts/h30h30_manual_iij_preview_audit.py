#!/usr/bin/env python3
"""Audit the new manual iij pivot against the H30H30 VCI sanity targets."""

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

COMPARE_NAMES = (
    "trusted_total",
    "manual_iij_pivot_candidate",
    "manual_iij_preview_total",
    "diag_1_iii_iij_0",
    "diag_0_iij_iij_1_candidate",
    "waterlike_basis1_candidate",
    "waterlike_basis2_candidate",
)


def _watson_piece(species: str, piece_name: str) -> dict[str, float]:
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
    return {key: float(wat[key]) for key in WATSON_KEYS}


def _fmt(values: dict[str, float]) -> str:
    return ", ".join(f"{k}={values[k]: .6g}" for k in WATSON_KEYS)


def main() -> None:
    for species, refs in VCI_PAPER.items():
        delta = {k: refs["PFIT_RVCI_E"][k] - refs["VPT"][k] for k in WATSON_KEYS}
        delta_vec = np.array([delta[k] for k in WATSON_KEYS], dtype=float)
        print(species.upper())
        print(f"  target delta : {_fmt(delta)}")
        for name in COMPARE_NAMES:
            values = _watson_piece(species, name)
            vec = np.array([values[k] for k in WATSON_KEYS], dtype=float)
            ratio = np.linalg.norm(vec) / max(np.linalg.norm(delta_vec), 1.0e-30)
            resid = np.linalg.norm(delta_vec - vec) / max(np.linalg.norm(delta_vec), 1.0e-30)
            print(f"  {name:>24} : ||piece||/||delta||={ratio:.6g}  ||delta-piece||/||delta||={resid:.6g}")
        print()


if __name__ == "__main__":
    main()
