#!/usr/bin/env python3
"""Report the current H30H30 scaffold split in Watson S-reduced constants."""

from __future__ import annotations

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="+", default=("h2o", "h2s"))
    args = ap.parse_args()

    for species in args.species:
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
        print(f"\n{species.upper()}")
        for name in ("diag_1_iii_iii", "diag_1_iii_iij_0", "diag_0_iii_iij_1_candidate", "placeholder_residual", "total"):
            wat = to_watson_khz(
                pieces[name],
                abc_mhz=model.abc_mhz,
                reduction="S",
                spectroscopic_axes=quart.spectroscopic_axes,
                tau_cm_scale=1.0,
            )
            vec = np.array([float(wat[key]) for key in WATSON_KEYS], dtype=float)
            print(f"{name:>26}  norm={np.linalg.norm(vec): .6e}  values={dict(zip(WATSON_KEYS, vec))}")


if __name__ == "__main__":
    main()
