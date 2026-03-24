#!/usr/bin/env python3
"""Compare the appendix-style S3 block against the current solver S3 block."""

from __future__ import annotations

import sys
from pathlib import Path

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
from quartic_channels import channel_h12h30_decomposed
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")
SPECIES = ("hfo", "h2co", "h2cs")


def _watson_vec(tau, *, abc_mhz, axes):
    wat = to_watson_khz(
        tau,
        abc_mhz=abc_mhz,
        reduction="S",
        spectroscopic_axes=axes,
        tau_cm_scale=1.0,
    )
    return np.array([wat[key] for key in WATSON_KEYS], dtype=float)


def main() -> None:
    print("H12H30 S3 appendix-vs-solver audit")
    for species in SPECIES:
        fchk = parse_gaussian_fchk_harmonic_data(Path(f"{species}.fchk"))
        anh = parse_gaussian_anharmonic_force_data(Path(f"{species}.log"))
        quart = parse_gaussian_quartic_benchmark(Path(f"{species}.log"))
        model, _meta = build_default_harmonic_model(
            fchk.masses_amu,
            fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
            fchk.cartesian_force_constants,
        )
        cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
        mu1 = sp.MutableDenseNDimArray(model.dInv_au.tolist())
        mu2 = sp.MutableDenseNDimArray(model.d2Inv_au.tolist())
        omega = tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm)
        pieces = channel_h12h30_decomposed(
            mu1,
            mu2,
            sp.MutableDenseNDimArray(cubic.reduced_cm.tolist()),
            omega,
            hbar=sp.Float(1.0),
        )
        appendix = _watson_vec(pieces["S3_pair"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes)
        solver = _watson_vec(pieces["S3"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes)
        addition = _watson_vec(pieces["S3_addition"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes)
        split = _watson_vec(pieces["S3_split"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes)
        print(f"\n{species}")
        print("  S3_appendix =", appendix)
        print("  S3_addition =", addition)
        print("  S3_solver   =", solver)
        print("  S3_split    =", split)
        print(f"  ||appendix|| = {np.linalg.norm(appendix):.6e}")
        print(f"  ||addition|| = {np.linalg.norm(addition):.6e}")
        print(f"  ||solver||   = {np.linalg.norm(solver):.6e}")
        print(f"  ||split||    = {np.linalg.norm(split):.6e}")
        if np.linalg.norm(appendix) and np.linalg.norm(solver):
            corr = float(np.dot(appendix, solver) / (np.linalg.norm(appendix) * np.linalg.norm(solver)))
            print(f"  corr(appendix, solver) = {corr:.6e}")
        if np.linalg.norm(appendix) and np.linalg.norm(addition):
            corr_add = float(np.dot(appendix, addition) / (np.linalg.norm(appendix) * np.linalg.norm(addition)))
            print(f"  corr(appendix, addition) = {corr_add:.6e}")


if __name__ == "__main__":
    main()
