#!/usr/bin/env python3
"""Summarize the diagonal balance of the current H30H30 placeholder channel."""

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
from quartic_channels import (
    _H30H30_DIAG1_COEFFS,
    _H30H30_DIAG1_III_III_CORR_COEFFS,
    channel_h30h30_decomposed,
)
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")


def _build_piece(species: str) -> tuple[dict[str, dict[str, sp.Expr]], tuple[float, float, float], str]:
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
    return pieces, tuple(model.abc_mhz), quart.spectroscopic_axes


def _watson_vec(piece: dict[str, sp.Expr], abc_mhz: tuple[float, float, float], axes: str) -> np.ndarray:
    wat = to_watson_khz(
        piece,
        abc_mhz=abc_mhz,
        reduction="S",
        spectroscopic_axes=axes,
        tau_cm_scale=1.0,
    )
    return np.array([float(wat[key]) for key in WATSON_KEYS], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="+", default=("h2o", "h2s", "h2co", "h2cs"))
    args = ap.parse_args()

    print("Componentwise D1/D0 coefficient ratios")
    for key in ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz"):
        ratio = float(sp.N(_H30H30_DIAG1_III_III_CORR_COEFFS[key] / _H30H30_DIAG1_COEFFS[key]))
        print(f"  {key:>8}: {ratio: .6e}")

    for species in args.species:
        pieces, abc_mhz, axes = _build_piece(species)
        pure = _watson_vec(pieces["pure_diagonal_total"], abc_mhz, axes)
        corr = _watson_vec(pieces["diag_1_iii_iii_correction_candidate"], abc_mhz, axes)
        trusted = _watson_vec(pieces["trusted_total"], abc_mhz, axes)
        pure_norm = max(np.linalg.norm(pure), 1.0e-30)
        print(
            f"{species.upper():>5}  "
            f"||D1||/||D0||={np.linalg.norm(corr)/pure_norm: .6e}  "
            f"||D0+D1||/||D0||={np.linalg.norm(trusted)/pure_norm: .6e}"
        )


if __name__ == "__main__":
    main()
