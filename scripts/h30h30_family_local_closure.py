#!/usr/bin/env python3
"""Build family-local post-iij closure bases for H30H30."""

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

FAMILIES = {
    "water_like": ("h2o", "h2s"),
    "carbonyl_like": ("h2co", "h2cs"),
}


def _species_vectors(species: str) -> tuple[np.ndarray, np.ndarray]:
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

    delta = np.array([VCI_PAPER[species][k] for k in WATSON_KEYS], dtype=float)
    trusted = vec("trusted_total")
    iij = vec("manual_iij_pivot_candidate")
    residual = delta - trusted
    alpha = float(np.dot(iij, residual) / max(np.dot(iij, iij), 1.0e-30))
    post = residual - alpha * iij
    return residual, post


def main() -> None:
    print("H30H30 family-local closure after trusted_total + best iij")
    for family, species_list in FAMILIES.items():
        post_cols = []
        for species in species_list:
            _res, post = _species_vectors(species)
            post_cols.append(post)
        post_mat = np.column_stack(post_cols)
        u, s, _vh = np.linalg.svd(post_mat, full_matrices=False)
        rank = np.linalg.matrix_rank(post_mat)
        print(f"\n{family}: species={species_list}")
        print(f"  post-iij rank = {rank}")
        print(f"  singular values = {s}")
        family_basis = u[:, :rank]
        for species in species_list:
            residual, post = _species_vectors(species)
            coeffs, *_ = np.linalg.lstsq(family_basis, post, rcond=None)
            fit = family_basis @ coeffs
            print(f"  {species.upper()}: ||post-fit|| = {np.linalg.norm(post-fit):.6g}  coeffs={coeffs}")


if __name__ == "__main__":
    main()
