#!/usr/bin/env python3
"""Diagnose the active H12H30 subspace when the three-index sector is present."""

from __future__ import annotations

import argparse
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
ACTIVE_BASIS = (
    "diag_iii_over_w5",
    "semidiagonal_effective",
    "semidiagonal_split",
    "three_index_effective",
    "three_index_split",
)
DEFAULT_SPECIES = ("hfo", "h2co", "h2cs")


def _load_case(species: str) -> dict[str, np.ndarray]:
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
    out: dict[str, np.ndarray] = {}
    for name in ACTIVE_BASIS:
        wat = to_watson_khz(
            pieces[name],
            abc_mhz=model.abc_mhz,
            reduction="S",
            spectroscopic_axes=quart.spectroscopic_axes,
            tau_cm_scale=1.0,
        )
        out[name] = np.array([wat[key] for key in WATSON_KEYS], dtype=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="*", default=list(DEFAULT_SPECIES))
    ap.add_argument("--corr-threshold", type=float, default=0.995)
    args = ap.parse_args()

    columns = []
    labels = []
    print("H12H30 active subspace diagnostic")
    print("species =", ", ".join(args.species))
    print("basis   =", ", ".join(ACTIVE_BASIS))
    for species in args.species:
        vecs = _load_case(species)
        print(f"\n{species}")
        for name in ACTIVE_BASIS:
            vec = vecs[name]
            columns.append(vec)
            labels.append(f"{species}:{name}")
            print(f"  {name:<22} norm={np.linalg.norm(vec):.6e} vec={vec}")

    mat = np.column_stack(columns)
    _, singular_values, _ = np.linalg.svd(mat, full_matrices=False)
    print(f"\nrank = {np.linalg.matrix_rank(mat)} / {mat.shape[1]}")
    print("singular values =", " ".join(f"{val:.6e}" for val in singular_values))
    print("high-correlation scaffold pairs:")
    printed = False
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if j <= i:
                continue
            a = mat[:, i]
            b = mat[:, j]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            corr = float(np.dot(a, b) / norm) if norm else 0.0
            if abs(corr) >= args.corr_threshold:
                printed = True
                print(f"  {left:<34} {right:<34} corr={corr:+.6f}")
    if not printed:
        print("  none above threshold")


if __name__ == "__main__":
    main()
