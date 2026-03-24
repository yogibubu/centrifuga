#!/usr/bin/env python3
"""Fit scalar weights for the current H30H30 scaffold-first reconstruction."""

from __future__ import annotations

import ast
import re
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
from quartic_channels import channel_h30h30_decomposed
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


NOTE_PATH = REPO_ROOT / "paper2_quartic_benchmark_note.md"
SPECIES = ("H2O", "H2S", "H2CO", "H2CS")
WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")
SCAFFOLDS = (
    "diag_0_iii_iii",
    "diag_1_iii_iij_0",
    "diag_0_iii_iij_1_candidate",
    "diag_0_iij_iij_1_candidate",
    "diag_0_iij_iij_2_candidate",
)


def parse_benchmark_note(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    text = path.read_text()
    species_blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for block in species_blocks[1:]:
        lines = block.strip().splitlines()
        species = lines[0].strip()
        match = re.search(r"^H30H30\s*=\s*(\{.*\})$", block, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing H30H30 entry for {species}")
        out[species] = {"H30H30": ast.literal_eval(match.group(1))}
    return out


def scaffold_vectors(species: str) -> dict[str, np.ndarray]:
    base = REPO_ROOT / species.lower()
    fchk = parse_gaussian_fchk_harmonic_data(base.with_suffix(".fchk"))
    anh = parse_gaussian_anharmonic_force_data(base.with_suffix(".log"))
    quart = parse_gaussian_quartic_benchmark(base.with_suffix(".log"))
    model, _ = build_default_harmonic_model(
        fchk.masses_amu,
        fchk.coords_bohr * (1.0 / ANGSTROM_TO_BOHR),
        fchk.cartesian_force_constants,
    )
    cubic = align_cubic_to_harmonic_model(anh, model.vib_freq_cm)
    mu1 = sp.MutableDenseNDimArray(model.dInv_au.tolist())
    omega = tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm)
    pieces = channel_h30h30_decomposed(mu1, sp.MutableDenseNDimArray(cubic.raw_au.tolist()), omega, hbar=sp.Float(1.0))
    vecs = {}
    for name in SCAFFOLDS:
        wat = to_watson_khz(
            pieces[name],
            abc_mhz=model.abc_mhz,
            reduction="S",
            spectroscopic_axes=quart.spectroscopic_axes,
            tau_cm_scale=1.0,
        )
        vecs[name] = np.array([float(wat[key]) for key in WATSON_KEYS], dtype=float)
    return vecs


def main() -> None:
    benchmark = parse_benchmark_note(NOTE_PATH)
    rows = []
    rhs = []
    for species in SPECIES:
        vecs = scaffold_vectors(species)
        ref = benchmark[species]["H30H30"]
        for idx, key in enumerate(WATSON_KEYS):
            rows.append([vecs[name][idx] for name in SCAFFOLDS])
            rhs.append(ref[key])
    A = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)
    coeffs, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)

    print("H30H30 scaffold scalar fit")
    print("Scaffolds:", ", ".join(SCAFFOLDS))
    print("coefficients =", dict(zip(SCAFFOLDS, coeffs)))
    print("rank =", rank)
    print("singular values =", singular)
    if residuals.size:
        print("residual sumsq =", float(residuals[0]))

    for species in SPECIES:
        vecs = scaffold_vectors(species)
        fitted = sum(coeffs[i] * vecs[name] for i, name in enumerate(SCAFFOLDS))
        ref = np.array([benchmark[species]["H30H30"][key] for key in WATSON_KEYS], dtype=float)
        delta = fitted - ref
        print(f"\n{species}")
        print("  fitted =", dict(zip(WATSON_KEYS, fitted)))
        print("  ref    =", dict(zip(WATSON_KEYS, ref)))
        print("  delta  =", dict(zip(WATSON_KEYS, delta)))
        print("  max|delta| =", float(np.max(np.abs(delta))))


if __name__ == "__main__":
    main()
