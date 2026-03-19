#!/usr/bin/env python3
"""Audit whether the H30H30 diagonal family can be repaired by global rescaling."""

from __future__ import annotations

import ast
import re
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


NOTE_PATH = REPO_ROOT / "paper2_quartic_benchmark_note.md"
SPECIES = ("H2O", "H2S", "H2CO", "H2CS")
WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")


def parse_benchmark_note(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    species_blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)
    out: dict[str, dict[str, float]] = {}
    for block in species_blocks[1:]:
        lines = block.strip().splitlines()
        species = lines[0].strip()
        match = re.search(r"^H30H30\s*=\s*(\{.*\})$", block, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing H30H30 entry for {species}")
        out[species] = ast.literal_eval(match.group(1))
    return out


def _diag_vectors(species: str) -> tuple[np.ndarray, np.ndarray]:
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
    pieces = channel_h30h30_decomposed(
        sp.MutableDenseNDimArray(model.dInv_au.tolist()),
        sp.MutableDenseNDimArray(cubic.raw_au.tolist()),
        tuple(abs(float(x)) / AU_FREQ_TO_CMINV for x in model.vib_freq_cm),
        hbar=sp.Float(1.0),
    )
    out = []
    for name in ("pure_diagonal_total", "diag_1_iii_iii_correction_candidate"):
        wat = to_watson_khz(
            pieces[name],
            abc_mhz=model.abc_mhz,
            reduction="S",
            spectroscopic_axes=quart.spectroscopic_axes,
            tau_cm_scale=1.0,
        )
        out.append(np.array([float(wat[key]) for key in WATSON_KEYS], dtype=float))
    return out[0], out[1]


def main() -> None:
    benchmark = parse_benchmark_note(NOTE_PATH)
    mon_keys = ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")
    v0 = np.array([float(sp.N(_H30H30_DIAG1_COEFFS[key])) for key in mon_keys], dtype=float)
    v1 = np.array([float(sp.N(_H30H30_DIAG1_III_III_CORR_COEFFS[key])) for key in mon_keys], dtype=float)
    corr = float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    rows = []
    rhs = []
    for species in SPECIES:
        d0, d1 = _diag_vectors(species)
        ref = np.array([benchmark[species][key] for key in WATSON_KEYS], dtype=float)
        for idx in range(len(WATSON_KEYS)):
            rows.append([d0[idx], d1[idx]])
            rhs.append(ref[idx])

    A = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)
    coeffs, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)

    print("Diagonal-only H30H30 fit audit")
    print(f"corr(D0,D1) = {corr: .6e}")
    print(f"alpha(D0) = {coeffs[0]: .6e}")
    print(f"beta(D1)  = {coeffs[1]: .6e}")
    print(f"rank      = {rank}")
    print(f"singular  = {singular}")
    if residuals.size:
        print(f"residual  = {float(residuals[0]): .6e}")
    print()

    for species in SPECIES:
        d0, d1 = _diag_vectors(species)
        fit = coeffs[0] * d0 + coeffs[1] * d1
        ref = np.array([benchmark[species][key] for key in WATSON_KEYS], dtype=float)
        delta = fit - ref
        print(
            f"{species:>4}  max|fit-ref|={np.max(np.abs(delta)): .6e}  "
            f"||fit-ref||={np.linalg.norm(delta): .6e}"
        )


if __name__ == "__main__":
    main()
