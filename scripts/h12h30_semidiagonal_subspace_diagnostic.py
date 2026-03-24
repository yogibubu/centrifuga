#!/usr/bin/env python3
"""Diagnose conditioning of the semi-diagonal H12H30 scaffold subspace."""

from __future__ import annotations

import argparse
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
from quartic_channels import channel_h12h30_decomposed
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")
SEMI_RAW_BASIS = (
    "diag_iii_over_w5",
    "sd_iij_over_w2_w_wp",
    "sd_iij_over_w_wp_2wipj",
    "sd_iij_over_w_wp_wi2wj",
)
SEMI_REDUCED_BASIS = (
    "diag_iii_over_w5",
    "sd_iij_over_w2_w_wp",
    "semidiagonal_effective",
    "semidiagonal_split",
)
SEMI_MINIMAL_BASIS = (
    "diag_iii_over_w5",
    "semidiagonal_effective",
    "semidiagonal_split",
)
DEFAULT_SPECIES = ("h2o", "h2s")
NOTE_PATH = REPO_ROOT / "paper2_quartic_benchmark_note.md"


def parse_h12h30_working_reference(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    out: dict[str, dict[str, float]] = {}
    for block in re.split(r"^##\s+", text, flags=re.MULTILINE)[1:]:
        species = block.strip().splitlines()[0].strip().lower()
        match = re.search(r"^H12H30\s*=\s*(\{.*\})$", block, flags=re.MULTILINE)
        if match:
            out[species] = ast.literal_eval(match.group(1))
    return out


def _watson_vector(tau: dict[str, sp.Expr], *, abc_mhz, spectroscopic_axes):
    wat = to_watson_khz(
        tau,
        abc_mhz=abc_mhz,
        reduction="S",
        spectroscopic_axes=spectroscopic_axes,
        tau_cm_scale=1.0,
    )
    return np.array([wat[key] for key in WATSON_KEYS], dtype=float)


def _load_case(species: str):
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
    raw_terms = {
        "diag_iii_over_w5": pieces["diag_iii_over_w5"],
        "sd_iij_over_w2_w_wp": pieces["sd_iij_over_w2_w_wp"],
        "sd_iij_over_w_wp_2wipj": pieces["sd_iij_over_w_wp_2wipj"],
        "sd_iij_over_w_wp_wi2wj": pieces["sd_iij_over_w_wp_wi2wj"],
    }
    reduced_terms = {
        "diag_iii_over_w5": pieces["diag_iii_over_w5"],
        "sd_iij_over_w2_w_wp": pieces["sd_iij_over_w2_w_wp"],
        "semidiagonal_effective": pieces["semidiagonal_effective"],
        "semidiagonal_split": pieces["semidiagonal_split"],
    }
    raw_vecs = {
        name: _watson_vector(raw_terms[name], abc_mhz=model.abc_mhz, spectroscopic_axes=quart.spectroscopic_axes)
        for name in SEMI_RAW_BASIS
    }
    reduced_vecs = {
        name: _watson_vector(reduced_terms[name], abc_mhz=model.abc_mhz, spectroscopic_axes=quart.spectroscopic_axes)
        for name in SEMI_REDUCED_BASIS
    }
    return raw_vecs, reduced_vecs


def _analyze_basis(
    *,
    species_list: list[str],
    basis_names: tuple[str, ...],
    case_vectors: dict[str, dict[str, np.ndarray]],
    targets_by_species: dict[str, np.ndarray],
    corr_threshold: float,
) -> None:
    columns = []
    labels = []
    rows = []
    targets = []
    for species in species_list:
        target = targets_by_species[species]
        vecs = case_vectors[species]
        for name in basis_names:
            columns.append(vecs[name])
            labels.append(f"{species}:{name}")
        for idx in range(len(WATSON_KEYS)):
            rows.append([vecs[name][idx] for name in basis_names])
            targets.append(target[idx])

    mat = np.column_stack(columns)
    amat = np.asarray(rows, dtype=float)
    bvec = np.asarray(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(amat, bvec, rcond=None)
    _, singular_values, _ = np.linalg.svd(mat, full_matrices=False)

    print("basis   =", ", ".join(basis_names))
    print(f"rank    = {np.linalg.matrix_rank(mat)} / {mat.shape[1]}")
    print("singular values =", " ".join(f"{val:.6e}" for val in singular_values))
    print("least-squares coefficients against working reference:")
    for name, coeff in zip(basis_names, coeffs):
        print(f"  {name:<28} {coeff: .8e}")

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
            if abs(corr) >= corr_threshold:
                printed = True
                print(f"  {left:<36} {right:<36} corr={corr:+.6f}")
    if not printed:
        print("  none above threshold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="*", default=list(DEFAULT_SPECIES), help="Species to include in the semi-diagonal audit.")
    ap.add_argument("--corr-threshold", type=float, default=0.995, help="Print scaffold-pair correlations whose absolute value exceeds this threshold.")
    args = ap.parse_args()

    ref = parse_h12h30_working_reference(NOTE_PATH)
    raw_case_vectors: dict[str, dict[str, np.ndarray]] = {}
    reduced_case_vectors: dict[str, dict[str, np.ndarray]] = {}
    targets_by_species: dict[str, np.ndarray] = {}
    for species in args.species:
        raw_vecs, reduced_vecs = _load_case(species)
        raw_case_vectors[species] = raw_vecs
        reduced_case_vectors[species] = reduced_vecs
        targets_by_species[species] = np.array([ref[species][key] for key in WATSON_KEYS], dtype=float)

    print("H12H30 semi-diagonal subspace diagnostic")
    print("species =", ", ".join(args.species))
    print("\nRaw semi-diagonal basis")
    _analyze_basis(
        species_list=args.species,
        basis_names=SEMI_RAW_BASIS,
        case_vectors=raw_case_vectors,
        targets_by_species=targets_by_species,
        corr_threshold=args.corr_threshold,
    )
    print("\nReduced semi-diagonal basis")
    _analyze_basis(
        species_list=args.species,
        basis_names=SEMI_REDUCED_BASIS,
        case_vectors=reduced_case_vectors,
        targets_by_species=targets_by_species,
        corr_threshold=args.corr_threshold,
    )
    print("\nMinimal semi-diagonal basis")
    _analyze_basis(
        species_list=args.species,
        basis_names=SEMI_MINIMAL_BASIS,
        case_vectors=reduced_case_vectors,
        targets_by_species=targets_by_species,
        corr_threshold=args.corr_threshold,
    )


if __name__ == "__main__":
    main()
