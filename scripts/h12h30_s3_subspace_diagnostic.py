#!/usr/bin/env python3
"""Compare active H12H30 bases built around the appendix S1/S2/S3 notation."""

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
NOTE_PATH = REPO_ROOT / "paper2_quartic_benchmark_note.md"
DEFAULT_SPECIES = ("hfo", "h2co", "h2cs")
BASES = (
    ("S1", "S2", "S3"),
    ("S1", "S2", "S3_split"),
    ("S1", "S2", "S3", "S3_split"),
)


def parse_h12h30_working_reference(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    out: dict[str, dict[str, float]] = {}
    for block in re.split(r"^##\s+", text, flags=re.MULTILINE)[1:]:
        species = block.strip().splitlines()[0].strip().lower()
        match = re.search(r"^H12H30\s*=\s*(\{.*\})$", block, flags=re.MULTILINE)
        if match:
            out[species] = ast.literal_eval(match.group(1))
    return out


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
    vecs: dict[str, np.ndarray] = {}
    for name in ("S1", "S2", "S3", "S3_split"):
        wat = to_watson_khz(
            pieces[name],
            abc_mhz=model.abc_mhz,
            reduction="S",
            spectroscopic_axes=quart.spectroscopic_axes,
            tau_cm_scale=1.0,
        )
        vecs[name] = np.array([wat[key] for key in WATSON_KEYS], dtype=float)
    return vecs


def _analyze_basis(
    *,
    basis: tuple[str, ...],
    species_list: list[str],
    case_vectors: dict[str, dict[str, np.ndarray]],
    refs: dict[str, dict[str, float]],
    corr_threshold: float,
) -> None:
    columns = []
    labels = []
    rows = []
    targets = []
    fitted_species = [species for species in species_list if species in refs]
    for species in species_list:
        vecs = case_vectors[species]
        for name in basis:
            columns.append(vecs[name])
            labels.append(f"{species}:{name}")
    for species in fitted_species:
        vecs = case_vectors[species]
        target = np.array([refs[species][key] for key in WATSON_KEYS], dtype=float)
        for idx in range(len(WATSON_KEYS)):
            rows.append([vecs[name][idx] for name in basis])
            targets.append(target[idx])

    mat = np.column_stack(columns)
    _, singular_values, _ = np.linalg.svd(mat, full_matrices=False)

    print("basis   =", ", ".join(basis))
    print(f"rank    = {np.linalg.matrix_rank(mat)} / {mat.shape[1]}")
    print("singular values =", " ".join(f"{val:.6e}" for val in singular_values))
    if fitted_species:
        amat = np.asarray(rows, dtype=float)
        bvec = np.asarray(targets, dtype=float)
        coeffs, *_ = np.linalg.lstsq(amat, bvec, rcond=None)
        print("least-squares coefficients against working reference:")
        print("fit species =", ", ".join(fitted_species))
        for name, coeff in zip(basis, coeffs):
            print(f"  {name:<12} {coeff: .8e}")
    else:
        print("least-squares coefficients against working reference: none available")
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
                print(f"  {left:<26} {right:<26} corr={corr:+.6f}")
    if not printed:
        print("  none above threshold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="*", default=list(DEFAULT_SPECIES))
    ap.add_argument("--corr-threshold", type=float, default=0.995)
    args = ap.parse_args()

    refs = parse_h12h30_working_reference(NOTE_PATH)
    case_vectors: dict[str, dict[str, np.ndarray]] = {}
    for species in args.species:
        case_vectors[species] = _load_case(species)

    print("H12H30 S3 subspace diagnostic")
    print("species =", ", ".join(args.species))
    for basis in BASES:
        print()
        _analyze_basis(
            basis=basis,
            species_list=args.species,
            case_vectors=case_vectors,
            refs=refs,
            corr_threshold=args.corr_threshold,
        )


if __name__ == "__main__":
    main()
