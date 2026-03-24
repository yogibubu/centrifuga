#!/usr/bin/env python3
"""Audit the appendix-style H12H30 basis built from S1/S2/S3."""

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


NOTE_PATH = REPO_ROOT / "paper2_quartic_benchmark_note.md"
WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")
FIT_SPECIES = ("h2o", "h2s", "h2co", "h2cs")
REPORT_SPECIES = ("h2o", "h2s", "h2co", "h2cs", "hfo")
BASIS = ("S1", "S2", "S3")


def parse_h12h30_working_reference(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    out: dict[str, dict[str, float]] = {}
    for block in re.split(r"^##\s+", text, flags=re.MULTILINE)[1:]:
        species = block.strip().splitlines()[0].strip().lower()
        match = re.search(r"^H12H30\s*=\s*(\{.*\})$", block, flags=re.MULTILINE)
        if match:
            out[species] = ast.literal_eval(match.group(1))
    return out


def _load_h12h30_vectors(species: str) -> dict[str, np.ndarray]:
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
    for name in ("S1", "S2", "S3", "S3_addition", "S3_split", "S3_pair", "S3_wpair"):
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
    ap.add_argument("--show-s3-split", action="store_true", help="Also report the norm ratio ||S3_split|| / ||S3|| for species where the three-mode block is active.")
    args = ap.parse_args()

    refs = parse_h12h30_working_reference(NOTE_PATH)
    rows = []
    targets = []
    for species in FIT_SPECIES:
        vecs = _load_h12h30_vectors(species)
        target = np.array([refs[species][key] for key in WATSON_KEYS], dtype=float)
        for idx in range(len(WATSON_KEYS)):
            rows.append([vecs[name][idx] for name in BASIS])
            targets.append(target[idx])

    amat = np.asarray(rows, dtype=float)
    bvec = np.asarray(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(amat, bvec, rcond=None)

    print("H12H30 appendix-basis audit")
    print("basis =", ", ".join(BASIS))
    print("fit species =", ", ".join(FIT_SPECIES))
    print("least-squares coefficients:")
    for name, coeff in zip(BASIS, coeffs):
        print(f"  {name:<3} {coeff: .8e}")

    print("\nPer-species reconstructed vectors")
    for species in REPORT_SPECIES:
        if not Path(f"{species}.fchk").exists() or not Path(f"{species}.log").exists():
            continue
        vecs = _load_h12h30_vectors(species)
        pred = np.zeros(len(WATSON_KEYS), dtype=float)
        for coeff, name in zip(coeffs, BASIS):
            pred += coeff * vecs[name]
        print(species)
        print("  pred =", pred)
        if species in refs:
            ref = np.array([refs[species][key] for key in WATSON_KEYS], dtype=float)
            print("  ref  =", ref)
            print("  max|delta| =", np.max(np.abs(pred - ref)))
        if args.show_s3_split:
            s3_norm = np.linalg.norm(vecs["S3"])
            add_norm = np.linalg.norm(vecs["S3_addition"])
            split_norm = np.linalg.norm(vecs["S3_split"])
            ratio = split_norm / s3_norm if s3_norm else float("inf")
            add_ratio = add_norm / s3_norm if s3_norm else float("inf")
            print(f"  ||S3_addition|| / ||S3|| = {add_ratio:.6e}")
            print(f"  ||S3_split|| / ||S3|| = {ratio:.6e}")
            norm_pair = np.linalg.norm(vecs["S3_pair"])
            norm_wpair = np.linalg.norm(vecs["S3_wpair"])
            if norm_pair and norm_wpair:
                pair_corr = float(np.dot(vecs["S3_pair"], vecs["S3_wpair"]) / (norm_pair * norm_wpair))
                print(f"  corr(S3_pair, S3_wpair) = {pair_corr:.6e}")
            else:
                print("  corr(S3_pair, S3_wpair) = n/a")


if __name__ == "__main__":
    main()
