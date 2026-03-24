#!/usr/bin/env python3
"""Fit simple semi-diagonal H12H30 scaffolds to the Paper 2 benchmark note.

This is a diagnostic reconstruction tool. It does not define the production
formula in ``quartic_channels.py``. Its purpose is to test whether a small
basis built from semi-diagonal cubic constants ``phi_iij`` can reproduce the
observed mixed-channel scale before reintroducing a hard-coded implementation.
"""

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
from rovib_distortion import ANGSTROM_TO_BOHR, AU_FREQ_TO_CMINV


NOTE_PATH = REPO_ROOT / "paper2_quartic_benchmark_note.md"
WATSON_KEYS = ("DJ", "DJK", "DK", "d1", "d2")
TAU_KEYS = ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")


def parse_benchmark_note(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    species_blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)
    out: dict[str, dict[str, float]] = {}
    for block in species_blocks[1:]:
        lines = block.strip().splitlines()
        species = lines[0].strip()
        match = re.search(r"^H12H30\s*=\s*(\{.*\})$", block, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing H12H30 entry for {species}")
        out[species.lower()] = ast.literal_eval(match.group(1))
    return out


def _empty_tau():
    return {k: sp.Integer(0) for k in TAU_KEYS}


def _component_mix(mu1, mu2, i, j, key: str):
    if key == "tau_xxxx":
        return mu1[0, 0, i] * mu2[0, 0, j, j]
    if key == "tau_yyyy":
        return mu1[1, 1, i] * mu2[1, 1, j, j]
    if key == "tau_zzzz":
        return mu1[2, 2, i] * mu2[2, 2, j, j]
    if key == "tau_xxyy":
        return (
            mu1[0, 0, i] * mu2[1, 1, j, j]
            + mu1[0, 1, i] * mu2[0, 1, j, j]
            + mu1[0, 1, i] * mu2[1, 0, j, j]
            + mu1[1, 0, i] * mu2[0, 1, j, j]
            + mu1[1, 0, i] * mu2[1, 0, j, j]
            + mu1[1, 1, i] * mu2[0, 0, j, j]
        )
    if key == "tau_xxzz":
        return (
            mu1[0, 0, i] * mu2[2, 2, j, j]
            + mu1[0, 2, i] * mu2[0, 2, j, j]
            + mu1[0, 2, i] * mu2[2, 0, j, j]
            + mu1[2, 0, i] * mu2[0, 2, j, j]
            + mu1[2, 0, i] * mu2[2, 0, j, j]
            + mu1[2, 2, i] * mu2[0, 0, j, j]
        )
    if key == "tau_yyzz":
        return (
            mu1[1, 1, i] * mu2[2, 2, j, j]
            + mu1[1, 2, i] * mu2[1, 2, j, j]
            + mu1[1, 2, i] * mu2[2, 1, j, j]
            + mu1[2, 1, i] * mu2[1, 2, j, j]
            + mu1[2, 1, i] * mu2[2, 1, j, j]
            + mu1[2, 2, i] * mu2[1, 1, j, j]
        )
    raise KeyError(key)


def build_scaffold_terms(mu1, mu2, phi3_red, omega):
    n = len(omega)
    terms: dict[str, dict[str, sp.Expr]] = {}

    diag = _empty_tau()
    for i in range(n):
        denom = omega[i] ** 5
        phi = sp.Float(float(phi3_red[i, i, i]))
        for key in TAU_KEYS:
            diag[key] += phi * _component_mix(mu1, mu2, i, i, key) / denom
    terms["diag_iii_over_w5"] = diag

    pair_a = _empty_tau()
    pair_b = _empty_tau()
    pair_c = _empty_tau()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            phi = sp.Float(float(phi3_red[i, i, j]))
            if abs(float(phi)) <= 1.0e-14:
                continue
            den_a = omega[i] ** 2 * omega[j] * (omega[i] + omega[j])
            den_b = omega[i] * (omega[i] + omega[j]) * (2 * omega[i] + omega[j])
            den_c = omega[i] * (omega[i] + omega[j]) * (omega[i] + 2 * omega[j])
            for key in TAU_KEYS:
                mix = _component_mix(mu1, mu2, i, j, key)
                pair_a[key] += phi * mix / den_a
                pair_b[key] += phi * mix / den_b
                pair_c[key] += phi * mix / den_c
    terms["sd_iij_over_w2_w_wp"] = pair_a
    terms["sd_iij_over_w_wp_2wipj"] = pair_b
    terms["sd_iij_over_w_wp_wi2wj"] = pair_c

    tri_a = _empty_tau()
    tri_b = _empty_tau()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                phi = sp.Float(float(phi3_red[i, j, k]))
                if abs(float(phi)) <= 1.0e-14:
                    continue
                den_a = (omega[i] + omega[j]) * (omega[j] + omega[k]) * (omega[i] + omega[k])
                den_b = omega[i] * den_a
                for key in TAU_KEYS:
                    mix = _component_mix(mu1, mu2, i, j, key)
                    tri_a[key] += phi * mix / den_a
                    tri_b[key] += phi * mix / den_b
    terms["tri_ijk_over_pair_sums"] = tri_a
    terms["tri_ijk_over_w_pair_sums"] = tri_b
    return terms


def watson_vector(tau, abc_mhz, axes):
    wat = to_watson_khz(tau, abc_mhz=abc_mhz, reduction="S", spectroscopic_axes=axes, tau_cm_scale=1.0)
    return np.array([float(wat[k]) for k in WATSON_KEYS], dtype=float)


def main() -> None:
    bench = parse_benchmark_note(NOTE_PATH)
    species_list = ("h2o", "h2s", "h2co", "h2cs")

    rows = []
    targets = []
    basis_names = None

    for species in species_list:
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
        terms = build_scaffold_terms(mu1, mu2, cubic.reduced_cm, omega)
        if basis_names is None:
            basis_names = tuple(terms.keys())
        basis_vecs = [watson_vector(terms[name], model.abc_mhz, quart.spectroscopic_axes) for name in basis_names]
        target = np.array([bench[species][k] for k in WATSON_KEYS], dtype=float)
        for idx in range(len(WATSON_KEYS)):
            rows.append([vec[idx] for vec in basis_vecs])
            targets.append(target[idx])

    amat = np.asarray(rows, dtype=float)
    bvec = np.asarray(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(amat, bvec, rcond=None)

    print("Semi-diagonal H12H30 scaffold fit")
    for name, coeff in zip(basis_names or (), coeffs):
        print(f"  {name:<28} {coeff: .8e}")
    print()

    for species in species_list:
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
        terms = build_scaffold_terms(mu1, mu2, cubic.reduced_cm, omega)
        pred = np.zeros(len(WATSON_KEYS), dtype=float)
        for coeff, name in zip(coeffs, basis_names or ()):
            pred += coeff * watson_vector(terms[name], model.abc_mhz, quart.spectroscopic_axes)
        ref = np.array([bench[species][k] for k in WATSON_KEYS], dtype=float)
        print(species)
        for key, got, want in zip(WATSON_KEYS, pred, ref):
            print(f"  {key:<3} pred={got: .6e} ref={want: .6e} delta={got-want: .6e}")
        print(f"  max|delta| = {np.max(np.abs(pred-ref)): .6e}")

    extra_species = [name for name in ("hfo", "HFO") if Path(f"{name}.fchk").exists() and Path(f"{name}.log").exists()]
    if extra_species:
        print("\nExtra diagnostic species")
    for species in extra_species:
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
        terms = build_scaffold_terms(mu1, mu2, cubic.reduced_cm, omega)
        print(species)
        for name in basis_names or ():
            vec = watson_vector(terms[name], model.abc_mhz, quart.spectroscopic_axes)
            print(f"  {name:<28} " + " ".join(f"{k}={v: .4e}" for k, v in zip(WATSON_KEYS, vec)))


if __name__ == "__main__":
    main()
