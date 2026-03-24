#!/usr/bin/env python3
"""Scan rotated internal bases for the appendix S3 block of H12H30."""

from __future__ import annotations

import argparse
import math
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
DEFAULT_SPECIES = ("hfo", "h2co", "h2cs")


def _watson_vec(tau, *, abc_mhz, axes):
    wat = to_watson_khz(
        tau,
        abc_mhz=abc_mhz,
        reduction="S",
        spectroscopic_axes=axes,
        tau_cm_scale=1.0,
    )
    return np.array([wat[key] for key in WATSON_KEYS], dtype=float)


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
    return {
        "S1": _watson_vec(pieces["S1"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes),
        "S2": _watson_vec(pieces["S2"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes),
        "A": _watson_vec(pieces["tri_ijk_over_pair_sums"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes),
        "B": _watson_vec(pieces["tri_ijk_over_w_pair_sums"], abc_mhz=model.abc_mhz, axes=quart.spectroscopic_axes),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", nargs="*", default=list(DEFAULT_SPECIES))
    ap.add_argument("--steps", type=int, default=181, help="Number of angles in [0, pi).")
    args = ap.parse_args()

    cases = {species: _load_case(species) for species in args.species}
    best = None
    best_payload = None

    print("H12H30 S3 rotation scan")
    print("species =", ", ".join(args.species))
    for idx in range(args.steps):
        theta = math.pi * idx / max(args.steps - 1, 1)
        c = math.cos(theta)
        s = math.sin(theta)
        cols = []
        max_corr = 0.0
        split_ratio_sum = 0.0
        for species in args.species:
            data = cases[species]
            t_eff = c * data["A"] + s * data["B"]
            t_res = -s * data["A"] + c * data["B"]
            cols.extend([data["S1"], data["S2"], t_eff, t_res])
            split_ratio_sum += np.linalg.norm(t_res) / max(np.linalg.norm(t_eff), 1.0e-30)
            max_corr = max(
                max_corr,
                abs(_corr(data["S1"], t_eff)),
                abs(_corr(data["S2"], t_eff)),
                abs(_corr(data["S1"], t_res)),
                abs(_corr(data["S2"], t_res)),
                abs(_corr(t_eff, t_res)),
            )
        mat = np.column_stack(cols)
        singular = np.linalg.svd(mat, compute_uv=False)
        payload = {
            "theta_deg": 180.0 * theta / math.pi,
            "cond_proxy": singular[0] / max(singular[-1], 1.0e-30),
            "min_sv": singular[-1],
            "max_corr": max_corr,
            "avg_split_ratio": split_ratio_sum / len(args.species),
        }
        key = (payload["max_corr"], -payload["min_sv"], payload["avg_split_ratio"])
        if best is None or key < best:
            best = key
            best_payload = payload

    assert best_payload is not None
    print("best rotation:")
    print(f"  theta_deg       = {best_payload['theta_deg']:.3f}")
    print(f"  max_corr        = {best_payload['max_corr']:.6f}")
    print(f"  min_singular    = {best_payload['min_sv']:.6e}")
    print(f"  cond_proxy      = {best_payload['cond_proxy']:.6e}")
    print(f"  avg_split_ratio = {best_payload['avg_split_ratio']:.6e}")


if __name__ == "__main__":
    main()
