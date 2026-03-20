#!/usr/bin/env python3
"""Extract a leading carbonyl-like post-iij residual basis."""

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
    "h2co": {"DJ": 68.746 - 69.622, "DJK": 1.28671 - 1.31233, "DK": 18.39056 - 18.54006, "d1": -10.1007 - (-9.8149), "d2": -2.4484 - (-2.1420)},
    "h2cs": {"DJ": 18.75614 - 18.66391, "DJK": 0.51221 - 0.52080, "DK": 22.34638 - 22.10999, "d1": -1.1693 - (-1.1267), "d2": -0.17351 - (-0.14854)},
}

BASIS = (
    "diag_bench_resid11_candidate",
    "diag_bench_resid13_candidate",
    "waterlike_basis1_candidate",
    "waterlike_basis2_candidate",
    "waterlike_scaffold_basis1_candidate",
    "waterlike_scaffold_basis2_candidate",
)


def _piece_vectors(species: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    basis = {name: vec(name) for name in BASIS}
    return post, basis


def main() -> None:
    res_cols = []
    basis_ref = None
    for species in ("h2co", "h2cs"):
        post, basis = _piece_vectors(species)
        res_cols.append(post)
        basis_ref = basis
        print(f"{species.upper()} post-iij residual = {post}")
    mat = np.column_stack(res_cols)
    u, s, _vh = np.linalg.svd(mat, full_matrices=False)
    lead = u[:, 0]
    print()
    print("carbonyl-like post-iij residual basis")
    print("rank =", np.linalg.matrix_rank(mat))
    print("singular values =", s)
    print("leading Watson-space vector =", lead)
    if basis_ref is not None:
        basis_mat = np.column_stack([basis_ref[name] for name in BASIS])
        coeffs, *_ = np.linalg.lstsq(basis_mat, lead, rcond=None)
        print("leading basis coefficients =", dict(zip(BASIS, coeffs)))


if __name__ == "__main__":
    main()
