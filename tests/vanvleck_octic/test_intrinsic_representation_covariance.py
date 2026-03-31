#!/usr/bin/env python3
"""Check representation covariance of mu1/B/N/mu2 on the H2O harmonic model."""

from __future__ import annotations

import numpy as np

from gaussian_vpt_parser import parse_gaussian_fchk_harmonic_data
from rovib_distortion import harmonic_inertia_model_from_geometry_hessian, representation_component_permutation


def _permute_mu1(arr: np.ndarray, q: np.ndarray) -> np.ndarray:
    return arr[q][:, q, :]


def _permute_modepair(arr: np.ndarray, q: np.ndarray, signs: np.ndarray) -> np.ndarray:
    out = arr[q][:, q, :, :]
    return out * signs[None, None, :, None] * signs[None, None, None, :]


def main() -> None:
    fchk = parse_gaussian_fchk_harmonic_data("h2o.fchk")
    models = {
        rep: harmonic_inertia_model_from_geometry_hessian(
            fchk.masses_amu,
            fchk.coords_bohr * 0.529177210903,
            fchk.cartesian_force_constants,
            representation=rep,
        )
        for rep in ("I", "II", "III")
    }

    src = models["I"]
    mu1_src = np.asarray(src.dInv_au, dtype=float)
    n_modes = mu1_src.shape[2]

    worst = 0.0
    for rep in ("II", "III"):
        tgt = models[rep]
        q = representation_component_permutation("I", rep)
        mu1_perm = _permute_mu1(mu1_src, q)
        signs = np.ones(n_modes, dtype=float)
        for k in range(n_modes):
            dot = float(np.sum(mu1_perm[:, :, k] * tgt.dInv_au[:, :, k]))
            signs[k] = 1.0 if dot >= 0.0 else -1.0

        mu1_err = float(np.max(np.abs(mu1_perm * signs[None, None, :] - tgt.dInv_au)))
        b_err = float(
            np.max(
                np.abs(
                    _permute_modepair(np.asarray(src.d2Inv_bilinear_au, dtype=float), q, signs)
                    - tgt.d2Inv_bilinear_au
                )
            )
        )
        n_err = float(
            np.max(
                np.abs(
                    _permute_modepair(np.asarray(src.d2Inv_intrinsic_au, dtype=float), q, signs)
                    - tgt.d2Inv_intrinsic_au
                )
            )
        )
        t_err = float(
            np.max(
                np.abs(
                    _permute_modepair(np.asarray(src.d2Inv_au, dtype=float), q, signs)
                    - tgt.d2Inv_au
                )
            )
        )
        worst = max(worst, mu1_err, b_err, n_err, t_err)
        print(f"{rep}: max |mu1 covariance error| = {mu1_err:.3e}")
        print(f"{rep}: max |B covariance error| = {b_err:.3e}")
        print(f"{rep}: max |N covariance error| = {n_err:.3e}")
        print(f"{rep}: max |mu2 covariance error| = {t_err:.3e}")

    if worst > 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
