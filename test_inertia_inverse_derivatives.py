#!/usr/bin/env python3
"""Finite-difference check for analytic inverse-inertia derivatives."""

from __future__ import annotations

import numpy as np

from rovib_distortion import (
    inverse_inertia_bilinear_from_mu1,
    inertia_derivative_tensors,
    inverse_inertia_derivatives,
    inverse_inertia_second_derivative_components,
)


def inertia_tensor_generic(masses: np.ndarray, coords: np.ndarray) -> np.ndarray:
    eye3 = np.eye(3)
    out = np.zeros((3, 3), dtype=float)
    for m, r in zip(masses, coords):
        out += float(m) * (float(np.dot(r, r)) * eye3 - np.outer(r, r))
    return out


def inverse_inertia_at_q(masses: np.ndarray, coords: np.ndarray, modes: np.ndarray, q: np.ndarray) -> np.ndarray:
    disp = np.tensordot(modes.reshape(coords.shape[0], 3, -1), q, axes=(2, 0))
    return np.linalg.inv(inertia_tensor_generic(masses, coords + disp))


def main() -> None:
    masses = np.array([10.0, 12.0, 14.0], dtype=float)
    coords = np.array(
        [
            [0.0, 0.1, 0.3],
            [1.2, -0.4, 0.7],
            [-0.9, 0.8, -0.2],
        ],
        dtype=float,
    )
    modes = np.array(
        [
            [0.10, 0.05],
            [0.20, -0.03],
            [-0.10, 0.04],
            [0.00, 0.07],
            [0.05, 0.02],
            [0.06, -0.01],
            [-0.03, 0.02],
            [0.04, 0.09],
            [0.02, -0.08],
        ],
        dtype=float,
    )

    I0, dI, d2I = inertia_derivative_tensors(masses, coords, modes)
    invI0, dInv, d2Inv = inverse_inertia_derivatives(I0, dI, d2I)
    invI_split, bilinear, intrinsic, d2Inv_split = inverse_inertia_second_derivative_components(I0, dI, d2I)
    bilinear_mu1 = inverse_inertia_bilinear_from_mu1(I0, dInv)

    eps = 1.0e-4
    q0 = np.zeros(modes.shape[1], dtype=float)

    split_err = float(np.max(np.abs(invI_split - invI0)))
    total_err = float(np.max(np.abs(d2Inv_split - d2Inv)))
    recon_err = float(np.max(np.abs((bilinear - intrinsic) - d2Inv)))
    bilinear_mu1_err = float(np.max(np.abs(bilinear_mu1 - bilinear)))

    max_first = 0.0
    for k in range(modes.shape[1]):
        qp = q0.copy()
        qm = q0.copy()
        qp[k] += eps
        qm[k] -= eps
        fd = (inverse_inertia_at_q(masses, coords, modes, qp) - inverse_inertia_at_q(masses, coords, modes, qm)) / (2 * eps)
        max_first = max(max_first, float(np.max(np.abs(fd - dInv[:, :, k]))))

    max_second = 0.0
    for k in range(modes.shape[1]):
        for l in range(modes.shape[1]):
            qpp = q0.copy()
            qpm = q0.copy()
            qmp = q0.copy()
            qmm = q0.copy()
            qpp[k] += eps
            qpp[l] += eps
            qpm[k] += eps
            qpm[l] -= eps
            qmp[k] -= eps
            qmp[l] += eps
            qmm[k] -= eps
            qmm[l] -= eps
            fd = (
                inverse_inertia_at_q(masses, coords, modes, qpp)
                - inverse_inertia_at_q(masses, coords, modes, qpm)
                - inverse_inertia_at_q(masses, coords, modes, qmp)
                + inverse_inertia_at_q(masses, coords, modes, qmm)
            ) / (4 * eps * eps)
            max_second = max(max_second, float(np.max(np.abs(fd - d2Inv[:, :, k, l]))))

    print(f"max |analytic d(I^-1)/dQ - finite diff| = {max_first:.3e}")
    print(f"max |analytic d2(I^-1)/dQdQ - finite diff| = {max_second:.3e}")
    print(f"max |split invI - direct invI| = {split_err:.3e}")
    print(f"max |split total - direct d2(I^-1)| = {total_err:.3e}")
    print(f"max |(bilinear - intrinsic) - direct d2(I^-1)| = {recon_err:.3e}")
    print(f"max |bilinear(mu1) - bilinear(dI)| = {bilinear_mu1_err:.3e}")

    if (
        max_first > 1e-7
        or max_second > 1e-7
        or split_err > 1e-12
        or total_err > 1e-12
        or recon_err > 1e-12
        or bilinear_mu1_err > 1e-12
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
