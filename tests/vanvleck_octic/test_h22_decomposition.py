#!/usr/bin/env python3
"""Sanity check for the bilinear/intrinsic H22 decomposition."""

from __future__ import annotations

import numpy as np
import sympy as sp

from quartic_channels import bilinear_modepair_from_mu1, channel_h22, channel_h22_decomposed, channel_h22_from_mu1_intrinsic


def _to_sympy_tensor(arr: np.ndarray) -> sp.MutableDenseNDimArray:
    flat = [sp.Float(x) for x in arr.reshape(-1)]
    return sp.MutableDenseNDimArray(flat, arr.shape)


def main() -> None:
    rng = np.random.default_rng(7)
    n_modes = 2
    mu1_np = rng.normal(size=(3, 3, n_modes))
    inertia0_np = rng.normal(size=(3, 3))
    inertia0_np = 0.5 * (inertia0_np + inertia0_np.T)
    bilinear_np = rng.normal(size=(3, 3, n_modes, n_modes))
    intrinsic_np = rng.normal(size=(3, 3, n_modes, n_modes))
    total_np = bilinear_np - intrinsic_np

    mu1 = _to_sympy_tensor(mu1_np)
    inertia0 = _to_sympy_tensor(inertia0_np)
    bilinear = _to_sympy_tensor(bilinear_np)
    intrinsic = _to_sympy_tensor(intrinsic_np)
    total = _to_sympy_tensor(total_np)
    omega = tuple(sp.Float(x) for x in (1.2, 1.7))
    hbar = sp.Integer(1)

    direct = channel_h22(total, omega, hbar)
    pieces = channel_h22_decomposed(bilinear, intrinsic, omega, hbar)

    max_err = 0.0
    for key in direct:
        err = abs(float(sp.N(direct[key] - pieces["total"][key])))
        max_err = max(max_err, err)

    bilinear_from_mu1_np = np.zeros_like(bilinear_np)
    for k in range(n_modes):
        for l in range(n_modes):
            bilinear_from_mu1_np[:, :, k, l] = (
                mu1_np[:, :, k] @ inertia0_np @ mu1_np[:, :, l] + mu1_np[:, :, l] @ inertia0_np @ mu1_np[:, :, k]
            )
    bilinear_from_mu1 = bilinear_modepair_from_mu1(mu1, inertia0)
    bilinear_mu1_err = 0.0
    for idx, val in np.ndenumerate(bilinear_from_mu1_np):
        bilinear_mu1_err = max(bilinear_mu1_err, abs(float(sp.N(bilinear_from_mu1[idx] - val))))

    pieces_from_primitives = channel_h22_from_mu1_intrinsic(mu1, intrinsic, inertia0, omega, hbar)
    primitive_err = 0.0
    for key in pieces_from_primitives["total"]:
        primitive_err = max(
            primitive_err,
            abs(float(sp.N(pieces_from_primitives["total"][key] - channel_h22(_to_sympy_tensor(bilinear_from_mu1_np - intrinsic_np), omega, hbar)[key]))),
        )

    print(f"max |direct H22(mu2) - decomposed total| = {max_err:.3e}")
    print(f"max |bilinear_modepair_from_mu1 - numeric bilinear| = {bilinear_mu1_err:.3e}")
    print(f"max |H22(mu1,N) - H22(B(mu1)-N)| = {primitive_err:.3e}")
    if max_err > 1.0e-10 or bilinear_mu1_err > 1.0e-10 or primitive_err > 1.0e-10:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
