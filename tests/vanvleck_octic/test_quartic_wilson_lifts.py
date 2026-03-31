#!/usr/bin/env python3
"""Check full Wilson-tensor lifts for the quartic channels."""

from __future__ import annotations

import numpy as np

from compare_gaussian_sextic import WILSON_TAU_AU_TO_CMINV, _quartic_tau_from_model
from distortion_workflow import _sympy_tensors_from_model
from quartic_channels import (
    channel_h12h12,
    channel_h12h12_wilson,
    channel_h22,
    channel_h22_wilson,
    compress_wilson_tau,
)
from rovib_distortion import harmonic_inertia_model_from_geometry_hessian
from gaussian_vpt_parser import parse_gaussian_fchk_harmonic_data


def _float_tau4(tau4) -> np.ndarray:
    out = np.zeros((3, 3, 3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    out[a, b, c, d] = float(tau4[(a, b, c, d)])
    return out


def main() -> None:
    fchk = parse_gaussian_fchk_harmonic_data("h2o.fchk")
    model = harmonic_inertia_model_from_geometry_hessian(
        fchk.masses_amu,
        fchk.coords_bohr * 0.529177210903,
        fchk.cartesian_force_constants,
        representation="I",
    )

    mu1, mu2, _mu2_b, _mu2_n, _inertia0, omega_au, hbar = _sympy_tensors_from_model(model)

    tau_std_model, _taup = _quartic_tau_from_model(model)
    tau_std_full = WILSON_TAU_AU_TO_CMINV * _float_tau4(channel_h12h12_wilson(mu1, omega_au))
    tau_h22_full = channel_h22_wilson(mu2, omega_au, hbar)

    # The Wilson lift must project back exactly to the existing compressed channels.
    tau_h22_compressed = {k: float(v) for k, v in channel_h22(mu2, omega_au, hbar).items()}
    tau_h22_projected = {k: float(v) for k, v in compress_wilson_tau(tau_h22_full).items()}
    max_h22_proj_err = max(abs(tau_h22_compressed[k] - tau_h22_projected[k]) for k in tau_h22_compressed)

    # For the standard quartic sector, the Wilson lift should match the direct
    # sextic-side quartic tensor used in the Gaussian-style machinery.
    max_std_err = float(np.max(np.abs(tau_std_model - tau_std_full)))

    print(f"max |H12H12 Wilson - sextic tau_model| = {max_std_err:.3e}")
    print(f"max |compress(H22 Wilson) - H22 compressed| = {max_h22_proj_err:.3e}")
    if max_h22_proj_err > 1.0e-10:
        raise SystemExit(1)
    if max_std_err > 1.0e-8:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
