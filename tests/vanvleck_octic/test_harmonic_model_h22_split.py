#!/usr/bin/env python3
"""Check that the harmonic model exposes the H22 split consistently."""

from __future__ import annotations

import numpy as np

from gaussian_vpt_parser import parse_gaussian_fchk_harmonic_data
from rovib_distortion import harmonic_inertia_model_from_geometry_hessian


def main() -> None:
    fchk = parse_gaussian_fchk_harmonic_data("h2o.fchk")
    model = harmonic_inertia_model_from_geometry_hessian(
        fchk.masses_amu,
        fchk.coords_bohr * 0.529177210903,
        fchk.cartesian_force_constants,
        representation="I",
    )

    recon_err = float(
        np.max(np.abs(model.d2Inv_bilinear_au - model.d2Inv_intrinsic_au - model.d2Inv_au))
    )
    print(f"max |B - N - d2Inv| on harmonic model = {recon_err:.3e}")
    if recon_err > 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
