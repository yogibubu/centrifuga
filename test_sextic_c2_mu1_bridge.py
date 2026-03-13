#!/usr/bin/env python3
"""Check the local sextic bridge c1 ~ s_i mu1 on the H2O model."""

from __future__ import annotations

import numpy as np

from compare_gaussian_sextic import (
    _c1_matrix,
    _c2_modepair_from_mu1_scaled,
    _c2_modepair_tensor,
    _didq_from_model,
    _modewise_proportionality_scalars,
    _mu1_mode_matrices,
    _representation_axis_values,
    _zeta_xyz,
)
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

    didq = _didq_from_model(model)
    freq_cm = np.abs(model.vib_freq_cm)
    pmom, rot_cm = _representation_axis_values(model)
    zeta = _zeta_xyz(model)
    c1 = _c1_matrix(didq, freq_cm, pmom)
    mu1 = _mu1_mode_matrices(model)
    scales = _modewise_proportionality_scalars(c1, mu1)

    c1_err = float(np.max(np.abs(c1 - scales[:, None, None] * mu1)))
    pair_direct = _c2_modepair_tensor(c1, zeta, freq_cm, rot_cm)
    pair_mu1 = _c2_modepair_from_mu1_scaled(mu1, scales, zeta, freq_cm, rot_cm)
    pair_err = float(np.max(np.abs(pair_direct - pair_mu1)))

    print(f"max |c1 - s_i mu1| on h2o = {c1_err:.3e}")
    print(f"max |c2_pair(c1) - c2_pair(mu1-scaled)| on h2o = {pair_err:.3e}")
    if c1_err > 1.0e-12 or pair_err > 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
