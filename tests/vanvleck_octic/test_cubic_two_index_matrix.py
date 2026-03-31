#!/usr/bin/env python3
"""Check 2-index cubic matrix expansion against the semi-diagonal tensor sector."""

from __future__ import annotations

import numpy as np

from gaussian_vpt_parser import align_gaussian_cubic_force_constants, parse_gaussian_anharmonic_force_data
from compare_gaussian_sextic import _aligned_model_from_gaussian, expand_cubic_two_index_matrix, split_cubic_force_constants


def main() -> None:
    model, _rep, _order, _signs, _didq_err = _aligned_model_from_gaussian("h2o.fchk", "h2o.log")
    anh = parse_gaussian_anharmonic_force_data("h2o.log")
    _mapping, phi3_full, _raw = align_gaussian_cubic_force_constants(anh, abs(model.vib_freq_cm))
    phi3_sd, _phi3_3 = split_cubic_force_constants(phi3_full)

    n = phi3_full.shape[0]
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = phi3_sd[i, i, j]
    rebuilt = expand_cubic_two_index_matrix(mat)
    err = float(np.max(np.abs(rebuilt - phi3_sd)))
    print(f"max |expanded 2-index cubic - semi-diagonal cubic| = {err:.3e}")
    if err > 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
