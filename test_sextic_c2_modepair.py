#!/usr/bin/env python3
"""Check that the unsummed sextic mode-pair tensor reconstructs c2."""

from __future__ import annotations

import numpy as np

from compare_gaussian_sextic import _c2_modepair_tensor, _c2_tensor


def main() -> None:
    rng = np.random.default_rng(11)
    n_modes = 3
    c1 = rng.normal(size=(n_modes, 3, 3))
    zeta = rng.normal(size=(3, n_modes, n_modes))
    freq_cm = np.array([100.0, 250.0, 400.0], dtype=float)
    rot_cm = np.array([1.0, 2.0, 3.0], dtype=float)

    pair = _c2_modepair_tensor(c1, zeta, freq_cm, rot_cm)
    c2 = _c2_tensor(c1, zeta, freq_cm, rot_cm)
    err = float(np.max(np.abs(np.sum(pair, axis=1) - c2)))
    cyc_err = float(np.max(np.abs(pair - np.transpose(pair, (0, 1, 3, 4, 2)))))
    print(f"max |sum_j c2_pair - c2| = {err:.3e}")
    print(f"max |c2_pair(a,b,c) - c2_pair(b,c,a)| = {cyc_err:.3e}")
    if err > 1.0e-12 or cyc_err > 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
