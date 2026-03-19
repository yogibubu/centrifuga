#!/usr/bin/env python3
"""Compare the current diag_1 placeholder against the iii-only rotmix candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.h30h30_diagonal_subspace_probe import _collapsed_diag_vector, MON_KEYS
from quartic_channels import _H30H30_DIAG1_III_III_CORR_COEFFS


def _placeholder_vector() -> np.ndarray:
    return np.array([float(sp.N(_H30H30_DIAG1_III_III_CORR_COEFFS[key])) for key in MON_KEYS], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", nargs="+", type=int, default=(1, 2))
    ap.add_argument("--seeds", nargs="+", type=int, default=(7, 11, 13))
    args = ap.parse_args()

    v1 = _placeholder_vector()
    print("Current diag_1 placeholder coefficients")
    print(dict(zip(MON_KEYS, v1)))
    print(f"||diag1|| = {np.linalg.norm(v1): .6e}")
    print()

    for n_modes in args.n_modes:
        for seed in args.seeds:
            diag = _collapsed_diag_vector(n_modes, seed, diag_rot_only=True)
            full = _collapsed_diag_vector(n_modes, seed, diag_rot_only=False)
            rotmix = full - diag
            if np.linalg.norm(rotmix) == 0.0:
                corr = 0.0
            else:
                corr = float(np.dot(v1, rotmix) / (np.linalg.norm(v1) * np.linalg.norm(rotmix)))
            print(f"n_modes={n_modes} seed={seed}")
            print(f"  rotmix  = {dict(zip(MON_KEYS, rotmix))}")
            print(f"  corr(diag1, rotmix) = {corr: .6e}")
            print()


if __name__ == "__main__":
    main()
