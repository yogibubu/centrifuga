#!/usr/bin/env python3
"""Probe the residual left by the {diag, rotmix} basis in the iii-only sector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.h30h30_diagonal_subspace_probe import _collapsed_diag_vector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-reference", type=int, default=7)
    ap.add_argument("--seeds", nargs="+", type=int, default=(7, 11, 13))
    args = ap.parse_args()

    rotmix_ref = _collapsed_diag_vector(1, args.seed_reference, diag_rot_only=False) - _collapsed_diag_vector(1, args.seed_reference, diag_rot_only=True)
    rows = []
    for seed in args.seeds:
        diag = _collapsed_diag_vector(1, seed, diag_rot_only=True)
        full = _collapsed_diag_vector(1, seed, diag_rot_only=False)
        basis = np.column_stack([diag, rotmix_ref])
        coeffs, *_ = np.linalg.lstsq(basis, full, rcond=None)
        residual = full - basis @ coeffs
        rows.append(residual)
        print(f"seed={seed}  coeffs={coeffs}  ||residual||={np.linalg.norm(residual): .6e}  residual={residual}")
    print(f"residual-span rank = {np.linalg.matrix_rank(np.array(rows))}")


if __name__ == "__main__":
    main()
