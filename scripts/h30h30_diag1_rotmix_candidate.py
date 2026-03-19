#!/usr/bin/env python3
"""Extract the rotational-mixing correction of the iii-only H30H30 core."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.h30h30_diagonal_subspace_probe import _collapsed_diag_vector, MON_KEYS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-modes", nargs="+", type=int, default=(1, 2))
    ap.add_argument("--seeds", nargs="+", type=int, default=(7, 11, 13))
    args = ap.parse_args()

    print("iii-only H30H30 rotational-mixing candidate")
    for n_modes in args.n_modes:
        for seed in args.seeds:
            diag = _collapsed_diag_vector(n_modes, seed, diag_rot_only=True)
            full = _collapsed_diag_vector(n_modes, seed, diag_rot_only=False)
            delta = full - diag
            print(f"\nn_modes={n_modes} seed={seed}")
            print("  diag-only =", dict(zip(MON_KEYS, diag)))
            print("  full      =", dict(zip(MON_KEYS, full)))
            print("  delta     =", dict(zip(MON_KEYS, delta)))
            print(f"  ||delta|| / ||diag|| = {np.linalg.norm(delta) / max(np.linalg.norm(diag), 1.0e-30): .6e}")


if __name__ == "__main__":
    main()
