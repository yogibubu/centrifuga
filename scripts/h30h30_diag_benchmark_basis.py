#!/usr/bin/env python3
"""Build a minimal benchmark-derived basis for the one-mode diagonal H30H30 sector."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.h30h30_diagonal_subspace_probe import _collapsed_diag_vector


def _full_vector(seed: int) -> np.ndarray:
    return _collapsed_diag_vector(1, seed, diag_rot_only=False)


def _diag_vector(seed: int) -> np.ndarray:
    return _collapsed_diag_vector(1, seed, diag_rot_only=True)


def build_one_mode_benchmark_basis(
    *,
    seed_reference: int = 7,
    residual_seeds: tuple[int, int] = (11, 13),
) -> tuple[np.ndarray, list[str]]:
    """Return the minimal benchmark-derived basis currently identified.

    The basis is:
    - the diagonal-only core from the reference seed;
    - the rotational-mixing correction at the same seed;
    - the residual left on each additional benchmark seed after projection onto
      the first two vectors.
    """
    diag_ref = _diag_vector(seed_reference)
    rotmix_ref = _full_vector(seed_reference) - diag_ref
    basis = [diag_ref, rotmix_ref]
    labels = [f"diag(seed={seed_reference})", f"rotmix(seed={seed_reference})"]
    for seed in residual_seeds:
        full = _full_vector(seed)
        current_basis = np.column_stack(basis)
        coeffs, *_ = np.linalg.lstsq(current_basis, full, rcond=None)
        residual = full - current_basis @ coeffs
        basis.append(residual)
        labels.append(f"residual(seed={seed})")
    return np.column_stack(basis), labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-reference", type=int, default=7)
    ap.add_argument("--residual-seeds", nargs=2, type=int, default=(11, 13))
    ap.add_argument("--check-seeds", nargs="+", type=int, default=(7, 11, 13))
    args = ap.parse_args()

    basis, labels = build_one_mode_benchmark_basis(
        seed_reference=args.seed_reference,
        residual_seeds=tuple(args.residual_seeds),
    )
    print("One-mode H30H30 benchmark-derived diagonal basis")
    print(f"labels = {labels}")
    print(f"basis rank = {np.linalg.matrix_rank(basis)}")
    print("columns:")
    for label, col in zip(labels, basis.T):
        print(f"  {label}: {col}")
    print()
    for seed in args.check_seeds:
        full = _full_vector(seed)
        coeffs, *_ = np.linalg.lstsq(basis, full, rcond=None)
        residual = full - basis @ coeffs
        print(
            f"seed={seed}  coeffs={coeffs}  "
            f"||residual||={np.linalg.norm(residual): .6e}"
        )


if __name__ == "__main__":
    main()
