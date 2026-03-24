#!/usr/bin/env python3
"""Compare the manual H30H30 iij pivot against existing follow-up candidates."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import sympy as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartic_channels import (
    _H30H30_DIAG1_III_IIJ0_COEFFS,
    _H30H30_DIAG0_III_IIJ1_COEFFS,
    _H30H30_DIAG0_III_IIJ2_COEFFS,
    _H30H30_IIJ_IIJ1_COEFFS,
    _H30H30_IIJ_IIJ2_COEFFS,
)
from scripts.h30h30_sector_symbolic_probe import _collapsed_sector_vector


MON_KEYS = ("tau_xxxx", "tau_yyyy", "tau_zzzz", "tau_xxyy", "tau_xxzz", "tau_yyzz")
ROT_PAIRS = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)}


def _coeff_vector(coeffs) -> np.ndarray:
    return np.array([float(sp.N(coeffs[key])) for key in MON_KEYS], dtype=float)


def main() -> None:
    iij = _collapsed_sector_vector(
        2,
        7,
        allowed_classes={"iij"},
        diag_rot_only=False,
        rot_pairs=ROT_PAIRS,
        manual=True,
    )
    candidates = [
        ("diag1_iii_iij0", _coeff_vector(_H30H30_DIAG1_III_IIJ0_COEFFS)),
        ("diag0_iii_iij1", _coeff_vector(_H30H30_DIAG0_III_IIJ1_COEFFS)),
        ("diag0_iii_iij2", _coeff_vector(_H30H30_DIAG0_III_IIJ2_COEFFS)),
        ("iij_iij1", _coeff_vector(_H30H30_IIJ_IIJ1_COEFFS)),
        ("iij_iij2", _coeff_vector(_H30H30_IIJ_IIJ2_COEFFS)),
    ]

    print("Manual H30H30 iij pivot audit")
    print("manual iij:", iij)
    for name, vec in candidates:
        corr = float(np.dot(iij, vec) / (np.linalg.norm(iij) * np.linalg.norm(vec)))
        scale = float(np.linalg.lstsq(vec.reshape(-1, 1), iij, rcond=None)[0][0])
        relres = float(np.linalg.norm(iij - scale * vec) / np.linalg.norm(iij))
        print(f"{name:>16}: corr={corr:+.6f}  scale={scale:+.6f}  relres={relres:.6f}")

    basis = np.column_stack([vec for _name, vec in candidates])
    coeffs, *_ = np.linalg.lstsq(basis, iij, rcond=None)
    relres = float(np.linalg.norm(iij - basis @ coeffs) / np.linalg.norm(iij))
    print(f"\nspan relres = {relres:.6f}")
    print("span coeffs =", coeffs)


if __name__ == "__main__":
    main()
