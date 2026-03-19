#!/usr/bin/env python3
"""Compress the water-like scaffold basis into iii,iij and iij,iij reduced pivots."""

from __future__ import annotations

import numpy as np


FOLLOWUP_CANDIDATES = (
    "diag_1_iii_iij_0",
    "diag_0_iii_iij_1_candidate",
    "diag_0_iii_iij_2_resonance_candidate",
    "diag_0_iii_iij_2_regularized_preview",
    "diag_0_iij_iij_1_candidate",
    "diag_0_iij_iij_2_candidate",
)

SCAFFOLD_BASIS_1 = np.array([
    0.0016723056589263618,
    0.9610237103797781,
    -0.2283854556552517,
    -0.102107640379879,
    -0.10475133627416916,
    0.05359013390205153,
], dtype=float)
SCAFFOLD_BASIS_2 = np.array([
    0.012700918862961983,
    -0.03900293399591244,
    0.03189795415480179,
    0.00939775421472365,
    -0.7648149932117112,
    -0.6420823052341297,
], dtype=float)


def main() -> None:
    iii_iij_mask = np.array([1, 1, 1, 1, 0, 0], dtype=float)
    iij_iij_mask = np.array([0, 0, 0, 0, 1, 1], dtype=float)

    reduced_iii_iij = SCAFFOLD_BASIS_1 * iii_iij_mask
    reduced_iij_iij = SCAFFOLD_BASIS_2 * iij_iij_mask

    if np.linalg.norm(reduced_iii_iij) > 0:
        reduced_iii_iij /= np.linalg.norm(reduced_iii_iij)
    if np.linalg.norm(reduced_iij_iij) > 0:
        reduced_iij_iij /= np.linalg.norm(reduced_iij_iij)

    print("Reduced H30H30 follow-up basis")
    print("iii,iij-dominant reduced pivot:")
    print(dict(zip(FOLLOWUP_CANDIDATES, reduced_iii_iij)))
    print()
    print("iij,iij-dominant reduced pivot:")
    print(dict(zip(FOLLOWUP_CANDIDATES, reduced_iij_iij)))


if __name__ == "__main__":
    main()
