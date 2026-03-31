#!/usr/bin/env python3
"""CeDiTT4-style axial projection of the derived octic cartesian coefficients."""

from __future__ import annotations

import sympy as sp

from scripts.octic_tensorial_projection import (
    axial_octic_from_cartesian_matrix,
    axial_octic_labels,
    linear_from_axial_vector,
)
from scripts.paper1_h08_orth_projector import degree8_orth_basis
from scripts.paper1_octic_final_orth_coefficients import paper1_octic_final_orth_coefficients


def _orth_to_cartesian_vector(coeffs: dict[tuple[int, int, int], sp.Expr]) -> sp.Matrix:
    """Return the 15-vector in the stable cartesian order c_008..c_800."""
    basis = degree8_orth_basis()
    return sp.Matrix([coeffs[mon] for mon in basis])


def paper1_octic_axial_projection() -> dict[str, object]:
    """Project the derived 15 orthorhombic octic coefficients to X0..X4 and L."""
    total = paper1_octic_final_orth_coefficients()["total_orth_coeffs"]
    vec15 = _orth_to_cartesian_vector(total)
    P = axial_octic_from_cartesian_matrix("a")
    xvec = sp.simplify(P * vec15)
    xlabels = axial_octic_labels()
    axial = {xlabel: sp.expand(xvec[i]) for i, xlabel in enumerate(xlabels)}
    L = sp.expand((linear_from_axial_vector() * xvec)[0])
    return {
        "cartesian_vector_15": vec15,
        "axial_projection_5x15": P,
        "axial_coefficients": axial,
        "linear_constant_L": L,
        "statement": (
            "Following CeDiTT4, the octic branch is first projected onto the "
            "axial symmetric-top coefficients X0..X4; the linear limit keeps "
            "only X0, so L = X0."
        ),
    }


if __name__ == "__main__":
    out = paper1_octic_axial_projection()
    print(out["axial_coefficients"])
    print("L =", out["linear_constant_L"])
