#!/usr/bin/env python3
"""CeDiTT4-style tensorial octic projections for symmetric-top and linear limits."""

from __future__ import annotations

import sympy as sp

from scripts.derive_octic_operator_basis import extract_asymmetric_top_cartesian_coefficients

Jx, Jy, Jz = sp.symbols("Jx Jy Jz")


def axial_octic_basis(axis: str = "a") -> tuple[sp.Expr, ...]:
    """Return the CeDiTT4-style axial octic operator basis.

    The basis is written directly in terms of J_perp^2 and J_parallel^2,
    exactly analogously to the quartic and sextic appendices of CeDiTT4.
    """
    axis = axis.lower()
    if axis == "a":
        Jpar2 = Jx**2
        Jperp2 = Jy**2 + Jz**2
    elif axis == "b":
        Jpar2 = Jy**2
        Jperp2 = Jx**2 + Jz**2
    elif axis == "c":
        Jpar2 = Jz**2
        Jperp2 = Jx**2 + Jy**2
    else:
        raise ValueError("axis must be one of 'a', 'b', 'c'")

    return (
        sp.expand(Jperp2**4),
        sp.expand(Jperp2**3 * Jpar2),
        sp.expand(Jperp2**2 * Jpar2**2),
        sp.expand(Jperp2 * Jpar2**3),
        sp.expand(Jpar2**4),
    )


def axial_octic_labels() -> tuple[str, ...]:
    return ("X0", "X1", "X2", "X3", "X4")


def axial_octic_from_cartesian_matrix(axis: str = "a") -> sp.Matrix:
    """Return the exact 5x15 left inverse cartesian -> axial coefficients."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis case is derived explicitly.")
    M = axial_octic_embedding_matrix(axis)
    return sp.simplify((M.T * M).inv() * M.T)


def axial_octic_embedding_matrix(axis: str = "a") -> sp.Matrix:
    """Return the exact 15x5 embedding of the axial image in cartesian space."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis case is derived explicitly.")
    basis = axial_octic_basis(axis)
    cols = []
    labels = (
        "c_008",
        "c_026",
        "c_044",
        "c_062",
        "c_080",
        "c_206",
        "c_224",
        "c_242",
        "c_260",
        "c_404",
        "c_422",
        "c_440",
        "c_602",
        "c_620",
        "c_800",
    )
    for expr in basis:
        coeffs = extract_asymmetric_top_cartesian_coefficients(expr)
        cols.append(sp.Matrix([coeffs[label] for label in labels]))
    return sp.Matrix.hstack(*cols)


def linear_from_axial_vector() -> sp.Matrix:
    """Return the exact axial->linear collapse vector.

    In the linear limit J_parallel = 0 and only X0 survives.
    """
    return sp.Matrix([[1, 0, 0, 0, 0]])


def octic_tensorial_projection_summary(axis: str = "a") -> dict[str, object]:
    M = axial_octic_embedding_matrix(axis)
    P = axial_octic_from_cartesian_matrix(axis)
    return {
        "axial_labels": axial_octic_labels(),
        "axial_basis": axial_octic_basis(axis),
        "embedding_15x5": M,
        "projection_5x15": P,
        "rank": int(M.rank()),
        "left_inverse_exact": sp.simplify(P * M - sp.eye(5)) == sp.zeros(5),
        "linear_from_axial": linear_from_axial_vector(),
    }


if __name__ == "__main__":
    print(octic_tensorial_projection_summary())
