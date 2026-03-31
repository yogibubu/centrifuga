#!/usr/bin/env python3
"""Exact 15->9 octic asymmetric-top reduction in the Watson/Aliev standard basis."""

from __future__ import annotations

import sympy as sp

from scripts.derive_octic_operator_basis import (
    asymmetric_top_cartesian_polynomial,
    asymmetric_top_monomial_labels,
    extract_asymmetric_top_cartesian_coefficients,
)

Jx, Jy, Jz = sp.symbols("Jx Jy Jz")


def standard_octic_reduced_basis() -> tuple[sp.Expr, ...]:
    """Return the 9 reduced octic operators in commuting form.

    This is the standard asymmetric-top octic basis written with the special
    axis chosen as z, matching the Watson/Aliev style expansion:
      J^8, J^6 Jz^2, J^4 Jz^4, J^2 Jz^6, Jz^8,
      J^6 Q2, J^4 Jz^2 Q2, J^2 Jz^4 Q2, Jz^6 Q2
    where Q2 = J_+^2 + J_-^2 = 2(Jx^2 - Jy^2) in the commuting reduction.
    """

    J2 = Jx**2 + Jy**2 + Jz**2
    Q2 = 2 * (Jx**2 - Jy**2)
    return (
        sp.expand(J2**4),
        sp.expand(J2**3 * Jz**2),
        sp.expand(J2**2 * Jz**4),
        sp.expand(J2 * Jz**6),
        sp.expand(Jz**8),
        sp.expand(J2**3 * Q2),
        sp.expand(J2**2 * Jz**2 * Q2),
        sp.expand(J2 * Jz**4 * Q2),
        sp.expand(Jz**6 * Q2),
    )


def standard_octic_reduced_labels() -> tuple[str, ...]:
    return (
        "L800",
        "L620",
        "L440",
        "L260",
        "L080",
        "L602",
        "L422",
        "L242",
        "L062",
    )


def cartesian_from_reduced_matrix() -> sp.Matrix:
    """Return the exact 15x9 matrix reduced -> commuting cartesian."""
    labels = asymmetric_top_monomial_labels()
    cols = []
    for expr in standard_octic_reduced_basis():
        coeffs = extract_asymmetric_top_cartesian_coefficients(expr)
        cols.append(sp.Matrix([sp.expand(coeffs[label]) for label in labels]))
    return sp.Matrix.hstack(*cols)


def reduced_from_cartesian_pseudoinverse() -> sp.Matrix:
    """Return the Moore-Penrose left inverse of the 15x9 reduction map."""
    M = cartesian_from_reduced_matrix()
    return sp.simplify((M.T * M).inv() * M.T)


def reduced_symbols() -> sp.Matrix:
    return sp.Matrix([sp.Symbol(label) for label in standard_octic_reduced_labels()])


def reduced_to_cartesian_coefficients() -> dict[str, sp.Expr]:
    M = cartesian_from_reduced_matrix()
    vec = M * reduced_symbols()
    labels = asymmetric_top_monomial_labels()
    return {label: sp.expand(vec[i]) for i, label in enumerate(labels)}


def cartesian_to_reduced_coefficients() -> dict[str, sp.Expr]:
    P = reduced_from_cartesian_pseudoinverse()
    cart = extract_asymmetric_top_cartesian_coefficients(asymmetric_top_cartesian_polynomial())
    labels = asymmetric_top_monomial_labels()
    cvec = sp.Matrix([cart[label] for label in labels])
    out = P * cvec
    return {
        label: sp.expand(out[i]) for i, label in enumerate(standard_octic_reduced_labels())
    }


def octic_asymmetric_top_reduction_summary() -> dict[str, object]:
    M = cartesian_from_reduced_matrix()
    P = reduced_from_cartesian_pseudoinverse()
    return {
        "reduced_labels": standard_octic_reduced_labels(),
        "matrix_15x9": M,
        "left_inverse_9x15": P,
        "rank": int(M.rank()),
        "cartesian_from_reduced": reduced_to_cartesian_coefficients(),
        "reduced_from_cartesian": cartesian_to_reduced_coefficients(),
        "nullity": int(M.shape[0] - M.rank()),
    }


if __name__ == "__main__":
    out = octic_asymmetric_top_reduction_summary()
    print("rank =", out["rank"])
    print("nullity =", out["nullity"])
