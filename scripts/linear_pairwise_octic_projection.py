#!/usr/bin/env python3
"""Exact projection algebra for the linear pairwise octic branch."""

from __future__ import annotations

import sympy as sp


def linear_pairwise_octic_projection() -> dict[str, object]:
    J2 = sp.Symbol("J^2", commutative=False)
    Xl = sp.Symbol("X_l", commutative=False)

    basis = (
        Xl,
        J2 * Xl,
        J2**2 * Xl,
    )
    labels = ("q_e", "q_J", "q_H")
    coeffs = sp.symbols("q_e q_J q_H")
    operator = sp.expand(coeffs[0] * basis[0] + coeffs[1] * basis[1] + coeffs[2] * basis[2])
    return {
        "labels": labels,
        "basis": basis,
        "generic_operator": operator,
        "qH_projector": sp.Symbol("extract coefficient of (J^2)^2 X_l"),
        "statement": (
            "The exact linear pairwise octic branch is three-dimensional. "
            "Its octic observable q_H is the coefficient of (J^2)^2 X_l."
        ),
    }


if __name__ == "__main__":
    print(linear_pairwise_octic_projection())
