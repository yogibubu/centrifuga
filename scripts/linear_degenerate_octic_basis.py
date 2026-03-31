#!/usr/bin/env python3
"""Exact linear degenerate operator hierarchy for the octic pairwise branch."""

from __future__ import annotations

import sympy as sp


def linear_degenerate_octic_basis() -> dict[str, object]:
    Xl = sp.Symbol("X_l", commutative=False)
    J2 = sp.Symbol("J^2", commutative=False)

    basis = (
        Xl,
        J2 * Xl,
        J2**2 * Xl,
    )
    labels = ("q_e", "q_J", "q_H")
    return {
        "labels": labels,
        "basis": basis,
        "effective_model": "H_l^(ij) = q_e X_l + q_J J^2 X_l + q_H (J^2)^2 X_l",
        "statement": (
            "For the exact linear degenerate pairwise branch, the observable "
            "operator hierarchy is three-dimensional: X_l, J^2 X_l, and "
            "(J^2)^2 X_l. The octic observable is q_H."
        ),
    }


if __name__ == "__main__":
    print(linear_degenerate_octic_basis())
