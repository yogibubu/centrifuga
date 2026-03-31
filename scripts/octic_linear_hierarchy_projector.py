#!/usr/bin/env python3
"""Exact scalar-hierarchy projector for the linear octic CeDiTT4 basis."""

from __future__ import annotations

import sympy as sp


def octic_linear_hierarchy_projector() -> dict[str, object]:
    X = sp.Symbol("X")
    basis = (
        X**4 + 4 * X**3 - 14 * X**2 + 12 * X,  # octic X0 image
        X**3,  # sextic scalar
        X**2,  # quartic scalar
        X,  # quadratic scalar
        sp.Integer(1),  # constant
    )
    M = sp.Matrix([[sp.expand(b).coeff(X, i) for b in basis] for i in range(5)])
    P = sp.simplify(M.inv())
    return {
        "basis": basis,
        "coefficient_matrix": M,
        "projector": P,
        "left_inverse_exact": sp.simplify(P * M - sp.eye(5)) == sp.zeros(5),
    }


if __name__ == "__main__":
    print(octic_linear_hierarchy_projector())
