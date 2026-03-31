#!/usr/bin/env python3
"""Exact principal-symbol projection for the non-linear optical sector.

At principal-symbol level, the degree-8 source is carried by:

1. the quartic-potential visible term of H08,
2. the six completion blocks of Eq. (99).

This module records the exact linear bookkeeping map from these seven blocks to
the 45 cartesian degree-8 coefficients, then to the 15 orthorhombic
coefficients, then to the 5 symmetric-top constants and finally to the linear
constant L.
"""

from __future__ import annotations

import sympy as sp

from scripts.derive_octic_operator_basis import (
    asymmetric_top_monomial_labels,
    project_asymmetric_cartesian_to_linear,
    project_asymmetric_cartesian_to_symmetric_top,
)
from scripts.paper1_h08_orth_projector import degree8_orth_basis


def paper1_octic_principal_term_labels() -> tuple[str, ...]:
    return (
        "H08_k4",
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
        "S07H02",
    )


def paper1_octic_principal_prefactors() -> dict[str, sp.Expr]:
    I = sp.I
    return {
        "H08_k4": sp.Integer(1),
        "S03S03S03H02": -I / 6,
        "S03S03H04": -sp.Rational(1, 2),
        "S03H06": I,
        "S05H04": I,
        "S05S03H02": -sp.Integer(1),
        "S07H02": I,
    }


def paper1_octic_principal_cartesian_source_coefficients() -> dict[str, dict[str, sp.Symbol]]:
    labels = asymmetric_top_monomial_labels()
    return {
        term: {label: sp.Symbol(f"{label}^{term}") for label in labels}
        for term in paper1_octic_principal_term_labels()
    }


def paper1_octic_principal_total_cartesian_coefficients() -> dict[str, sp.Expr]:
    src = paper1_octic_principal_cartesian_source_coefficients()
    pref = paper1_octic_principal_prefactors()
    return {
        label: sp.simplify(sum(pref[term] * src[term][label] for term in paper1_octic_principal_term_labels()))
        for label in asymmetric_top_monomial_labels()
    }


def paper1_octic_principal_orth_coefficients() -> dict[str, sp.Expr]:
    total = paper1_octic_principal_total_cartesian_coefficients()
    out: dict[str, sp.Expr] = {}
    for a, b, c in degree8_orth_basis():
        label = f"c_{a}{b}{c}"
        out[label] = total[label]
    return out


def paper1_octic_principal_symmetric_top_constants(axis: str = "a") -> dict[str, sp.Expr]:
    return project_asymmetric_cartesian_to_symmetric_top(paper1_octic_principal_total_cartesian_coefficients(), axis)


def paper1_octic_principal_linear_constant(axis: str = "a") -> sp.Expr:
    return project_asymmetric_cartesian_to_linear(paper1_octic_principal_total_cartesian_coefficients(), axis)


def paper1_octic_principal_termwise_linear_decomposition(axis: str = "a") -> dict[str, sp.Expr]:
    src = paper1_octic_principal_cartesian_source_coefficients()
    pref = paper1_octic_principal_prefactors()
    return {
        term: sp.simplify(pref[term] * project_asymmetric_cartesian_to_linear(src[term], axis))
        for term in paper1_octic_principal_term_labels()
    }


def paper1_octic_principal_reduced_term_labels() -> tuple[str, ...]:
    """Return the principal-symbol source blocks that survive in orth projection.

    The S07 block is used only to cancel the nonorthorhombic complement and has
    no direct contribution to the 15 orthorhombic coefficients at principal-symbol
    level.
    """
    return (
        "H08_k4",
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
    )


def paper1_octic_principal_reduced_orth_coefficients() -> dict[str, sp.Expr]:
    src = paper1_octic_principal_cartesian_source_coefficients()
    pref = paper1_octic_principal_prefactors()
    out: dict[str, sp.Expr] = {}
    for a, b, c in degree8_orth_basis():
        label = f"c_{a}{b}{c}"
        out[label] = sp.simplify(sum(pref[term] * src[term][label] for term in paper1_octic_principal_reduced_term_labels()))
    return out


def paper1_octic_principal_reduced_symmetric_top_constants(axis: str = "a") -> dict[str, sp.Expr]:
    return project_asymmetric_cartesian_to_symmetric_top(paper1_octic_principal_reduced_orth_coefficients(), axis)


def paper1_octic_principal_reduced_linear_constant(axis: str = "a") -> sp.Expr:
    return project_asymmetric_cartesian_to_linear(paper1_octic_principal_reduced_orth_coefficients(), axis)


def main() -> None:
    print(paper1_octic_principal_linear_constant("a"))


if __name__ == "__main__":
    main()
