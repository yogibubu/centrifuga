#!/usr/bin/env python3
"""Exact linear map from Paper1 octic source blocks to cartesian coefficients.

This module does not invent the unreadable parts of Table V.  It fixes the
exact bookkeeping identity:

    Eq. (99) source blocks -> 15 cartesian octic coefficients -> 5 symmetric-top
    constants -> 1 linear constant.

The content is purely symbolic and linear.
"""

from __future__ import annotations

import sympy as sp

from scripts.derive_octic_operator_basis import (
    asymmetric_top_monomial_labels,
    project_asymmetric_cartesian_to_linear,
    project_asymmetric_cartesian_to_symmetric_top,
)


def paper1_octic_eq99_term_labels() -> tuple[str, ...]:
    """Return stable labels for the seven Eq. (99) source blocks."""
    return (
        "H08",
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
        "S07H02",
    )


def paper1_octic_eq99_prefactors() -> dict[str, sp.Expr]:
    """Return the exact printed prefactors of Eq. (99)."""
    I = sp.I
    return {
        "H08": sp.Integer(1),
        "S03S03S03H02": -I / 6,
        "S03S03H04": -sp.Rational(1, 2),
        "S03H06": I,
        "S05H04": I,
        "S05S03H02": -sp.Integer(1),
        "S07H02": I,
    }


def paper1_octic_cartesian_source_coefficients() -> dict[str, dict[str, sp.Symbol]]:
    """Return the 15 cartesian coefficients carried by each Eq. (99) block."""
    out: dict[str, dict[str, sp.Symbol]] = {}
    labels = asymmetric_top_monomial_labels()
    for term in paper1_octic_eq99_term_labels():
        out[term] = {label: sp.Symbol(f"{label}^{term}") for label in labels}
    return out


def paper1_octic_total_cartesian_coefficients() -> dict[str, sp.Expr]:
    """Return the exact Eq. (99) sum for the 15 cartesian octic coefficients."""
    coeffs = paper1_octic_cartesian_source_coefficients()
    pref = paper1_octic_eq99_prefactors()
    out: dict[str, sp.Expr] = {}
    for label in asymmetric_top_monomial_labels():
        out[label] = sp.simplify(
            sum(pref[term] * coeffs[term][label] for term in paper1_octic_eq99_term_labels())
        )
    return out


def paper1_octic_total_symmetric_top_constants(axis: str = "a") -> dict[str, sp.Expr]:
    """Project the exact total cartesian coefficients to the 5 symmetric-top constants."""
    return project_asymmetric_cartesian_to_symmetric_top(paper1_octic_total_cartesian_coefficients(), axis)


def paper1_octic_total_linear_constant(axis: str = "a") -> sp.Expr:
    """Project the exact total cartesian coefficients to the linear constant."""
    return project_asymmetric_cartesian_to_linear(paper1_octic_total_cartesian_coefficients(), axis)


def paper1_octic_termwise_linear_decomposition(axis: str = "a") -> dict[str, sp.Expr]:
    """Return the exact termwise contribution of Eq. (99) to the linear constant."""
    coeffs = paper1_octic_cartesian_source_coefficients()
    pref = paper1_octic_eq99_prefactors()
    out: dict[str, sp.Expr] = {}
    for term in paper1_octic_eq99_term_labels():
        out[term] = sp.simplify(pref[term] * project_asymmetric_cartesian_to_linear(coeffs[term], axis))
    return out


def paper1_octic_termwise_symmetric_top_decomposition(axis: str = "a") -> dict[str, dict[str, sp.Expr]]:
    """Return the exact termwise contribution of Eq. (99) to the 5 symmetric-top constants."""
    coeffs = paper1_octic_cartesian_source_coefficients()
    pref = paper1_octic_eq99_prefactors()
    out: dict[str, dict[str, sp.Expr]] = {}
    for term in paper1_octic_eq99_term_labels():
        proj = project_asymmetric_cartesian_to_symmetric_top(coeffs[term], axis)
        out[term] = {key: sp.simplify(pref[term] * value) for key, value in proj.items()}
    return out


def main() -> None:
    print("Eq.(99) term labels:", paper1_octic_eq99_term_labels())
    print("Linear constant:", paper1_octic_total_linear_constant("a"))


if __name__ == "__main__":
    main()
