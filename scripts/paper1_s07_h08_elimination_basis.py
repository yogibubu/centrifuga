#!/usr/bin/env python3
"""Explicit nonorthorhombic basis for the S07 -> H08 elimination problem."""

from __future__ import annotations

import sympy as sp

from scripts.orthorhombic_parity_decomposition import split_orthorhombic_monomials


def degree7_nonorth_basis() -> tuple[tuple[int, int, int], ...]:
    return split_orthorhombic_monomials(7)["nonorthorhombic_monomials"]


def degree8_nonorth_basis() -> tuple[tuple[int, int, int], ...]:
    return split_orthorhombic_monomials(8)["nonorthorhombic_monomials"]


def degree7_orth_basis() -> tuple[tuple[int, int, int], ...]:
    return split_orthorhombic_monomials(7)["orthorhombic_monomials"]


def s07_nonorth_symbols() -> dict[str, sp.Symbol]:
    basis = degree7_nonorth_basis()
    return {
        f"s7_{a}{b}{c}": sp.Symbol(f"s7_{a}{b}{c}")
        for (a, b, c) in basis
    }


def h08_nonorth_symbols() -> dict[str, sp.Symbol]:
    basis = degree8_nonorth_basis()
    return {
        f"h8_{a}{b}{c}": sp.Symbol(f"h8_{a}{b}{c}")
        for (a, b, c) in basis
    }


def s07_nonorth_ansatz() -> dict[str, object]:
    basis = degree7_nonorth_basis()
    coeffs = s07_nonorth_symbols()
    return {
        "basis": basis,
        "coefficients": coeffs,
        "dimension": len(basis),
    }


def h08_nonorth_target() -> dict[str, object]:
    basis = degree8_nonorth_basis()
    coeffs = h08_nonorth_symbols()
    return {
        "basis": basis,
        "coefficients": coeffs,
        "dimension": len(basis),
    }


def elimination_problem_summary() -> dict[str, object]:
    s07 = s07_nonorth_ansatz()
    h08 = h08_nonorth_target()
    return {
        "S07_nonorth_dimension": s07["dimension"],
        "H08_nonorth_dimension": h08["dimension"],
        "S07_nonorth_basis": s07["basis"],
        "H08_nonorth_basis": h08["basis"],
        "elimination_matrix_shape": (h08["dimension"], s07["dimension"]),
        "well_posed_square_problem": s07["dimension"] == h08["dimension"] == 30,
    }


def main() -> None:
    print(elimination_problem_summary())


if __name__ == "__main__":
    main()
