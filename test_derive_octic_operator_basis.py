#!/usr/bin/env python3
"""Exact checks for the commuting octic operator basis."""

from __future__ import annotations

import sympy as sp

from scripts.derive_octic_operator_basis import (
    asymmetric_top_cartesian_polynomial,
    asymmetric_top_cartesian_symbols,
    asymmetric_top_octic_basis,
    basis_dimensions,
    derive_octic_projection_summary,
    extract_asymmetric_top_cartesian_coefficients,
    linear_cartesian_constraints,
    linear_octic_binomial_coefficients,
    linear_constant_from_symmetric_top,
    linear_octic_operator,
    project_asymmetric_cartesian_to_symmetric_top,
    prove_linear_limit_from_symmetric_top,
    solve_symmetric_top_octic_projection,
    symmetric_top_cartesian_constraints,
    symmetric_top_cartesian_polynomial,
    symmetric_top_octic_basis,
    symmetric_top_projection_matrix,
)


def test_octic_basis_dimensions_are_exact() -> None:
    dims = basis_dimensions()
    assert dims == {"asymmetric_top": 15, "symmetric_top": 5, "linear": 1}


def test_asymmetric_octic_basis_has_15_distinct_monomials() -> None:
    basis = asymmetric_top_octic_basis()
    assert len(basis) == 15
    assert len({sp.srepr(sp.expand(expr)) for expr in basis}) == 15


def test_symmetric_top_basis_has_5_distinct_operators() -> None:
    basis = symmetric_top_octic_basis("a")
    assert len(basis) == 5
    assert len({sp.srepr(sp.expand(expr)) for expr in basis}) == 5


def test_linear_limit_is_single_binomial_operator() -> None:
    Jx, Jy, Jz = sp.symbols("Jx Jy Jz")
    expr = linear_octic_operator("a")
    assert sp.expand(expr - (Jy**2 + Jz**2) ** 4) == 0


def test_linear_binomial_coefficients_are_exact() -> None:
    Jy, Jz = sp.symbols("Jy Jz")
    coeffs = linear_octic_binomial_coefficients("a")
    expected = {
        Jy**8: 1,
        Jy**6 * Jz**2: 4,
        Jy**4 * Jz**4: 6,
        Jy**2 * Jz**6: 4,
        Jz**8: 1,
    }
    assert coeffs == expected


def test_symmetric_top_projection_matrix_has_shape_15x5() -> None:
    proj = symmetric_top_projection_matrix("a")
    assert proj.shape == (15, 5)


def test_symmetric_top_basis_collapses_to_single_linear_operator() -> None:
    out = prove_linear_limit_from_symmetric_top("a")
    linear = out["linear_operator"]
    assert sp.expand(out["basis_1"] - linear) == 0
    assert sp.expand(out["basis_2"]) == 0
    assert sp.expand(out["basis_3"]) == 0
    assert sp.expand(out["basis_4"]) == 0
    assert sp.expand(out["basis_5"]) == 0


def test_symmetric_top_cartesian_to_basis_projection_solves_exactly() -> None:
    sol = solve_symmetric_top_octic_projection("a")
    target = symmetric_top_cartesian_polynomial("a")
    LJ4, LJ3K, LJ2K2, LJK3, LK4 = sol["basis_symbols"]
    inv = sol["inverse_on_image"]
    Ja, Jb, Jc = sp.symbols("Jx Jy Jz")
    J2 = Ja**2 + Jb**2 + Jc**2
    combo = sp.expand(
        inv[LJ4] * J2**4
        + inv[LJ3K] * J2**3 * Ja**2
        + inv[LJ2K2] * J2**2 * Ja**4
        + inv[LJK3] * J2 * Ja**6
        + inv[LK4] * Ja**8
    )
    constraints = symmetric_top_cartesian_constraints("a")
    residual = sp.expand(combo - target)
    residual = residual.subs(
        {
            sp.Symbol("c211"): 2 * sp.Symbol("c220"),
            sp.Symbol("c121"): 3 * sp.Symbol("c130"),
            sp.Symbol("c031"): 4 * sp.Symbol("c040"),
            sp.Symbol("c022"): 6 * sp.Symbol("c040"),
        }
    )
    assert sp.expand(residual) == 0


def test_linear_constant_is_first_symmetric_top_coefficient_only() -> None:
    LJ4, LJ3K, LJ2K2, LJK3, LK4 = sp.symbols("LJ4 LJ3K LJ2K2 LJK3 LK4")
    coeff = linear_constant_from_symmetric_top("a")
    assert sp.simplify(coeff - LJ4) == 0


def test_symmetric_top_cartesian_constraints_have_codimension_four() -> None:
    constraints = symmetric_top_cartesian_constraints("a")
    assert set(constraints) == {
        "constraint_c211",
        "constraint_c121",
        "constraint_c031",
        "constraint_c022",
    }


def test_linear_cartesian_constraints_are_exact() -> None:
    constraints = linear_cartesian_constraints("a")
    assert constraints["c_800"] == sp.Symbol("c_800")
    assert constraints["c_620"] == sp.Symbol("c_620")
    assert constraints["c_602"] == sp.Symbol("c_602")
    assert constraints["c_440"] == sp.Symbol("c_440")
    assert constraints["c_422"] == sp.Symbol("c_422")
    assert constraints["c_404"] == sp.Symbol("c_404")
    assert constraints["c_260"] == sp.Symbol("c_260")
    assert constraints["c_242"] == sp.Symbol("c_242")
    assert constraints["c_224"] == sp.Symbol("c_224")
    assert constraints["c_206"] == sp.Symbol("c_206")
    assert constraints["c_080"] == sp.Symbol("c_080") - sp.Symbol("L")
    assert constraints["c_062"] == sp.Symbol("c_062") - 4 * sp.Symbol("L")


def test_extract_asymmetric_cartesian_coefficients_is_exact() -> None:
    expr = asymmetric_top_cartesian_polynomial()
    coeffs = extract_asymmetric_top_cartesian_coefficients(expr)
    expected = asymmetric_top_cartesian_symbols()
    assert coeffs == expected


def test_project_asymmetric_cartesian_to_symmetric_top_is_exact_on_image() -> None:
    cart = {
        "c_800": sp.Symbol("c400"),
        "c_620": sp.Symbol("c310"),
        "c_602": sp.Symbol("c310"),
        "c_440": sp.Symbol("c220"),
        "c_422": 2 * sp.Symbol("c220"),
        "c_404": sp.Symbol("c220"),
        "c_260": sp.Symbol("c130"),
        "c_242": 3 * sp.Symbol("c130"),
        "c_224": 3 * sp.Symbol("c130"),
        "c_206": sp.Symbol("c130"),
        "c_080": sp.Symbol("c040"),
        "c_062": 4 * sp.Symbol("c040"),
        "c_044": 6 * sp.Symbol("c040"),
        "c_026": 4 * sp.Symbol("c040"),
        "c_008": sp.Symbol("c040"),
    }
    proj = project_asymmetric_cartesian_to_symmetric_top(cart, "a")
    assert proj["LJ4"] == sp.Symbol("c040")
    assert proj["LJ3K"] == -4 * sp.Symbol("c040") + sp.Symbol("c130")
    assert proj["LJ2K2"] == 6 * sp.Symbol("c040") - 3 * sp.Symbol("c130") + sp.Symbol("c220")
    assert proj["LJK3"] == -4 * sp.Symbol("c040") + 3 * sp.Symbol("c130") - 2 * sp.Symbol("c220") + sp.Symbol("c310")
    assert proj["LK4"] == sp.Symbol("c040") - sp.Symbol("c130") + sp.Symbol("c220") - sp.Symbol("c310") + sp.Symbol("c400")


def test_octic_projection_summary_is_consistent() -> None:
    out = derive_octic_projection_summary("a")
    assert len(out["cartesian_labels"]) == 15
    assert out["linear_constant"] == sp.Symbol("c_080")
    assert out["symmetric_top_constants"]["LJ4"] == sp.Symbol("c_080")
