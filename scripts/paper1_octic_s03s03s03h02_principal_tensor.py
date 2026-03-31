#!/usr/bin/env python3
"""Principal-symbol tensor for the cubic optical channel [S03,[S03,[S03,H02]]].

At principal-symbol level the orthorhombic part of S03 collapses to a single
degree-3 monomial Jx Jy Jz. Starting from Eq. (89),

    S03 = sum_{alpha beta gamma} s_{alpha beta gamma} J_alpha J_beta J_gamma

with s_{alpha beta gamma} fully symmetric, the orthorhombic contribution is
the sum over the six permutations of xyz. Watson--Aliev note that the
coefficient s_xyz is written as -S111/4. Therefore the commuting principal
symbol of the orthorhombic S03 block is

    S03^(ps,orth) = sigma3 * Jx Jy Jz,  sigma3 = 6 s_xyz = -3 S111 / 2.

The present script computes the exact nested Lie-Poisson bracket

    {S03^(ps,orth), {S03^(ps,orth), {S03^(ps,orth), H02}}}

with H02 = A Jx^2 + B Jy^2 + C Jz^2, and decomposes the result on the
15-dimensional orthorhombic degree-8 basis.
"""

from __future__ import annotations

import sympy as sp

try:
    from scripts.paper1_h08_orth_projector import degree8_orth_basis
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.paper1_h08_orth_projector import degree8_orth_basis

Monomial = tuple[int, int, int]


def _add_term(out: dict[Monomial, sp.Expr], mon: Monomial, coeff: sp.Expr) -> None:
    out[mon] = sp.expand(out.get(mon, sp.Integer(0)) + coeff)


def _poisson_monomial_with_h02(a: int, b: int, c: int) -> dict[Monomial, sp.Expr]:
    A, B, C = sp.symbols("A B C")
    out: dict[Monomial, sp.Expr] = {}
    if a:
        _add_term(out, (a - 1, b + 1, c + 1), sp.Integer(2) * a * (C - B))
    if b:
        _add_term(out, (a + 1, b - 1, c + 1), sp.Integer(2) * b * (A - C))
    if c:
        _add_term(out, (a + 1, b + 1, c - 1), sp.Integer(2) * c * (B - A))
    return out


def _poisson_monomials(m1: Monomial, m2: Monomial) -> dict[Monomial, sp.Expr]:
    a, b, c = m1
    d, e, f = m2
    out: dict[Monomial, sp.Expr] = {}
    term_xy = sp.Integer(1) * (b * d - a * e)
    if term_xy:
        _add_term(out, (a + d - 1, b + e - 1, c + f + 1), term_xy)
    term_xz = sp.Integer(1) * (a * f - c * d)
    if term_xz:
        _add_term(out, (a + d - 1, b + e + 1, c + f - 1), term_xz)
    term_yz = sp.Integer(1) * (c * e - b * f)
    if term_yz:
        _add_term(out, (a + d + 1, b + e - 1, c + f - 1), term_yz)
    return out


def _poisson_poly_with_monomial(poly: dict[Monomial, sp.Expr], mon: Monomial) -> dict[Monomial, sp.Expr]:
    out: dict[Monomial, sp.Expr] = {}
    for m, coeff in poly.items():
        for tgt, tgt_coeff in _poisson_monomials(mon, m).items():
            _add_term(out, tgt, sp.expand(coeff * tgt_coeff))
    return out


def paper1_octic_s03s03s03h02_principal_tensor() -> dict[str, object]:
    A, B, C, sigma3, S111 = sp.symbols("A B C sigma3 S111")
    s03_mon = (1, 1, 1)

    first = _poisson_monomial_with_h02(*s03_mon)
    first = {m: sp.expand(sigma3 * coeff) for m, coeff in first.items()}

    second = _poisson_poly_with_monomial(first, s03_mon)
    second = {m: sp.expand(sigma3 * coeff) for m, coeff in second.items()}

    third = _poisson_poly_with_monomial(second, s03_mon)
    third = {m: sp.expand(sigma3 * coeff) for m, coeff in third.items()}

    orth_basis = degree8_orth_basis()
    orth_coeffs = {m: sp.expand(third.get(m, sp.Integer(0))) for m in orth_basis}

    sigma_to_S111 = sp.expand(-sp.Rational(3, 2) * S111)
    orth_coeffs_S111 = {
        m: sp.expand(coeff.subs({sigma3: sigma_to_S111}))
        for m, coeff in orth_coeffs.items()
    }

    return {
        "source_block": "[S03,[S03,[S03,H02]]]",
        "principal_generator": sigma3 * sp.Symbol("J_x J_y J_z", commutative=False),
        "sigma3_to_S111": sp.Eq(sigma3, sigma_to_S111),
        "degree8_orth_basis": orth_basis,
        "orth_coeffs_sigma3": orth_coeffs,
        "orth_coeffs_S111": orth_coeffs_S111,
        "nonzero_orth_terms": tuple(m for m, coeff in orth_coeffs.items() if coeff != 0),
        "statement": (
            "The orthorhombic principal symbol of S03 is one-dimensional. Its "
            "triple Lie-Poisson action on H02 generates an explicit degree-8 "
            "orthorhombic polynomial, hence a direct contribution to the 15 "
            "commuting octic coefficients."
        ),
    }


def main() -> None:
    out = paper1_octic_s03s03s03h02_principal_tensor()
    print(out["sigma3_to_S111"])
    for mon, coeff in out["orth_coeffs_S111"].items():
        if coeff != 0:
            print(mon, coeff)


if __name__ == "__main__":
    main()
