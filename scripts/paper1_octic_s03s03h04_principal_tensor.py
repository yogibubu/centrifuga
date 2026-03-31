#!/usr/bin/env python3
"""Principal-symbol tensor for the optical channel [S03,[S03,H04]].

The orthorhombic principal symbol of S03 is one-dimensional:

    S03^(ps,orth) = sigma3 * Jx Jy Jz,  sigma3 = -3 S111 / 2.

For the quartic commuting Hamiltonian we use the general orthorhombic degree-4
polynomial

    H04^(ps,orth) =
      q400 Jx^4 + q040 Jy^4 + q004 Jz^4
    + q220 Jx^2 Jy^2 + q202 Jx^2 Jz^2 + q022 Jy^2 Jz^2.

The present script computes the exact double Lie-Poisson bracket

    {S03^(ps,orth), {S03^(ps,orth), H04^(ps,orth)}}

and decomposes the result on the 15-dimensional orthorhombic degree-8 basis.
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


def paper1_octic_s03s03h04_principal_tensor() -> dict[str, object]:
    sigma3, S111 = sp.symbols("sigma3 S111")
    q400, q040, q004, q220, q202, q022 = sp.symbols(
        "q400 q040 q004 q220 q202 q022"
    )

    s03_mon = (1, 1, 1)
    h04_poly: dict[Monomial, sp.Expr] = {
        (4, 0, 0): q400,
        (0, 4, 0): q040,
        (0, 0, 4): q004,
        (2, 2, 0): q220,
        (2, 0, 2): q202,
        (0, 2, 2): q022,
    }

    first = _poisson_poly_with_monomial(h04_poly, s03_mon)
    first = {m: sp.expand(sigma3 * coeff) for m, coeff in first.items()}

    second = _poisson_poly_with_monomial(first, s03_mon)
    second = {m: sp.expand(sigma3 * coeff) for m, coeff in second.items()}

    orth_basis = degree8_orth_basis()
    orth_coeffs = {m: sp.expand(second.get(m, sp.Integer(0))) for m in orth_basis}

    sigma_to_S111 = sp.expand(-sp.Rational(3, 2) * S111)
    orth_coeffs_S111 = {
        m: sp.expand(coeff.subs({sigma3: sigma_to_S111}))
        for m, coeff in orth_coeffs.items()
    }

    return {
        "source_block": "[S03,[S03,H04]]",
        "h04_basis": tuple(h04_poly.keys()),
        "sigma3_to_S111": sp.Eq(sigma3, sigma_to_S111),
        "degree8_orth_basis": orth_basis,
        "orth_coeffs_sigma3": orth_coeffs,
        "orth_coeffs_S111": orth_coeffs_S111,
        "nonzero_orth_terms": tuple(m for m, coeff in orth_coeffs.items() if coeff != 0),
        "statement": (
            "The double principal-symbol bracket [S03,[S03,H04]] sends the "
            "general orthorhombic quartic source onto an explicit degree-8 "
            "orthorhombic polynomial, giving a concrete mu1*phi3^2 contribution "
            "to the 15 commuting octic coefficients."
        ),
    }


def main() -> None:
    out = paper1_octic_s03s03h04_principal_tensor()
    print(out["sigma3_to_S111"])
    for mon, coeff in out["orth_coeffs_S111"].items():
        if coeff != 0:
            print(mon, coeff)


if __name__ == "__main__":
    main()
