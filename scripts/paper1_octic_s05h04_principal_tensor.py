#!/usr/bin/env python3
"""Principal-symbol tensor for the optical channel [S05,H04]."""

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


def paper1_octic_s05h04_principal_tensor() -> dict[str, object]:
    s311, s131, s113 = sp.symbols("s311 s131 s113")
    q400, q040, q004, q220, q202, q022 = sp.symbols(
        "q400 q040 q004 q220 q202 q022"
    )

    s05_poly: dict[Monomial, sp.Expr] = {
        (3, 1, 1): s311,
        (1, 3, 1): s131,
        (1, 1, 3): s113,
    }
    h04_poly: dict[Monomial, sp.Expr] = {
        (4, 0, 0): q400,
        (0, 4, 0): q040,
        (0, 0, 4): q004,
        (2, 2, 0): q220,
        (2, 0, 2): q202,
        (0, 2, 2): q022,
    }

    outer: dict[Monomial, sp.Expr] = {}
    for mon_s05, coeff_s05 in s05_poly.items():
        contrib = _poisson_poly_with_monomial(h04_poly, mon_s05)
        for mon, coeff in contrib.items():
            _add_term(outer, mon, sp.expand(coeff_s05 * coeff))

    orth_basis = degree8_orth_basis()
    orth_coeffs = {m: sp.expand(outer.get(m, sp.Integer(0))) for m in orth_basis}

    return {
        "source_block": "[S05,H04]",
        "s05_basis": tuple(s05_poly.keys()),
        "h04_basis": tuple(h04_poly.keys()),
        "degree8_orth_basis": orth_basis,
        "orth_coeffs": orth_coeffs,
        "nonzero_orth_terms": tuple(m for m, coeff in orth_coeffs.items() if coeff != 0),
        "statement": (
            "The principal-symbol bracket [S05,H04] sends the 3-parameter "
            "orthorhombic S05 source and the 6-parameter orthorhombic quartic "
            "source onto an explicit degree-8 orthorhombic polynomial, giving "
            "the last direct optical channel in concrete form."
        ),
    }


def main() -> None:
    out = paper1_octic_s05h04_principal_tensor()
    for mon, coeff in out["orth_coeffs"].items():
        if coeff != 0:
            print(mon, coeff)


if __name__ == "__main__":
    main()
