#!/usr/bin/env python3
"""Exact maps from printed Watson--Aliev coefficients to principal-symbol ones."""

from __future__ import annotations

import sympy as sp


def paper1_printed_to_principal_maps() -> dict[str, object]:
    S111, S311, S131, S113 = sp.symbols("S111 S311 S131 S113")
    Wxxx, Wyyy, Wzzz = sp.symbols("Wxxx Wyyy Wzzz")
    Wxxy, Wxxz, Wyyx, Wyyz, Wzzx, Wzzy = sp.symbols(
        "Wxxy Wxxz Wyyx Wyyz Wzzx Wzzy"
    )

    sigma3 = -sp.Rational(3, 2) * S111

    # From Eq. (95): each anticommutator-like pair has the same principal symbol
    # and therefore contributes a factor 2 to the commuting monomial.
    s311 = -2 * S311
    s131 = -2 * S131
    s113 = -2 * S113

    # From Eq. (96): J_a^6 keeps coefficient 1, while
    # (J_a^4 J_b^2 + J_b^2 J_a^4) -> 2 J_a^4 J_b^2 in the principal symbol.
    w600 = Wxxx
    w060 = Wyyy
    w006 = Wzzz
    w420 = 2 * Wxxy
    w402 = 2 * Wxxz
    w240 = 2 * Wyyx
    w042 = 2 * Wyyz
    w204 = 2 * Wzzx
    w024 = 2 * Wzzy

    return {
        "sigma3_map": sp.Eq(sp.Symbol("sigma3"), sigma3),
        "s05_maps": {
            "s311": sp.Eq(sp.Symbol("s311"), s311),
            "s131": sp.Eq(sp.Symbol("s131"), s131),
            "s113": sp.Eq(sp.Symbol("s113"), s113),
        },
        "h06_maps": {
            "w600": sp.Eq(sp.Symbol("w600"), w600),
            "w060": sp.Eq(sp.Symbol("w060"), w060),
            "w006": sp.Eq(sp.Symbol("w006"), w006),
            "w420": sp.Eq(sp.Symbol("w420"), w420),
            "w402": sp.Eq(sp.Symbol("w402"), w402),
            "w240": sp.Eq(sp.Symbol("w240"), w240),
            "w042": sp.Eq(sp.Symbol("w042"), w042),
            "w204": sp.Eq(sp.Symbol("w204"), w204),
            "w024": sp.Eq(sp.Symbol("w024"), w024),
        },
        "h06_missing_central_term": (
            "The principal-symbol coefficient w222 has no direct printed counterpart "
            "in Eq. (96), which is consistent with the fact that it drops out "
            "identically from [S03,H06]."
        ),
    }


def main() -> None:
    out = paper1_printed_to_principal_maps()
    print(out["sigma3_map"])
    for eq in out["s05_maps"].values():
        print(eq)
    for eq in out["h06_maps"].values():
        print(eq)
    print(out["h06_missing_central_term"])


if __name__ == "__main__":
    main()
