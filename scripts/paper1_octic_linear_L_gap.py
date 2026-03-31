#!/usr/bin/env python3
"""Audit the gap between raw quartic c080 and the established linear L formula."""

from __future__ import annotations

import sympy as sp


def linear_L_gap_one_mode() -> dict[str, sp.Expr]:
    B, D_J, omega = sp.symbols("B D_J omega", nonzero=True)
    C0, r0, k4 = sp.symbols("C0 r0 k4")

    raw_c080 = sp.simplify(sp.Rational(1, 24) * k4 * C0**4)
    known_L = sp.simplify(raw_c080 - 8 * D_J**3 / B**2 + 8 * B * D_J * C0**2 / omega - r0**2 / (2 * omega))
    gap = sp.simplify(known_L - raw_c080)

    return {
        "raw_c080": raw_c080,
        "known_L": known_L,
        "gap": gap,
    }


def main() -> None:
    out = linear_L_gap_one_mode()
    print("raw_c080 =", out["raw_c080"])
    print("known_L =", out["known_L"])
    print("gap =", out["gap"])


if __name__ == "__main__":
    main()
