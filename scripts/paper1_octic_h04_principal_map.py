#!/usr/bin/env python3
"""Principal-symbol map for the orthorhombic quartic Hamiltonian H04."""

from __future__ import annotations

import sympy as sp


def paper1_h04_principal_map() -> dict[str, object]:
    tau_xxxx, tau_yyyy, tau_zzzz, tau_xxyy, tau_xxzz, tau_yyzz = sp.symbols(
        "tau_xxxx tau_yyyy tau_zzzz tau_xxyy tau_xxzz tau_yyzz"
    )
    q400, q040, q004, q220, q202, q022 = sp.symbols(
        "q400 q040 q004 q220 q202 q022"
    )

    maps = {
        "q400": sp.Eq(q400, sp.Rational(1, 4) * tau_xxxx),
        "q040": sp.Eq(q040, sp.Rational(1, 4) * tau_yyyy),
        "q004": sp.Eq(q004, sp.Rational(1, 4) * tau_zzzz),
        "q220": sp.Eq(q220, sp.Rational(3, 2) * tau_xxyy),
        "q202": sp.Eq(q202, sp.Rational(3, 2) * tau_xxzz),
        "q022": sp.Eq(q022, sp.Rational(3, 2) * tau_yyzz),
    }

    return {
        "maps": maps,
        "statement": (
            "The orthorhombic quartic principal source is the commuting degree-4 "
            "polynomial obtained from the standard tau tensor; the six principal "
            "coefficients are linear in the six orthorhombic quartic components."
        ),
    }


def main() -> None:
    out = paper1_h04_principal_map()
    for eq in out["maps"].values():
        print(eq)


if __name__ == "__main__":
    main()
