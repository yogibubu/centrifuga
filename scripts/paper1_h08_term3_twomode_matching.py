#!/usr/bin/env python3
"""Minimal bimode matching condition for the H08 term3 residual geometry."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_twomode_matching() -> dict[str, sp.Expr]:
    B, DJ, w1, w2 = sp.symbols("B D_J omega1 omega2", nonzero=True)
    C1, C2 = sp.symbols("C1 C2", nonzero=True)
    rp1, rp2, rp12 = sp.symbols("r'_1 r'_2 r'_{12}", nonzero=True)

    x4_1 = sp.simplify(sp.Rational(5, 64) * C1 * C2 * rp12 * rp1 / w1)
    x4_2 = sp.simplify(sp.Rational(5, 64) * C1 * C2 * rp12 * rp2 / w2)

    target_1 = sp.simplify(8 * B * DJ * C1**2 / w1)
    target_2 = sp.simplify(8 * B * DJ * C2**2 / w2)

    cond_1 = sp.Eq(rp12 * rp1, sp.simplify(sp.Rational(512, 5) * B * DJ * C1 / C2))
    cond_2 = sp.Eq(rp12 * rp2, sp.simplify(sp.Rational(512, 5) * B * DJ * C2 / C1))
    ratio = sp.Eq(rp1 / rp2, sp.simplify(C1**2 / C2**2))

    return {
        "x4_mode1": x4_1,
        "x4_mode2": x4_2,
        "target_mode1": target_1,
        "target_mode2": target_2,
        "cond_mode1": cond_1,
        "cond_mode2": cond_2,
        "ratio_condition": ratio,
        "statement": (
            "The first bimode/off-diagonal source probe implies that exact matching "
            "of the geometric term requires r'_{12} r'_1 = (512/5) B D_J C1/C2 and "
            "r'_{12} r'_2 = (512/5) B D_J C2/C1. Hence r'_1/r'_2 = C1^2/C2^2. "
            "This is the minimal multimode condition imposed by the probe."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_twomode_matching()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
