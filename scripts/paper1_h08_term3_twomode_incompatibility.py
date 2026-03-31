#!/usr/bin/env python3
"""Incompatibility between the bimode matching condition and the known linear r_k branch."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_twomode_incompatibility() -> dict[str, sp.Expr]:
    B, D = sp.symbols("B D_J", nonzero=True)
    C1, C2, w1, w2 = sp.symbols("C1 C2 omega1 omega2", nonzero=True)
    cubic1, cubic2 = sp.symbols("cubic1 cubic2")

    r1 = sp.expand(4 * w1 * C1 * (D / B - B**2 / w1**2) + cubic1)
    r2 = sp.expand(4 * w2 * C2 * (D / B - B**2 / w2**2) + cubic2)
    ratio_known = sp.simplify(r1 / r2)
    ratio_probe = sp.simplify(C1**2 / C2**2)

    return {
        "known_ratio": ratio_known,
        "probe_ratio": ratio_probe,
        "difference_no_cubic": sp.simplify((ratio_known.subs({cubic1: 0, cubic2: 0}) - ratio_probe)),
        "statement": (
            "The bimode source probe requires r'_1/r'_2 = C1^2/C2^2. The known linear "
            "branch gives instead r_1/r_2 = [C1 omega1 (D_J/B - B^2/omega1^2)] / "
            "[C2 omega2 (D_J/B - B^2/omega2^2)] up to cubic corrections. These two "
            "ratios are generically different, so the probe condition is incompatible "
            "with identifying the reduced one-index prime block with the known linear r_k family."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_twomode_incompatibility()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
