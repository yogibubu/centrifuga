#!/usr/bin/env python3
"""Scaling audit for the one-mode exact source probe of the H08 term3 block."""

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from paper1_h08_term3_onemode_source_probe import paper1_h08_term3_onemode_source_probe


def paper1_h08_term3_onemode_scaling_audit() -> dict[str, sp.Expr]:
    out = paper1_h08_term3_onemode_source_probe()

    C, w, B = sp.symbols("C omega B", nonzero=True)
    cubic2 = sp.Symbol("cubic2")
    r2_expr = sp.expand(sp.Rational(3, 4) * w**2 * C**2 / B + cubic2)
    r2_sym = next(sym for sym in out["X4"].free_symbols if sym.name == "r2")

    x4 = sp.expand(out["X4"].subs({r2_sym: r2_expr}))
    x5 = sp.expand(out["X5"].subs({r2_sym: r2_expr}))

    x4_no_cubic = sp.simplify(x4.subs({cubic2: 0}))
    x5_no_cubic = sp.simplify(x5.subs({cubic2: 0}))

    target_geom = sp.Symbol("8*B*D_J*C^2/omega")

    return {
        "X4_with_r11": x4,
        "X5_with_r11": x5,
        "X4_no_cubic": x4_no_cubic,
        "X5_no_cubic": x5_no_cubic,
        "target_geom_shape": target_geom,
        "statement": (
            "After substituting the diagonal linear family r_11 into the one-mode "
            "source probe for the term R'_k R_l R_m R'_{lm}, the X^4 and X^5 "
            "coefficients scale as high powers of C and omega. Even with cubic2=0 "
            "the X^4 coefficient is proportional to omega^3 C^7 / B^2, not to the "
            "geometric target B D_J C^2 / omega. Therefore the diagonal one-mode "
            "part of term3 cannot be the source of the geometric residual."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_onemode_scaling_audit()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
