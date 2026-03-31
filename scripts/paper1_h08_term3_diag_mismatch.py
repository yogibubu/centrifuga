#!/usr/bin/env python3
"""Show that the naive diagonal term3 reduction cannot equal the geometric L piece."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_diag_mismatch() -> dict[str, sp.Expr]:
    B, D, wn, Cn = sp.symbols("B D wn Cn", nonzero=True)
    cubic_n, cubic2_n = sp.symbols("cubic_n cubic2_n")

    rn = 4 * wn * Cn * (D / B - B**2 / wn**2) + cubic_n
    rnn = sp.Rational(3, 4) * wn**2 * Cn**2 / B + cubic2_n
    term3_diag = sp.expand(-Cn**2 * rn * rnn / wn)
    geom_target = 8 * B * D * Cn**2 / wn
    gap = sp.expand(term3_diag - geom_target)

    return {
        "term3_diag": term3_diag,
        "geom_target": geom_target,
        "gap": gap,
        "statement": (
            "If the one-line R'_kRRR'_{lm} term is reduced naively by taking only "
            "the diagonal matching R'_k -> r_n and R'_{nn} -> r_nn, its result is "
            "not the geometric 8 B D_J C_n^2 / omega_n term. Therefore the "
            "geometric piece must come from a nontrivial effective reduction of "
            "the full lm block, not from the naive diagonal alone."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_diag_mismatch()
    for k, v in out.items():
        print(k, "=", v)


if __name__ == "__main__":
    main()
