#!/usr/bin/env python3
"""Exact off-diagonal residue required in the one-line term3 linear matching."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term3_offdiag_residue() -> dict[str, sp.Expr]:
    B, D, wn, Cn = sp.symbols("B D wn Cn", nonzero=True)
    cubic_n, cubic2_n = sp.symbols("cubic_n cubic2_n")

    rn = sp.expand(4 * wn * Cn * (D / B - B**2 / wn**2) + cubic_n)
    rnn = sp.expand(sp.Rational(3, 4) * wn**2 * Cn**2 / B + cubic2_n)

    target_S = sp.simplify(-8 * B * D * Cn**2 / rn)
    diag_S = sp.expand(Cn**2 * rnn)
    offdiag_residue = sp.simplify(target_S - diag_S)

    return {
        "r_n": rn,
        "r_nn": rnn,
        "target_S_n": target_S,
        "diag_S_n": diag_S,
        "offdiag_residue": offdiag_residue,
        "statement": (
            "If u_n = r_n and the diagonal reduction gives R'_{nn} = r_nn, then "
            "the full Watson-Aliev contraction sum_lm C_l C_m R'_{lm} must contain "
            "an off-diagonal residue equal to target_S_n - C_n^2 r_nn in order to "
            "reproduce the geometric 8 B D_J C_n^2 / omega_n term."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_offdiag_residue()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
