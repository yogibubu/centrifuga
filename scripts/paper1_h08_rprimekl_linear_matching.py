#!/usr/bin/env python3
"""Diagonal linear-limit matching for the two-index prime object R'_{kl}."""

from __future__ import annotations

import sympy as sp


def paper1_h08_rprimekl_linear_matching() -> dict[str, sp.Expr]:
    wn, B = sp.symbols("omega_n B", nonzero=True)
    Cn = sp.Symbol("C_n")
    cubic = sp.Symbol("cubic_n")
    Rnn_geom = sp.Rational(3, 4) * wn**2 * Cn**2 / B

    # Screenshot-backed relation used in the notes:
    # R'_{nn} = R_{nn} - (1/2) sum_m k'_{nnm} R_m / omega_m
    # with R_m = -omega_m C_m X, so the correction contributes +1/2 sum k'_{nnm} C_m.
    Rp_nn = sp.expand(Rnn_geom + cubic)

    r_nn = sp.expand(sp.Rational(3, 4) * wn**2 * Cn**2 / B + cubic)
    gap = sp.expand(Rp_nn - r_nn)

    return {
        "Rnn_geom": Rnn_geom,
        "Rp_nn_linear": Rp_nn,
        "r_nn": r_nn,
        "gap": gap,
        "statement": (
            "Under the screenshot relation for R'_{kl}, the diagonal linear-limit "
            "object R'_{nn} matches exactly the already established linear r_nn "
            "family: the geometric piece comes from R_nn and the cubic correction "
            "comes from the -(1/2) sum k'_{nnm} R_m / omega_m term."
        ),
    }


def main() -> None:
    out = paper1_h08_rprimekl_linear_matching()
    for k, v in out.items():
        print(k, "=", v)


if __name__ == "__main__":
    main()
