#!/usr/bin/env python3
"""True-linear audit of the one-line commutator term -i R'_k R_l [R_k,R_l]."""

from __future__ import annotations

import sympy as sp


def paper1_h08_term4_true_linear_vanishing() -> dict[str, sp.Expr]:
    wk, wl = sp.symbols("omega_k omega_l", nonzero=True)
    Ck, Cl = sp.symbols("C_k C_l")

    Xperp = sp.Symbol("J_y^2+J_z^2", commutative=False)
    Rk = -wk * Ck * Xperp
    Rl = -wl * Cl * Xperp
    comm = sp.expand(Rk * Rl - Rl * Rk)

    term4 = sp.expand(-sp.I * sp.Symbol("R'_k") * Rl * comm / (wk**2 * wl))

    return {
        "R_k_linear": Rk,
        "R_l_linear": Rl,
        "commutator": comm,
        "term4_linear": term4,
        "statement": (
            "In the true linear limit, if the exact rotational blocks R_k and R_l "
            "collapse to scalar multiples of the same transverse operator J_y^2+J_z^2, "
            "then [R_k,R_l]=0 identically and the visible one-line commutator term "
            "-i R'_k R_l [R_k,R_l]/(omega_k^2 omega_l) vanishes."
        ),
    }


def main() -> None:
    out = paper1_h08_term4_true_linear_vanishing()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
