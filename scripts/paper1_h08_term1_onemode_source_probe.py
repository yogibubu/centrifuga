#!/usr/bin/env python3
"""One-mode exact source probe for the one-line term R_k R_l R_m R_klm."""

from __future__ import annotations

import sympy as sp


def _j_matrices(J: int) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, int]:
    mvals = list(range(-J, J + 1))
    dim = 2 * J + 1
    jp = sp.zeros(dim)
    jm = sp.zeros(dim)
    idx = {m: i for i, m in enumerate(mvals)}
    for m in mvals:
        if m < J:
            jp[idx[m + 1], idx[m]] = sp.sqrt(J * (J + 1) - m * (m + 1))
        if m > -J:
            jm[idx[m - 1], idx[m]] = sp.sqrt(J * (J + 1) - m * (m - 1))
    jx = (jp + jm) / 2
    jy = (jp - jm) / (2 * sp.I)
    jz = sp.diag(*mvals)
    return jx, jy, jz, idx[0]


def paper1_h08_term1_onemode_source_probe() -> dict[str, sp.Expr]:
    C, w, B = sp.symbols("C omega B", nonzero=True)
    a0, a1, a2, a3, a4, a5 = sp.symbols("a0:6")

    eqs = []
    vals = {}
    for J in range(0, 6):
        jx, jy, jz, i0 = _j_matrices(J)
        Xperp = jy**2 + jz**2
        R = -w * C * Xperp
        R3 = -w**3 * C**3 * Xperp**3
        R111 = -(w**3 * C**3 / (2 * B**2)) * (jy**3 + jz**3)
        term1 = sp.simplify(-(R3 * R111) / w**3)
        val = sp.simplify(term1[i0, i0])
        vals[J] = val
        X = sp.Integer(J * (J + 1))
        eqs.append(sp.Eq(a0 + a1 * X + a2 * X**2 + a3 * X**3 + a4 * X**4 + a5 * X**5, val))

    sol = sp.solve(eqs, [a0, a1, a2, a3, a4, a5], dict=True)[0]

    return {
        "k0_values": vals,
        "X0": sp.simplify(sol[a0]),
        "X1": sp.simplify(sol[a1]),
        "X2": sp.simplify(sol[a2]),
        "X3": sp.simplify(sol[a3]),
        "X4": sp.simplify(sol[a4]),
        "X5": sp.simplify(sol[a5]),
        "statement": (
            "In the one-mode true-linear source probe, the exact K=0 diagonal of "
            "the one-line term R_k R_l R_m R_klm vanishes identically for all tested J. "
            "So this naive source model cannot generate the missing constant X^4 piece."
        ),
    }


def main() -> None:
    out = paper1_h08_term1_onemode_source_probe()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
