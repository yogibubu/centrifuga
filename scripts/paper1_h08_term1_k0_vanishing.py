#!/usr/bin/env python3
"""Exact K=0 vanishing of the naive odd transverse term from RRR R_klm."""

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


def paper1_h08_term1_k0_vanishing() -> dict[str, object]:
    vals = {}
    for J in range(0, 7):
        jx, jy, jz, i0 = _j_matrices(J)
        X = sp.Integer(J * (J + 1))
        op = X**3 * (jy**3 + jz**3)
        vals[J] = sp.simplify(op[i0, i0])

    return {
        "k0_diagonal_values": vals,
        "statement": (
            "The naive true-linear geometric reduction of the one-line term "
            "RRR R_klm is proportional to X^3 (J_y^3 + J_z^3). Its exact K=0 "
            "diagonal matrix element vanishes for all tested J, so this naive "
            "object contributes nothing to the linear scalar L."
        ),
    }


def main() -> None:
    out = paper1_h08_term1_k0_vanishing()
    print(out["k0_diagonal_values"])
    print(out["statement"])


if __name__ == "__main__":
    main()
