#!/usr/bin/env python3
"""Two-mode/off-diagonal source probe for the one-line H08 term3 block."""

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


def paper1_h08_term3_twomode_probe() -> dict[str, sp.Expr]:
    C1, C2, w1, w2, r12, rp1 = sp.symbols("C1 C2 omega1 omega2 r12 rp1", nonzero=True)
    a0, a1, a2, a3, a4, a5 = sp.symbols("a0:6")

    eqs = []
    vals = {}
    for J in range(0, 6):
        jx, jy, jz, i0 = _j_matrices(J)
        Xperp = jy**2 + jz**2

        R1 = -w1 * C1 * Xperp
        R2 = -w2 * C2 * Xperp
        Rp12 = r12 * Xperp
        Rp1 = rp1 * Xperp**2

        term3 = sp.simplify(-(Rp1 * R1 * R2 * Rp12) / (w1 * w1 * w2))
        val = sp.simplify(term3[i0, i0])
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
            "The first off-diagonal two-mode source probe for the term "
            "R'_k R_l R_m R'_{lm} still produces a degree-5 polynomial in X. "
            "Its X^4 coefficient scales as rp1 * r12 * C1 * C2 divided by one "
            "harmonic frequency, so any recovery of the linear geometric term "
            "must come from a specific nontrivial identification of the reduced "
            "two-index and one-index prime blocks."
        ),
    }


def main() -> None:
    out = paper1_h08_term3_twomode_probe()
    for key, val in out.items():
        print(key, "=", val)


if __name__ == "__main__":
    main()
