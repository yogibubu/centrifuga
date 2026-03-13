#!/usr/bin/env python3
"""Derive sextic transformation matrices in a Yamada-style invariant framework.

The key structural object is a representation-independent five-dimensional
subspace S_6 of the degree-six rotational polynomial space.  Choosing a basis
{sigma_1,...,sigma_5} for S_6 gives, in each principal-axis representation R,

    H_R = B_R sigma

where B_R is the 7x5 matrix expressing the canonical sigma coordinates in
Watson's sextic constants for that representation.  Representation changes act
inside the same subspace and therefore induce

    T6(R->R') = B_R' * pinv(B_R).

The complementary two-dimensional sector is the residual space used in the
CeDiTT 5+2 decomposition.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

Ja, Jb, Jc = sp.symbols("Ja Jb Jc")


def quartic_Q(rep: str, A: sp.Expr, B: sp.Expr, C: sp.Expr) -> sp.Matrix:
    rep = rep.strip().upper()
    r1 = sp.Matrix([[-1, -1, -1, 0, 0]])
    r2 = sp.Matrix([[-1, 0, 0, -2, 0]])
    r3 = sp.Matrix([[-1, 0, 0, 2, 0]])
    r4 = sp.Matrix([[-3, -1, 0, 0, 0]])

    if rep == "I":
        rows = [r1, r2, r3, r4, sp.Matrix([[A + B + C, -(B + C) / 2, 0, B - C, B - C]])]
    elif rep == "II":
        rows = [r3, r1, r2, r4, sp.Matrix([[A + B + C, -(A + B) / 2, 0, A - B, A - B]])]
    elif rep == "III":
        rows = [r2, r3, r1, r4, sp.Matrix([[A + B + C, -(C + A) / 2, 0, C - A, C - A]])]
    else:
        raise ValueError("rep must be I, II, III")
    return sp.Matrix.vstack(*rows)


def infer_axis_maps_from_quartic() -> dict[str, tuple[sp.Symbol, sp.Symbol, sp.Symbol]]:
    A, B, C = sp.Integer(11), sp.Integer(7), sp.Integer(5)
    qI = quartic_Q("I", A, B, C)
    tpls = [tuple(qI[i, :]) for i in range(3)]

    cycles: dict[str, tuple[int, int, int]] = {}
    for rep in ("I", "II", "III"):
        q = quartic_Q(rep, A, B, C)
        cycles[rep] = tuple(tpls.index(tuple(q[i, :])) for i in range(3))

    if cycles != {"I": (0, 1, 2), "II": (2, 0, 1), "III": (1, 2, 0)}:
        raise RuntimeError(f"Unexpected quartic cycle pattern: {cycles}")

    return {
        "I": (Ja, Jb, Jc),
        "II": (Jc, Ja, Jb),
        "III": (Jb, Jc, Ja),
    }


def sextic_watson_basis(a: sp.Expr, b: sp.Expr, c: sp.Expr) -> list[sp.Expr]:
    J2 = a**2 + b**2 + c**2
    Jz, Jx, Jy = a, b, c
    return [
        sp.expand(J2**3),
        sp.expand(J2**2 * Jz**2),
        sp.expand(J2 * Jz**4),
        sp.expand(Jz**6),
        sp.expand(Jx**4 * Jy**2 - Jx**2 * Jy**4),
        sp.expand(Jx**2 * Jy**2 * Jz**2),
        sp.expand(Jz**2 * (Jx**4 - Jy**4)),
    ]


def invariant_sigma_basis() -> list[sp.Expr]:
    """Canonical basis of the sextic invariant subspace S_6.

    These five operators span the representation-independent subspace used by
    the CeDiTT sextic transform.  Under cyclic relabelings of the principal
    axes, the span is preserved even though the individual basis elements mix.
    """
    wI = sextic_watson_basis(Ja, Jb, Jc)
    return [
        sp.expand(wI[0] + 2 * wI[1] - wI[4]),
        sp.expand(3 * wI[2] + wI[5]),
        sp.expand(wI[3] - wI[6]),
        sp.expand(wI[0] - wI[2] + 2 * wI[4]),
        sp.expand(wI[1] + wI[6]),
    ]


def monomial_list(exprs: list[sp.Expr]) -> list[tuple[int, int, int]]:
    mons: set[tuple[int, int, int]] = set()
    for e in exprs:
        mons.update(sp.Poly(sp.expand(e), Ja, Jb, Jc).monoms())
    return sorted(mons)


def poly_vec(e: sp.Expr, mons: list[tuple[int, int, int]]) -> sp.Matrix:
    p = sp.Poly(sp.expand(e), Ja, Jb, Jc)
    return sp.Matrix([p.coeff_monomial(m) for m in mons])


def full_col_rank_pinv(B: sp.Matrix) -> sp.Matrix:
    bt = B.T
    return (bt * B).inv() * bt


def derive_B_matrix(rep: str, sigma_ops: list[sp.Expr]) -> sp.Matrix:
    axis = infer_axis_maps_from_quartic()[rep]
    sigma_rep = [sp.expand(op) for op in sigma_ops]

    w = sextic_watson_basis(*axis)
    mons = monomial_list(w + sigma_rep)

    Wmat = sp.Matrix.hstack(*[poly_vec(op, mons) for op in w])

    cols = []
    for s in sigma_rep:
        v = poly_vec(s, mons)
        coeff = full_col_rank_pinv(Wmat) * v
        cols.append(sp.nsimplify(coeff, rational=True))

    return sp.Matrix.hstack(*cols)


def derive_T6(B_from: sp.Matrix, B_to: sp.Matrix) -> sp.Matrix:
    return sp.simplify(B_to * full_col_rank_pinv(B_from))


def pretty_matrix(name: str, M: sp.Matrix) -> None:
    print(name)
    print("exact rational:")
    sp.pprint(sp.nsimplify(M, rational=True))
    print("numeric:")
    arr = np.array(M.evalf(), dtype=float)
    with np.printoptions(precision=6, suppress=True):
        print(arr)
    print()


def verify_subspace_consistency(Ba: sp.Matrix, Bb: sp.Matrix, ntests: int = 6, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    Ta_b = np.array(derive_T6(Ba, Bb).evalf(), dtype=float)
    Tb_a = np.array(derive_T6(Bb, Ba).evalf(), dtype=float)
    BA = np.array(Ba.evalf(), dtype=float)

    errs = []
    for _ in range(ntests):
        sigma = rng.normal(size=5)
        h_a = BA @ sigma
        h_b = Ta_b @ h_a
        h_back = Tb_a @ h_b
        errs.append(float(np.max(np.abs(h_back - h_a))))

    print(f"subspace round-trip errors: {errs}")
    print(f"worst error: {max(errs):.3e}\n")


def main() -> None:
    sigma_ops = invariant_sigma_basis()

    BI = derive_B_matrix("I", sigma_ops)
    BII = derive_B_matrix("II", sigma_ops)
    BIII = derive_B_matrix("III", sigma_ops)

    T6_I_II = derive_T6(BI, BII)
    T6_I_III = derive_T6(BI, BIII)
    T6_II_III = derive_T6(BII, BIII)

    pretty_matrix("B_I (7x5)", BI)
    pretty_matrix("B_II (7x5)", BII)
    pretty_matrix("B_III (7x5)", BIII)

    pretty_matrix("T6_I_II", T6_I_II)
    pretty_matrix("T6_I_III", T6_I_III)
    pretty_matrix("T6_II_III", T6_II_III)

    print("Verification I <-> II")
    verify_subspace_consistency(BI, BII)

    print("Verification I <-> III")
    verify_subspace_consistency(BI, BIII)


if __name__ == "__main__":
    main()
