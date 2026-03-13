#!/usr/bin/env python3
"""Derive quartic representation transforms for Watson S reduction.

The script constructs Q_S(R) directly from the quartic S-reduction Hamiltonian,
without assuming Q_S = Q_A, and computes

    D_S(R') = T_S(R->R') D_S(R)

for R,R' in {I, II, III}, with

    D_S = (DJ, DJK, DK, d1, d2)^T.
"""

from __future__ import annotations

import sympy as sp


# Symbols for rotational operators (commuting polynomial representation).
Ja, Jb, Jc = sp.symbols("Ja Jb Jc", real=True)
Jx, Jy, Jz = sp.symbols("Jx Jy Jz", real=True)

# Quartic S-reduction constants.
DJ, DJK, DK, d1, d2 = sp.symbols("DJ DJK DK d1 d2", real=True)

# Rotational constants (needed in weighted determinable combination).
A, B, C = sp.symbols("A B C", real=True)


def axis_map(rep: str) -> dict[sp.Symbol, sp.Symbol]:
    """Return x,y,z -> a,b,c mapping for representation Ir/IIr/IIIr."""
    rep = rep.strip().upper()
    if rep == "I":
        # a=x, b=y, c=z
        return {Jx: Ja, Jy: Jb, Jz: Jc}
    if rep == "II":
        # a=z, b=x, c=y  => x=b, y=c, z=a
        return {Jx: Jb, Jy: Jc, Jz: Ja}
    if rep == "III":
        # a=y, b=z, c=x  => x=c, y=a, z=b
        return {Jx: Jc, Jy: Ja, Jz: Jb}
    raise ValueError("Representation must be I, II, or III")


def h4_s_in_abc(rep: str) -> sp.Expr:
    """Quartic S-reduction Hamiltonian expanded in Ja,Jb,Jc for a representation."""
    J2 = Jx**2 + Jy**2 + Jz**2
    h4 = (
        -DJ * J2**2
        - DJK * J2 * Jz**2
        - DK * Jz**4
        - d1 * J2 * (Jx**2 - Jy**2)
        - d2 * (Jx**4 - Jy**4)
    )

    mapped = sp.expand(h4.subs(axis_map(rep)))
    return mapped


def monomial_coeffs(poly: sp.Expr) -> dict[str, sp.Expr]:
    """Extract quartic monomial coefficients in principal-axis basis."""
    p = sp.Poly(sp.expand(poly), Ja, Jb, Jc)
    return {
        "aa": p.coeff_monomial(Ja**4),
        "bb": p.coeff_monomial(Jb**4),
        "cc": p.coeff_monomial(Jc**4),
        "ab": p.coeff_monomial(Ja**2 * Jb**2),
        "ac": p.coeff_monomial(Ja**2 * Jc**2),
        "bc": p.coeff_monomial(Jb**2 * Jc**2),
    }


def determinable_vector_from_coeffs(coeffs: dict[str, sp.Expr]) -> sp.Matrix:
    """Build T = (Taa, Tbb, Tcc, T1, T2)^T from quartic monomial coefficients.

    Definitions used:
      Taa = coeff(Ja^4)
      Tbb = coeff(Jb^4)
      Tcc = coeff(Jc^4)
      T1  = 1/2 * [coeff(Ja^2 Jb^2) + coeff(Ja^2 Jc^2) + coeff(Jb^2 Jc^2)]
      T2  = A*coeff(Ja^2 Jb^2) + B*coeff(Ja^2 Jc^2) + C*coeff(Jb^2 Jc^2)

    T2 is the rotational-constant weighted cross-term combination.
    """
    c_ab = coeffs["ab"]
    c_ac = coeffs["ac"]
    c_bc = coeffs["bc"]
    return sp.Matrix([
        coeffs["aa"],
        coeffs["bb"],
        coeffs["cc"],
        sp.Rational(1, 2) * (c_ab + c_ac + c_bc),
        A * c_ab + B * c_ac + C * c_bc,
    ])


def q_s(rep: str) -> sp.Matrix:
    """Construct Q_S(R) from T = Q_S(R) D_S(R)."""
    poly = h4_s_in_abc(rep)
    coeffs = monomial_coeffs(poly)
    T = determinable_vector_from_coeffs(coeffs)
    D = sp.Matrix([DJ, DJK, DK, d1, d2])

    # Jacobian wrt D gives the linear map Q_S.
    Q = T.jacobian(D)
    return sp.simplify(Q)


def t_s(rep_from: str, rep_to: str) -> sp.Matrix:
    """Representation transform D_S(rep_to) = T_S D_S(rep_from)."""
    Qf = q_s(rep_from)
    Qt = q_s(rep_to)
    return sp.simplify(Qt.inv() * Qf)


def ns_matrix(M: sp.Matrix) -> sp.Matrix:
    return M.applyfunc(lambda x: sp.nsimplify(sp.simplify(x), rational=True))


def print_matrix(title: str, M: sp.Matrix) -> None:
    print(f"\n{title}")
    sp.pprint(ns_matrix(M), use_unicode=True)


def verify_identities() -> None:
    I5 = sp.eye(5)

    T_I_II = t_s("I", "II")
    T_II_I = t_s("II", "I")
    T_I_III = t_s("I", "III")
    T_III_I = t_s("III", "I")
    T_II_III = t_s("II", "III")

    round_1 = (T_I_II * T_II_I - I5).applyfunc(sp.simplify)
    round_2 = (T_I_III * T_III_I - I5).applyfunc(sp.simplify)

    # Composition acts right-to-left on D vectors.
    cyclic = (T_III_I * T_II_III * T_I_II - I5).applyfunc(sp.simplify)

    # Determinable invariance: Q_S(R') T(R->R') = Q_S(R)
    QI = q_s("I")
    QII = q_s("II")
    QIII = q_s("III")
    inv_12 = (QII * T_I_II - QI).applyfunc(sp.simplify)
    inv_13 = (QIII * T_I_III - QI).applyfunc(sp.simplify)
    inv_23 = (QIII * T_II_III - QII).applyfunc(sp.simplify)

    print("\nVerification checks (all must be zero matrices):")
    print("Round-trip I<->II zero:", round_1 == sp.zeros(5))
    print("Round-trip I<->III zero:", round_2 == sp.zeros(5))
    print("Cyclic closure zero:", cyclic == sp.zeros(5))
    print("Invariance I->II zero:", inv_12 == sp.zeros(5))
    print("Invariance I->III zero:", inv_13 == sp.zeros(5))
    print("Invariance II->III zero:", inv_23 == sp.zeros(5))


def print_numeric_example(A0: float, B0: float, C0: float) -> None:
    subs_abc = {A: sp.Float(A0), B: sp.Float(B0), C: sp.Float(C0)}

    mats = {
        "T_S(I->II)": t_s("I", "II"),
        "T_S(I->III)": t_s("I", "III"),
        "T_S(II->III)": t_s("II", "III"),
    }

    print(f"\nNumeric evaluation at A={A0}, B={B0}, C={C0}:")
    for name, M in mats.items():
        print(f"\n{name}")
        Mn = M.subs(subs_abc).evalf(12)
        sp.pprint(Mn, use_unicode=True)


def main() -> None:
    QI = q_s("I")
    QII = q_s("II")
    QIII = q_s("III")

    T_I_II = t_s("I", "II")
    T_I_III = t_s("I", "III")
    T_II_III = t_s("II", "III")

    print_matrix("Q_S(I)", QI)
    print_matrix("Q_S(II)", QII)
    print_matrix("Q_S(III)", QIII)

    print_matrix("T_S(I->II)", T_I_II)
    print_matrix("T_S(I->III)", T_I_III)
    print_matrix("T_S(II->III)", T_II_III)

    verify_identities()
    print_numeric_example(7036.58, 6910.83, 4218.78)


if __name__ == "__main__":
    main()
