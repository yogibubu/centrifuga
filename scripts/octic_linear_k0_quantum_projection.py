#!/usr/bin/env python3
"""Exact K=0 reduction of ordered degree-8 plane monomials.

For the linear rotor control limit, the final scalar is read from diagonal
K=0 matrix elements as a polynomial in X = J(J+1). This script derives the
exact polynomial coefficients of the five ordered plane monomials of degree 8:

    J_b^8, J_b^6 J_c^2, J_b^4 J_c^4, J_b^2 J_c^6, J_c^8

for the chosen ordered representative. The coefficient of X^4 is the octic
contribution to the linear scalar L in this ordered-basis convention.
"""

from __future__ import annotations

import sympy as sp

PlaneMonomial = tuple[int, int, int]


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


def _k0_me_z_axis(J: int, mon: PlaneMonomial) -> sp.Expr:
    a, b, c = mon
    jx, jy, jz, i0 = _j_matrices(J)
    op = (jx**a) * (jy**b) * (jz**c)
    return sp.simplify(op[i0, i0])


def derive_plane_monomial_projection_weights_z_axis() -> dict[PlaneMonomial, dict[str, sp.Expr]]:
    """Return exact K=0 polynomial coefficients for the five plane monomials."""
    plane = [
        (8, 0, 0),
        (6, 2, 0),
        (4, 4, 0),
        (2, 6, 0),
        (0, 8, 0),
    ]
    out: dict[PlaneMonomial, dict[str, sp.Expr]] = {}
    a0, a1, a2, a3, a4 = sp.symbols("a0:5")
    for mon in plane:
        eqs = []
        for J in range(0, 5):
            X = sp.Integer(J * (J + 1))
            val = _k0_me_z_axis(J, mon)
            eqs.append(sp.Eq(a0 + a1 * X + a2 * X**2 + a3 * X**3 + a4 * X**4, val))
        sol = sp.solve(eqs, [a0, a1, a2, a3, a4], dict=True)[0]
        out[mon] = {
            "X0": sp.simplify(sol[a0]),
            "X1": sp.simplify(sol[a1]),
            "X2": sp.simplify(sol[a2]),
            "X3": sp.simplify(sol[a3]),
            "X4": sp.simplify(sol[a4]),
        }
    return out


def derive_plane_monomial_projection_weights_x_axis() -> dict[PlaneMonomial, dict[str, sp.Expr]]:
    """Return the same weights permuted to the x-axis linear convention."""
    z_axis = derive_plane_monomial_projection_weights_z_axis()
    # z-axis plane monomials (x,y) -> x-axis plane monomials (y,z)
    permuted = {}
    for (a, b, c), coeffs in z_axis.items():
        permuted[(c, a, b)] = coeffs
    return permuted


def project_x_axis_linear_polynomial(coeffs: dict[PlaneMonomial, sp.Expr]) -> dict[str, sp.Expr]:
    """Project x-axis plane coefficients to the K=0 polynomial in X = J(J+1)."""
    weights = derive_plane_monomial_projection_weights_x_axis()
    out = {f"X{p}": sp.Integer(0) for p in range(5)}
    for mon, coeff in coeffs.items():
        if mon not in weights:
            continue
        for key, w in weights[mon].items():
            out[key] = sp.simplify(out[key] + coeff * w)
    return out


def main() -> None:
    out = derive_plane_monomial_projection_weights_x_axis()
    for mon, coeffs in out.items():
        print(mon, coeffs)


if __name__ == "__main__":
    main()
