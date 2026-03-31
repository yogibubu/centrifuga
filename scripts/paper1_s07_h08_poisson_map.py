#!/usr/bin/env python3
"""Exact Lie-Poisson elimination map from S07(nonorth) to H08(nonorth).

For the principal symbol of the orthorhombic elimination problem one uses the
rigid-rotor quadratic Hamiltonian

    H02 = A Jx^2 + B Jy^2 + C Jz^2

and the so(3)^* Lie-Poisson bracket

    {f,g} = J · (∇f × ∇g).

For a monomial f = Jx^a Jy^b Jz^c this gives

    {f,H02}
      = 2 a (C-B) Jx^(a-1) Jy^(b+1) Jz^(c+1)
      + 2 b (A-C) Jx^(a+1) Jy^(b-1) Jz^(c+1)
      + 2 c (B-A) Jx^(a+1) Jy^(b+1) Jz^(c-1).

This maps degree 7 to degree 8 and preserves the nonorthorhombic parity class.
"""

from __future__ import annotations

import sympy as sp

from scripts.paper1_s07_h08_elimination_basis import degree7_nonorth_basis, degree8_nonorth_basis


def poisson_h02_on_monomial(a: int, b: int, c: int) -> dict[tuple[int, int, int], sp.Expr]:
    A, B, C = sp.symbols("A B C")
    out: dict[tuple[int, int, int], sp.Expr] = {}
    if a:
        out[(a - 1, b + 1, c + 1)] = sp.Integer(2) * a * (C - B)
    if b:
        out[(a + 1, b - 1, c + 1)] = out.get((a + 1, b - 1, c + 1), sp.Integer(0)) + sp.Integer(2) * b * (A - C)
    if c:
        out[(a + 1, b + 1, c - 1)] = out.get((a + 1, b + 1, c - 1), sp.Integer(0)) + sp.Integer(2) * c * (B - A)
    return out


def s07_h08_poisson_matrix() -> dict[str, object]:
    src = degree7_nonorth_basis()
    tgt = degree8_nonorth_basis()
    index = {m: i for i, m in enumerate(tgt)}
    M = sp.zeros(len(tgt), len(src))
    for j, (a, b, c) in enumerate(src):
        image = poisson_h02_on_monomial(a, b, c)
        for mon, coeff in image.items():
            M[index[mon], j] += coeff
    return {
        "source_basis": src,
        "target_basis": tgt,
        "matrix": M,
        "shape": M.shape,
        "nnz": sum(1 for x in M if x != 0),
    }


def s07_h08_poisson_invertibility_certificate() -> dict[str, object]:
    out = s07_h08_poisson_matrix()
    A, B, C = sp.symbols("A B C")
    M = out["matrix"]
    sample = {A: 1, B: 2, C: 4}
    M_sample = sp.Matrix(M.subs(sample))
    det_sample = sp.Integer(M_sample.det())
    return {
        "shape": out["shape"],
        "nnz": out["nnz"],
        "sample_point": sample,
        "sample_rank": M_sample.rank(),
        "sample_det": det_sample,
        "generic_invertibility_proved_by_nonzero_specialization": det_sample != 0,
    }


def main() -> None:
    print(s07_h08_poisson_invertibility_certificate())


if __name__ == "__main__":
    main()
