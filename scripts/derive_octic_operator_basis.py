#!/usr/bin/env python3
"""Exact commuting degree-8 rotational bases.

This file does not use any Gaussian-specific machinery.
It only fixes the operator targets for a rigorous non-linear extension of the
linear optical constant ``L``:

- asymmetric top: full commuting octic basis in ``Jx^2, Jy^2, Jz^2``,
- symmetric top: reduced basis in ``J^2`` and ``J_a^2``,
- linear rotor: single scalar multiplying ``(J_b^2 + J_c^2)^4``.
"""

from __future__ import annotations

import sympy as sp

Jx, Jy, Jz = sp.symbols("Jx Jy Jz")


def asymmetric_top_octic_basis() -> list[sp.Expr]:
    """Return the 15 commuting degree-8 monomials in ``Jx, Jy, Jz``."""
    out: list[sp.Expr] = []
    for a in range(5):
        for b in range(5 - a):
            c = 4 - a - b
            out.append(sp.expand(Jx ** (2 * a) * Jy ** (2 * b) * Jz ** (2 * c)))
    return out


def asymmetric_top_monomial_labels() -> tuple[str, ...]:
    """Return stable labels for the 15 commuting octic monomials."""
    labels: list[str] = []
    for a in range(5):
        for b in range(5 - a):
            c = 4 - a - b
            labels.append(f"c_{2*a}{2*b}{2*c}")
    return tuple(labels)


def asymmetric_top_cartesian_symbols() -> dict[str, sp.Symbol]:
    """Return the 15 independent cartesian octic symbols."""
    return {label: sp.Symbol(label) for label in asymmetric_top_monomial_labels()}


def asymmetric_top_cartesian_polynomial() -> sp.Expr:
    """Return the generic commuting octic cartesian polynomial."""
    coeffs = asymmetric_top_cartesian_symbols()
    expr = sp.Integer(0)
    idx = 0
    for a in range(5):
        for b in range(5 - a):
            c = 4 - a - b
            label = asymmetric_top_monomial_labels()[idx]
            expr += coeffs[label] * sp.expand(Jx ** (2 * a) * Jy ** (2 * b) * Jz ** (2 * c))
            idx += 1
    return sp.expand(expr)


def symmetric_top_octic_basis(axis: str = "a") -> list[sp.Expr]:
    """Return the 5 commuting degree-8 monomials for a symmetric top."""
    axis = axis.lower()
    if axis == "a":
        Jpar = Jx
        Jperp2 = Jy**2 + Jz**2
    elif axis == "b":
        Jpar = Jy
        Jperp2 = Jx**2 + Jz**2
    elif axis == "c":
        Jpar = Jz
        Jperp2 = Jx**2 + Jy**2
    else:
        raise ValueError("axis must be one of 'a', 'b', 'c'")

    J2 = sp.expand(Jpar**2 + Jperp2)
    return [
        sp.expand(J2**4),
        sp.expand(J2**3 * Jpar**2),
        sp.expand(J2**2 * Jpar**4),
        sp.expand(J2 * Jpar**6),
        sp.expand(Jpar**8),
    ]


def _poly_monomial_vector(expr: sp.Expr, monoms: list[tuple[int, int, int]]) -> sp.Matrix:
    poly = sp.Poly(sp.expand(expr), Jx, Jy, Jz)
    return sp.Matrix([poly.coeff_monomial(m) for m in monoms])


def symmetric_top_projection_matrix(axis: str = "a") -> sp.Matrix:
    """Return the exact 15x5 projection matrix from the symmetric-top basis.

    Columns are the 5 symmetric-top operators expressed on the 15 commuting
    degree-8 asymmetric-top monomials.
    """
    asym = asymmetric_top_octic_basis()
    sym = symmetric_top_octic_basis(axis)
    monoms = sorted(
        {
            m
            for expr in asym
            for m in sp.Poly(sp.expand(expr), Jx, Jy, Jz).monoms()
        }
    )
    return sp.Matrix.hstack(*[_poly_monomial_vector(expr, monoms) for expr in sym])


def extract_asymmetric_top_cartesian_coefficients(expr: sp.Expr) -> dict[str, sp.Expr]:
    """Extract the 15 commuting octic cartesian coefficients from an expression."""
    poly = sp.Poly(sp.expand(expr), Jx, Jy, Jz)
    out: dict[str, sp.Expr] = {}
    idx = 0
    for a in range(5):
        for b in range(5 - a):
            c = 4 - a - b
            out[asymmetric_top_monomial_labels()[idx]] = sp.simplify(poly.coeff_monomial(Jx ** (2 * a) * Jy ** (2 * b) * Jz ** (2 * c)))
            idx += 1
    return out


def project_asymmetric_cartesian_to_symmetric_top(cart: dict[str, sp.Expr], axis: str = "a") -> dict[str, sp.Expr]:
    """Project a cartesian octic polynomial to symmetric-top constants.

    The input must already satisfy the b<->c symmetric-top image constraints.
    """
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis symmetric-top case is derived explicitly.")
    req = asymmetric_top_monomial_labels()
    missing = [label for label in req if label not in cart]
    if missing:
        raise KeyError(f"Missing cartesian octic coefficients: {missing}")

    c400 = cart["c_800"]
    c310 = cart["c_620"]
    c220 = cart["c_440"]
    c130 = cart["c_260"]
    c040 = cart["c_080"]
    return {
        "LJ4": sp.simplify(c040),
        "LJ3K": sp.simplify(-4 * c040 + c130),
        "LJ2K2": sp.simplify(6 * c040 - 3 * c130 + c220),
        "LJK3": sp.simplify(-4 * c040 + 3 * c130 - 2 * c220 + c310),
        "LK4": sp.simplify(c040 - c130 + c220 - c310 + c400),
    }


def project_asymmetric_cartesian_to_linear(cart: dict[str, sp.Expr], axis: str = "a") -> sp.Expr:
    """Return the exact linear octic constant from cartesian coefficients."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis linear case is derived explicitly.")
    return sp.simplify(cart["c_080"])


def octic_operator_from_cartesian_coefficients(cart: dict[str, sp.Expr]) -> sp.Expr:
    """Build the general commuting octic operator from the 15 cartesian coefficients."""
    expr = sp.Integer(0)
    idx = 0
    labels = asymmetric_top_monomial_labels()
    for a in range(5):
        for b in range(5 - a):
            c = 4 - a - b
            expr += cart[labels[idx]] * Jx ** (2 * a) * Jy ** (2 * b) * Jz ** (2 * c)
            idx += 1
    return sp.expand(expr)


def derive_octic_projection_summary(axis: str = "a") -> dict[str, object]:
    """Return the exact projection hierarchy for the general octic operator."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis hierarchy is derived explicitly.")
    cart = asymmetric_top_cartesian_symbols()
    sym = project_asymmetric_cartesian_to_symmetric_top(cart, axis)
    lin = project_asymmetric_cartesian_to_linear(cart, axis)
    return {
        "cartesian_labels": asymmetric_top_monomial_labels(),
        "cartesian_operator": octic_operator_from_cartesian_coefficients(cart),
        "symmetric_top_constants": sym,
        "linear_constant": lin,
        "symmetric_top_constraints": symmetric_top_cartesian_constraints(axis),
        "linear_constraints": linear_cartesian_constraints(axis),
    }


def prove_linear_limit_from_symmetric_top(axis: str = "a") -> dict[str, sp.Expr]:
    """Return the exact symmetric-top -> linear collapse identities."""
    axis = axis.lower()
    if axis == "a":
        Jpar = Jx
        J2 = Jx**2 + Jy**2 + Jz**2
        Jperp2 = Jy**2 + Jz**2
    elif axis == "b":
        Jpar = Jy
        J2 = Jx**2 + Jy**2 + Jz**2
        Jperp2 = Jx**2 + Jz**2
    elif axis == "c":
        Jpar = Jz
        J2 = Jx**2 + Jy**2 + Jz**2
        Jperp2 = Jx**2 + Jy**2
    else:
        raise ValueError("axis must be one of 'a', 'b', 'c'")

    basis = symmetric_top_octic_basis(axis)
    substitutions = {J2: Jperp2, Jpar: sp.Integer(0)}
    collapsed = [sp.expand(expr.subs(substitutions)) for expr in basis]
    return {
        "basis_1": collapsed[0],
        "basis_2": collapsed[1],
        "basis_3": collapsed[2],
        "basis_4": collapsed[3],
        "basis_5": collapsed[4],
        "linear_operator": sp.expand(Jperp2**4),
    }


def symmetric_top_coefficient_symbols(axis: str = "a") -> dict[str, sp.Symbol]:
    """Return the independent cartesian coefficients under b<->c symmetry."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis symmetric-top case is derived explicitly.")
    names = (
        "c400",  # J_a^8
        "c310",  # J_a^6 J_b^2 = J_a^6 J_c^2
        "c220",  # J_a^4 J_b^4 = J_a^4 J_c^4
        "c211",  # J_a^4 J_b^2 J_c^2
        "c130",  # J_a^2 J_b^6 = J_a^2 J_c^6
        "c121",  # J_a^2 J_b^4 J_c^2 = J_a^2 J_b^2 J_c^4
        "c040",  # J_b^8 = J_c^8
        "c031",  # J_b^6 J_c^2 = J_b^2 J_c^6
        "c022",  # J_b^4 J_c^4
    )
    return {name: sp.Symbol(name) for name in names}


def symmetric_top_cartesian_polynomial(axis: str = "a") -> sp.Expr:
    """Return the generic b<->c symmetric commuting octic polynomial."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis symmetric-top case is derived explicitly.")
    c = symmetric_top_coefficient_symbols(axis)
    Ja = Jx
    Jb = Jy
    Jc = Jz
    return sp.expand(
        c["c400"] * Ja**8
        + c["c310"] * (Ja**6 * Jb**2 + Ja**6 * Jc**2)
        + c["c220"] * (Ja**4 * Jb**4 + Ja**4 * Jc**4)
        + c["c211"] * Ja**4 * Jb**2 * Jc**2
        + c["c130"] * (Ja**2 * Jb**6 + Ja**2 * Jc**6)
        + c["c121"] * (Ja**2 * Jb**4 * Jc**2 + Ja**2 * Jb**2 * Jc**4)
        + c["c040"] * (Jb**8 + Jc**8)
        + c["c031"] * (Jb**6 * Jc**2 + Jb**2 * Jc**6)
        + c["c022"] * Jb**4 * Jc**4
    )


def solve_symmetric_top_octic_projection(axis: str = "a") -> dict[str, sp.Expr]:
    """Solve the exact cartesian -> symmetric-top coefficient map on the image."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis symmetric-top case is derived explicitly.")

    LJ4, LJ3K, LJ2K2, LJK3, LK4 = sp.symbols("LJ4 LJ3K LJ2K2 LJK3 LK4")
    Ja = Jx
    J2 = Jx**2 + Jy**2 + Jz**2
    basis = [
        sp.expand(J2**4),
        sp.expand(J2**3 * Ja**2),
        sp.expand(J2**2 * Ja**4),
        sp.expand(J2 * Ja**6),
        sp.expand(Ja**8),
    ]
    combo = sp.expand(LJ4 * basis[0] + LJ3K * basis[1] + LJ2K2 * basis[2] + LJK3 * basis[3] + LK4 * basis[4])
    monoms = {
        "c400": Jx**8,
        "c310": Jx**6 * Jy**2,
        "c220": Jx**4 * Jy**4,
        "c211": Jx**4 * Jy**2 * Jz**2,
        "c130": Jx**2 * Jy**6,
        "c121": Jx**2 * Jy**4 * Jz**2,
        "c040": Jy**8,
        "c031": Jy**6 * Jz**2,
        "c022": Jy**4 * Jz**4,
    }
    poly = sp.Poly(combo, Jx, Jy, Jz)
    image = {name: sp.simplify(poly.coeff_monomial(monom)) for name, monom in monoms.items()}

    solve_eqs = [
        sp.Eq(sp.Symbol("c400"), image["c400"]),
        sp.Eq(sp.Symbol("c310"), image["c310"]),
        sp.Eq(sp.Symbol("c220"), image["c220"]),
        sp.Eq(sp.Symbol("c130"), image["c130"]),
        sp.Eq(sp.Symbol("c040"), image["c040"]),
    ]
    sol = sp.solve(solve_eqs, [LJ4, LJ3K, LJ2K2, LJK3, LK4], dict=True)
    if len(sol) != 1:
        raise RuntimeError(f"Unexpected number of image-solve solutions: {len(sol)}")
    return {
        "basis_symbols": (LJ4, LJ3K, LJ2K2, LJK3, LK4),
        "image": image,
        "inverse_on_image": sol[0],
    }


def symmetric_top_cartesian_constraints(axis: str = "a") -> dict[str, sp.Expr]:
    """Return the exact four cartesian constraints defining the 5-d image."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis symmetric-top case is derived explicitly.")
    c400, c310, c220, c211, c130, c121, c040, c031, c022 = sp.symbols(
        "c400 c310 c220 c211 c130 c121 c040 c031 c022"
    )
    solved = solve_symmetric_top_octic_projection(axis)
    inv = solved["inverse_on_image"]
    image = solved["image"]
    back = {key: sp.simplify(expr.subs(inv)) for key, expr in image.items()}
    return {
        "constraint_c211": sp.simplify(back["c211"] - c211),
        "constraint_c121": sp.simplify(back["c121"] - c121),
        "constraint_c031": sp.simplify(back["c031"] - c031),
        "constraint_c022": sp.simplify(back["c022"] - c022),
    }


def linear_constant_from_symmetric_top(axis: str = "a") -> sp.Expr:
    """Return the exact linear-limit coefficient in terms of symmetric-top constants."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis symmetric-top case is derived explicitly.")
    LJ4, LJ3K, LJ2K2, LJK3, LK4 = sp.symbols("LJ4 LJ3K LJ2K2 LJK3 LK4")
    out = prove_linear_limit_from_symmetric_top(axis)
    expr = sp.expand(
        LJ4 * out["basis_1"]
        + LJ3K * out["basis_2"]
        + LJ2K2 * out["basis_3"]
        + LJK3 * out["basis_4"]
        + LK4 * out["basis_5"]
    )
    linear = out["linear_operator"]
    ratio = sp.simplify(expr / linear)
    return ratio


def linear_cartesian_constraints(axis: str = "a") -> dict[str, sp.Expr]:
    """Return the exact cartesian constraints defining the 1-d linear image."""
    axis = axis.lower()
    if axis != "a":
        raise NotImplementedError("Only the a-axis linear case is derived explicitly.")
    c = asymmetric_top_cartesian_symbols()
    L = sp.Symbol("L")
    linear_coeffs = {
        "c_008": L,
        "c_026": 4 * L,
        "c_044": 6 * L,
        "c_062": 4 * L,
        "c_080": L,
        "c_206": 0,
        "c_224": 0,
        "c_242": 0,
        "c_260": 0,
        "c_404": 0,
        "c_422": 0,
        "c_440": 0,
        "c_602": 0,
        "c_620": 0,
        "c_800": 0,
    }
    return {
        label: sp.simplify(c[label] - linear_coeffs[label]) for label in asymmetric_top_monomial_labels()
    }


def linear_octic_operator(axis: str = "a") -> sp.Expr:
    """Return the unique commuting degree-8 linear-rotor octic operator."""
    axis = axis.lower()
    if axis == "a":
        Jperp2 = Jy**2 + Jz**2
    elif axis == "b":
        Jperp2 = Jx**2 + Jz**2
    elif axis == "c":
        Jperp2 = Jx**2 + Jy**2
    else:
        raise ValueError("axis must be one of 'a', 'b', 'c'")
    return sp.expand(Jperp2**4)


def linear_octic_binomial_coefficients(axis: str = "a") -> dict[sp.Expr, sp.Expr]:
    """Return the exact monomial coefficients of the linear octic operator."""
    expr = linear_octic_operator(axis)
    poly = sp.Poly(expr, Jx, Jy, Jz)
    return {
        sp.expand(Jx ** a * Jy ** b * Jz ** c): sp.Integer(coeff)
        for (a, b, c), coeff in poly.terms()
    }


def basis_dimensions() -> dict[str, int]:
    """Return the exact commuting-basis dimensions by rotor class."""
    return {
        "asymmetric_top": len(asymmetric_top_octic_basis()),
        "symmetric_top": len(symmetric_top_octic_basis()),
        "linear": 1,
    }


def main() -> None:
    print("basis dimensions:", basis_dimensions())
    print("linear operator (axis a):")
    sp.pprint(linear_octic_operator("a"))
    print("symmetric-top projection matrix shape:", symmetric_top_projection_matrix("a").shape)
    print("linear constant from symmetric-top basis:")
    sp.pprint(linear_constant_from_symmetric_top("a"))


if __name__ == "__main__":
    main()
