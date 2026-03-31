#!/usr/bin/env python3
"""Octic extensions of the symbolic Van Vleck BCH engine.

This module does not reimplement the BCH machinery. It lifts the existing
quartic engine to generic rotational-ground extraction at degrees 4, 6, and 8,
plus exact commuting and linear-scalar projections needed for the linear-octic
branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections import defaultdict
from typing import Dict, Iterable, Tuple

import sympy as sp

import derive_watson_quartic_vanvleck as qvv


JWord = qvv.JWord
Key = qvv.Key
Series = qvv.Series

Jx, Jy, Jz = qvv.Jx, qvv.Jy, qvv.Jz


def _normal_order_expr_unpruned(expr: Dict[Key, sp.Expr]) -> Dict[Key, sp.Expr]:
    out: Dict[Key, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (vword, jword, origin), coeff in expr.items():
        for nvword, prefactor in qvv.normal_order_word(vword):
            out[(nvword, jword, origin)] += coeff * prefactor
    return {key: sp.simplify(val) for key, val in out.items() if sp.simplify(val) != 0}


def extract_rot_ground_degree(expr: Dict[Key, sp.Expr], degree: int) -> Dict[JWord, sp.Expr]:
    """Extract the rotational ground block at fixed rotational degree."""
    out: Dict[JWord, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for (vword, jword, _origin), coeff in _normal_order_expr_unpruned(expr).items():
        if len(vword) == 0 and len(jword) == degree:
            out[jword] += coeff
    return {jword: sp.simplify(coeff) for jword, coeff in out.items() if sp.simplify(coeff) != 0}


def extract_rot_ground_degree_by_origin(
    expr: Dict[Key, sp.Expr], degree: int
) -> Dict[qvv.Origin, Dict[JWord, sp.Expr]]:
    """Extract the rotational ground block at fixed degree, grouped by source origin."""
    out: Dict[qvv.Origin, Dict[JWord, sp.Expr]] = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for (vword, jword, origin), coeff in _normal_order_expr_unpruned(expr).items():
        if len(vword) == 0 and len(jword) == degree:
            out[qvv.normalize_origin(origin)][jword] += coeff
    cleaned: Dict[qvv.Origin, Dict[JWord, sp.Expr]] = {}
    for origin, terms in out.items():
        reduced = {jw: sp.simplify(c) for jw, c in terms.items() if sp.simplify(c) != 0}
        if reduced:
            cleaned[origin] = reduced
    return cleaned


def commuting_projection_degree(rot_ops: Dict[JWord, sp.Expr]) -> Dict[Tuple[int, int, int], sp.Expr]:
    """Project ordered non-commuting rotational monomials to commuting Cartesian monomials."""
    x, y, z = sp.symbols("x y z", real=True)
    repl = {Jx: x, Jy: y, Jz: z}

    poly = sp.Integer(0)
    for jword, coeff in rot_ops.items():
        mon = sp.Integer(1)
        for jop in jword:
            mon *= repl[jop]
        poly += coeff * mon

    p = sp.Poly(sp.expand(poly), x, y, z)
    out: Dict[Tuple[int, int, int], sp.Expr] = {}
    total_degree = next(iter(rot_ops.keys())).__len__() if rot_ops else 0
    for i in range(total_degree + 1):
        for j in range(total_degree + 1 - i):
            k = total_degree - i - j
            out[(i, j, k)] = sp.simplify(p.coeff_monomial(x**i * y**j * z**k))
    return out


def commuting_polynomial_degree(rot_ops: Dict[JWord, sp.Expr]) -> sp.Expr:
    """Return the full commuting polynomial image of a rotational block."""
    x, y, z = sp.symbols("x y z", real=True)
    repl = {Jx: x, Jy: y, Jz: z}
    poly = sp.Integer(0)
    for jword, coeff in rot_ops.items():
        mon = sp.Integer(1)
        for jop in jword:
            mon *= repl[jop]
        poly += coeff * mon
    return sp.expand(poly)


def linear_scalar_image_degree(degree: int) -> tuple[sp.Expr, ...]:
    """Exact K=0 images of the scalar hierarchy at fixed rotational degree."""
    X = sp.Symbol("X")
    if degree == 4:
        return (X**2 - X,)
    if degree == 6:
        return (X**3, X**2 - X, X, sp.Integer(1))
    if degree == 8:
        return (X**4 + 4 * X**3 - 14 * X**2 + 12 * X, X**3, X**2, X, sp.Integer(1))
    raise ValueError("degree must be one of 4, 6, 8")


def linear_scalar_projector_degree(degree: int) -> sp.Matrix:
    """Exact projector from X-polynomial coefficients to scalar hierarchy coefficients."""
    X = sp.Symbol("X")
    basis = linear_scalar_image_degree(degree)
    max_pow = degree // 2
    m = sp.Matrix([[sp.expand(b).coeff(X, i) for b in basis] for i in range(max_pow + 1)])
    return sp.simplify(m.inv())


def project_linear_scalar_polynomial(poly_x: sp.Expr, degree: int) -> dict[str, sp.Expr]:
    """Project a scalar X-polynomial onto the exact linear hierarchy basis."""
    X = sp.Symbol("X")
    max_pow = degree // 2
    vec = sp.Matrix([sp.expand(poly_x).coeff(X, i) for i in range(max_pow + 1)])
    coeffs = sp.simplify(linear_scalar_projector_degree(degree) * vec)
    labels_by_degree = {
        4: ("D_scalar", "B_scalar", "H2_scalar"),
        6: ("H_scalar", "D_scalar", "B_scalar", "const"),
        8: ("L_scalar", "H_scalar", "D_scalar", "B_scalar", "const"),
    }
    labels = labels_by_degree[degree]
    return {label: sp.simplify(coeffs[i, 0]) for i, label in enumerate(labels)}


def analyze_effective_series(
    k_series: Series,
    max_order: int,
) -> dict[str, object]:
    """Extract quartic, sextic, and octic rotational-ground content from a BCH output."""
    total_by_degree: dict[int, Dict[JWord, sp.Expr]] = {}
    by_order_and_degree: dict[int, dict[int, Dict[JWord, sp.Expr]]] = {}
    for degree in (4, 6, 8):
        total_by_degree[degree] = defaultdict(lambda: sp.Integer(0))
    for order in range(1, max_order + 1):
        by_order_and_degree[order] = {}
        for degree in (4, 6, 8):
            block = extract_rot_ground_degree(k_series.get(order, {}), degree)
            by_order_and_degree[order][degree] = block
            for jword, coeff in block.items():
                total_by_degree[degree][jword] += coeff
    total_by_degree = {
        degree: {jw: sp.simplify(c) for jw, c in terms.items() if sp.simplify(c) != 0}
        for degree, terms in total_by_degree.items()
    }
    return {
        "by_order_and_degree": by_order_and_degree,
        "quartic_total": total_by_degree[4],
        "sextic_total": total_by_degree[6],
        "octic_total": total_by_degree[8],
        "quartic_commuting": commuting_polynomial_degree(total_by_degree[4]),
        "sextic_commuting": commuting_polynomial_degree(total_by_degree[6]),
        "octic_commuting": commuting_polynomial_degree(total_by_degree[8]),
    }


def summarize_linear_scalar_branch(k_series: Series, max_order: int) -> dict[str, object]:
    """Return the extracted rotational-ground content.

    The exact projection to the linear scalar hierarchy must be applied only
    after an independent reduction to the scalar branch. This function therefore
    stops at the commuting sextic/octic polynomials and does not hide any
    additional projection.
    """
    summary = analyze_effective_series(k_series, max_order=max_order)
    return summary


__all__ = [
    "extract_rot_ground_degree",
    "extract_rot_ground_degree_by_origin",
    "commuting_projection_degree",
    "commuting_polynomial_degree",
    "linear_scalar_image_degree",
    "linear_scalar_projector_degree",
    "project_linear_scalar_polynomial",
    "analyze_effective_series",
    "summarize_linear_scalar_branch",
]
