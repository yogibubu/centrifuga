#!/usr/bin/env python3

from __future__ import annotations

import sympy as sp

import derive_watson_octic_vanvleck as dovv


def _key(jword, coeff, origin=("T",)):
    return ((tuple(), tuple(jword), origin), coeff)


def test_extract_rot_ground_degree_filters_exact_degree():
    expr = dict(
        [
            _key((dovv.Jy,) * 8, sp.Integer(3)),
            _key((dovv.Jy,) * 6, sp.Integer(5)),
            (((("ad", 0),), (dovv.Jy,) * 8, ("V",)), sp.Integer(7)),
        ]
    )
    octic = dovv.extract_rot_ground_degree(expr, 8)
    sextic = dovv.extract_rot_ground_degree(expr, 6)
    assert octic == {((dovv.Jy,) * 8): 3}
    assert sextic == {((dovv.Jy,) * 6): 5}


def test_commuting_projection_degree_collects_all_cartesian_monomials():
    ops = {
        (dovv.Jy,) * 6 + (dovv.Jz,) * 2: sp.Integer(2),
        (dovv.Jz,) * 8: sp.Integer(4),
    }
    proj = dovv.commuting_projection_degree(ops)
    assert proj[(0, 6, 2)] == 2
    assert proj[(0, 0, 8)] == 4
    assert proj[(8, 0, 0)] == 0


def test_linear_scalar_projector_degree8_recovers_exact_basis_coefficients():
    X = sp.Symbol("X")
    basis = dovv.linear_scalar_image_degree(8)
    poly = sp.expand(7 * basis[0] - 3 * basis[1] + 2 * basis[2] + 5 * basis[3] - 11 * basis[4])
    coeffs = dovv.project_linear_scalar_polynomial(poly, degree=8)
    assert sp.simplify(coeffs["L_scalar"] - 7) == 0
    assert sp.simplify(coeffs["H_scalar"] + 3) == 0
    assert sp.simplify(coeffs["D_scalar"] - 2) == 0
    assert sp.simplify(coeffs["B_scalar"] - 5) == 0
    assert sp.simplify(coeffs["const"] + 11) == 0


def test_analyze_effective_series_splits_quartic_sextic_octic():
    k_series = {
        1: {
            (tuple(), (dovv.Jy,) * 4, ("Q",)): sp.Integer(1),
            (tuple(), (dovv.Jy,) * 6, ("S",)): sp.Integer(2),
            (tuple(), (dovv.Jy,) * 8, ("O",)): sp.Integer(3),
        }
    }
    summary = dovv.analyze_effective_series(k_series, max_order=1)
    assert summary["quartic_total"] == {((dovv.Jy,) * 4): 1}
    assert summary["sextic_total"] == {((dovv.Jy,) * 6): 2}
    assert summary["octic_total"] == {((dovv.Jy,) * 8): 3}
