import sympy as sp

from scripts.octic_linear_k0_quantum_projection import (
    derive_plane_monomial_projection_weights_x_axis,
    project_x_axis_linear_polynomial,
)


def test_plane_weights_reconstruct_exact_linear_operator_x_axis() -> None:
    coeffs = {
        (0, 8, 0): sp.Integer(1),
        (0, 6, 2): sp.Integer(4),
        (0, 4, 4): sp.Integer(6),
        (0, 2, 6): sp.Integer(4),
        (0, 0, 8): sp.Integer(1),
    }
    poly = project_x_axis_linear_polynomial(coeffs)
    assert sp.simplify(poly["X4"] - 1) == 0
    assert sp.simplify(poly["X3"] - 4) == 0
    assert sp.simplify(poly["X2"] + 14) == 0
    assert sp.simplify(poly["X1"] - 12) == 0


def test_x_axis_plane_x4_weights_are_exact() -> None:
    weights = derive_plane_monomial_projection_weights_x_axis()
    assert weights[(0, 8, 0)]["X4"] == sp.Rational(35, 128)
    assert weights[(0, 6, 2)]["X4"] == sp.Rational(5, 128)
    assert weights[(0, 4, 4)]["X4"] == sp.Rational(3, 128)
    assert weights[(0, 2, 6)]["X4"] == sp.Rational(5, 128)
    assert weights[(0, 0, 8)]["X4"] == sp.Rational(35, 128)
