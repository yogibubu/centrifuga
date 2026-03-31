#!/usr/bin/env python3
"""Exact checks for octic representation linear maps and pseudoinverses."""

from __future__ import annotations

import sympy as sp

from scripts.octic_representation_linear_maps import (
    octic_cartesian_vector,
    octic_linear_embedding_matrix,
    octic_linear_embedding_pseudoinverse,
    octic_linear_from_cartesian_matrix,
    octic_linear_image_vector,
    octic_linear_pseudoinverse,
    octic_symmetric_top_from_cartesian_matrix,
    octic_symmetric_top_pseudoinverse,
)


def test_symmetric_top_map_has_shape_5x15() -> None:
    M = octic_symmetric_top_from_cartesian_matrix("a")
    assert M.shape == (5, 15)


def test_linear_map_has_shape_1x15() -> None:
    M = octic_linear_from_cartesian_matrix("a")
    assert M.shape == (1, 15)


def test_symmetric_top_pseudoinverse_is_exact_right_inverse() -> None:
    M = octic_symmetric_top_from_cartesian_matrix("a")
    P = octic_symmetric_top_pseudoinverse("a")
    assert sp.simplify(M * P - sp.eye(5)) == sp.zeros(5, 5)


def test_linear_pseudoinverse_is_exact_right_inverse() -> None:
    M = octic_linear_from_cartesian_matrix("a")
    P = octic_linear_pseudoinverse("a")
    assert sp.simplify(M * P - sp.eye(1)) == sp.zeros(1, 1)


def test_linear_embedding_has_exact_left_inverse() -> None:
    E = octic_linear_embedding_matrix("a")
    Pinv = octic_linear_embedding_pseudoinverse("a")
    assert sp.simplify(Pinv * E - sp.eye(1)) == sp.zeros(1, 1)


def test_linear_image_vector_is_recovered_from_linear_embedding() -> None:
    L = sp.Symbol("L")
    M = octic_linear_embedding_matrix("a")
    P = octic_linear_embedding_pseudoinverse("a")
    vec = octic_linear_image_vector("a") * L
    assert sp.simplify(P * vec - sp.Matrix([L])) == sp.zeros(1, 1)
    assert sp.simplify(M * sp.Matrix([L]) - sp.Matrix(vec)) == sp.zeros(15, 1)


def test_cartesian_to_symmetric_to_cartesian_is_projector() -> None:
    M = octic_symmetric_top_from_cartesian_matrix("a")
    P = octic_symmetric_top_pseudoinverse("a")
    C = octic_cartesian_vector()
    proj = sp.simplify(P * (M * C))
    assert proj.shape == (15, 1)
