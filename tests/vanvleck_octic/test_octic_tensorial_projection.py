import sympy as sp

from scripts.octic_tensorial_projection import (
    axial_octic_basis,
    axial_octic_embedding_matrix,
    axial_octic_from_cartesian_matrix,
    linear_from_axial_vector,
    octic_tensorial_projection_summary,
)


def test_axial_octic_basis_has_five_elements() -> None:
    basis = axial_octic_basis("a")
    assert len(basis) == 5


def test_axial_embedding_has_rank_five() -> None:
    M = axial_octic_embedding_matrix("a")
    assert M.shape == (15, 5)
    assert int(M.rank()) == 5


def test_projection_is_left_inverse_on_axial_image() -> None:
    M = axial_octic_embedding_matrix("a")
    P = axial_octic_from_cartesian_matrix("a")
    assert sp.simplify(P * M - sp.eye(5)) == sp.zeros(5)


def test_linear_limit_keeps_only_x0() -> None:
    v = linear_from_axial_vector()
    assert v == sp.Matrix([[1, 0, 0, 0, 0]])


def test_summary_is_consistent() -> None:
    out = octic_tensorial_projection_summary("a")
    assert out["rank"] == 5
    assert out["left_inverse_exact"] is True
