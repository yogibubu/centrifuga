import sympy as sp

from scripts.octic_asymmetric_top_reduction import (
    cartesian_from_reduced_matrix,
    octic_asymmetric_top_reduction_summary,
    reduced_from_cartesian_pseudoinverse,
    standard_octic_reduced_basis,
    standard_octic_reduced_labels,
)


def test_standard_reduced_basis_has_nine_operators() -> None:
    basis = standard_octic_reduced_basis()
    assert len(basis) == 9
    assert standard_octic_reduced_labels() == (
        "L800",
        "L620",
        "L440",
        "L260",
        "L080",
        "L602",
        "L422",
        "L242",
        "L062",
    )


def test_cartesian_from_reduced_matrix_has_rank_nine() -> None:
    M = cartesian_from_reduced_matrix()
    assert M.shape == (15, 9)
    assert int(M.rank()) == 9


def test_pseudoinverse_is_left_inverse() -> None:
    M = cartesian_from_reduced_matrix()
    P = reduced_from_cartesian_pseudoinverse()
    assert sp.simplify(P * M - sp.eye(9)) == sp.zeros(9)


def test_summary_reports_six_redundant_directions() -> None:
    out = octic_asymmetric_top_reduction_summary()
    assert out["rank"] == 9
    assert out["nullity"] == 6
