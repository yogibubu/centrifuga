#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_s07_h08_elimination_basis import elimination_problem_summary


def test_explicit_elimination_bases_are_square() -> None:
    out = elimination_problem_summary()
    assert out["S07_nonorth_dimension"] == 30
    assert out["H08_nonorth_dimension"] == 30
    assert out["elimination_matrix_shape"] == (30, 30)
    assert out["well_posed_square_problem"] is True


def test_degree7_orth_basis_has_expected_six_monomials() -> None:
    out = elimination_problem_summary()
    s7_basis = out["S07_nonorth_basis"]
    assert (7, 0, 0) in s7_basis
    assert (5, 2, 0) in s7_basis
    assert (3, 2, 2) in s7_basis
    assert (1, 4, 2) in s7_basis
    assert (1, 0, 6) in s7_basis
