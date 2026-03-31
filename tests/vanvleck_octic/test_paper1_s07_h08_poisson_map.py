#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_s07_h08_poisson_map import (
    poisson_h02_on_monomial,
    s07_h08_poisson_invertibility_certificate,
    s07_h08_poisson_matrix,
)


def test_poisson_formula_on_sample_monomial() -> None:
    image = poisson_h02_on_monomial(3, 2, 2)
    assert image[(2, 3, 3)] != 0
    assert image[(4, 1, 3)] != 0
    assert image[(4, 3, 1)] != 0


def test_s07_h08_poisson_matrix_is_square() -> None:
    out = s07_h08_poisson_matrix()
    assert out["shape"] == (30, 30)
    assert out["nnz"] == 66


def test_s07_h08_poisson_matrix_is_generically_invertible() -> None:
    out = s07_h08_poisson_invertibility_certificate()
    assert out["sample_rank"] == 30
    assert out["sample_det"] != 0
    assert out["generic_invertibility_proved_by_nonzero_specialization"] is True
