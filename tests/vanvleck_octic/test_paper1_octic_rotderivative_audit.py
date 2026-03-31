#!/usr/bin/env python3
"""Exact checks for the octic rotational-derivative dependency audit."""

from __future__ import annotations

from scripts.paper1_octic_rotderivative_audit import (
    paper1_octic_mu2_verdict,
    paper1_rotderivative_dependency_map,
)


def test_eq101_side_objects_do_not_introduce_mu2() -> None:
    dep = paper1_rotderivative_dependency_map()
    assert "mu2" not in dep["R_kl"].rot
    assert "mu2" not in dep["R'_kl"].rot
    assert "mu2" not in dep["X"].rot
    assert "mu2" not in dep["U"].rot
    assert "mu2" not in dep["V"].rot
    assert "mu2" not in dep["E"].rot
    assert "mu2" not in dep["F"].rot


def test_block_diagonal_completion_terms_do_not_introduce_mu2() -> None:
    dep = paper1_rotderivative_dependency_map()
    for key in (
        "term_S03S03S03H02",
        "term_S03S03H04",
        "term_S03H06",
        "term_S05H04",
        "term_S05S03H02",
        "term_S07H02",
    ):
        assert "mu2" not in dep[key].rot


def test_tilde_h08_has_no_mu2_under_sextic_theorem() -> None:
    out = paper1_octic_mu2_verdict()
    assert out["mu2_in_tilde_H08"] is False
    assert out["rotational_derivative_support_tilde_H08"] == ("mu1",)
    assert out["assumption"] == "H06 depends on mu1 but not mu2"
