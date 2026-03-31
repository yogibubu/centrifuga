#!/usr/bin/env python3
"""Exact checks for the Eq. (99) octic cartesian projection map."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_cartesian_projection import (
    paper1_octic_cartesian_source_coefficients,
    paper1_octic_eq99_prefactors,
    paper1_octic_eq99_term_labels,
    paper1_octic_termwise_linear_decomposition,
    paper1_octic_termwise_symmetric_top_decomposition,
    paper1_octic_total_cartesian_coefficients,
    paper1_octic_total_linear_constant,
    paper1_octic_total_symmetric_top_constants,
)


def test_eq99_term_labels_and_prefactors_are_exact() -> None:
    assert paper1_octic_eq99_term_labels() == (
        "H08",
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
        "S07H02",
    )
    pref = paper1_octic_eq99_prefactors()
    assert pref["H08"] == 1
    assert pref["S03S03S03H02"] == -sp.I / 6
    assert pref["S03S03H04"] == -sp.Rational(1, 2)
    assert pref["S03H06"] == sp.I
    assert pref["S05H04"] == sp.I
    assert pref["S05S03H02"] == -1
    assert pref["S07H02"] == sp.I


def test_total_cartesian_coefficients_are_exact_eq99_sums() -> None:
    src = paper1_octic_cartesian_source_coefficients()
    pref = paper1_octic_eq99_prefactors()
    tot = paper1_octic_total_cartesian_coefficients()
    assert tot["c_080"] == sum(pref[t] * src[t]["c_080"] for t in paper1_octic_eq99_term_labels())
    assert tot["c_800"] == sum(pref[t] * src[t]["c_800"] for t in paper1_octic_eq99_term_labels())


def test_linear_constant_is_exact_sum_of_termwise_contributions() -> None:
    termwise = paper1_octic_termwise_linear_decomposition("a")
    total = paper1_octic_total_linear_constant("a")
    assert sp.simplify(total - sum(termwise.values())) == 0
    assert termwise["H08"] == sp.Symbol("c_080^H08")
    assert termwise["S03S03H04"] == -sp.Symbol("c_080^S03S03H04") / 2


def test_symmetric_top_constants_are_exact_sums_of_termwise_contributions() -> None:
    termwise = paper1_octic_termwise_symmetric_top_decomposition("a")
    total = paper1_octic_total_symmetric_top_constants("a")
    for key in ("LJ4", "LJ3K", "LJ2K2", "LJK3", "LK4"):
        assert sp.simplify(total[key] - sum(termwise[t][key] for t in paper1_octic_eq99_term_labels())) == 0


def test_h08_term_projects_to_expected_first_symmetric_top_constant() -> None:
    termwise = paper1_octic_termwise_symmetric_top_decomposition("a")
    assert termwise["H08"]["LJ4"] == sp.Symbol("c_080^H08")
    assert termwise["H08"]["LJ3K"] == -4 * sp.Symbol("c_080^H08") + sp.Symbol("c_260^H08")
