#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_principal_symbol_projection import (
    paper1_octic_principal_cartesian_source_coefficients,
    paper1_octic_principal_linear_constant,
    paper1_octic_principal_orth_coefficients,
    paper1_octic_principal_prefactors,
    paper1_octic_principal_reduced_linear_constant,
    paper1_octic_principal_reduced_term_labels,
    paper1_octic_principal_symmetric_top_constants,
    paper1_octic_principal_term_labels,
    paper1_octic_principal_termwise_linear_decomposition,
    paper1_octic_principal_total_cartesian_coefficients,
)


def test_principal_term_labels_and_prefactors() -> None:
    assert paper1_octic_principal_term_labels() == (
        "H08_k4",
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
        "S07H02",
    )
    pref = paper1_octic_principal_prefactors()
    assert pref["H08_k4"] == 1
    assert pref["S03S03H04"] == -sp.Rational(1, 2)


def test_principal_total_cartesian_coefficients_are_exact_sums() -> None:
    src = paper1_octic_principal_cartesian_source_coefficients()
    pref = paper1_octic_principal_prefactors()
    tot = paper1_octic_principal_total_cartesian_coefficients()
    assert tot["c_080"] == sum(pref[t] * src[t]["c_080"] for t in paper1_octic_principal_term_labels())


def test_principal_orth_coefficients_have_15_entries() -> None:
    orth = paper1_octic_principal_orth_coefficients()
    assert len(orth) == 15
    assert "c_080" in orth
    assert "c_800" in orth


def test_principal_linear_constant_is_termwise_sum() -> None:
    termwise = paper1_octic_principal_termwise_linear_decomposition("a")
    total = paper1_octic_principal_linear_constant("a")
    assert sp.simplify(total - sum(termwise.values())) == 0
    assert termwise["H08_k4"] == sp.Symbol("c_080^H08_k4")


def test_principal_symmetric_top_constants_exist() -> None:
    total = paper1_octic_principal_symmetric_top_constants("a")
    assert set(total) == {"LJ4", "LJ3K", "LJ2K2", "LJK3", "LK4"}


def test_reduced_principal_source_excludes_s07_block() -> None:
    labels = paper1_octic_principal_reduced_term_labels()
    assert "S07H02" not in labels
    assert len(labels) == 6


def test_reduced_linear_constant_has_no_direct_s07_symbol() -> None:
    expr = paper1_octic_principal_reduced_linear_constant("a")
    assert "S07H02" not in str(expr)
