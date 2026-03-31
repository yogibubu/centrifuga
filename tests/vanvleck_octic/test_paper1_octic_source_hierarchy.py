#!/usr/bin/env python3
"""Exact checks for the printed octic source hierarchy of Paper 1."""

from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_source_hierarchy import (
    paper1_octic_dependency_statement,
    paper1_octic_source_blocks,
    paper1_tilde_h08_structure,
)


def test_paper1_tilde_h08_contains_exact_seven_source_terms() -> None:
    out = paper1_tilde_h08_structure()
    expr = out["tilde_H08"]
    terms = sp.Add.make_args(expr)
    assert len(terms) == 7


def test_paper1_tilde_h08_has_exact_printed_prefactors() -> None:
    out = paper1_tilde_h08_structure()
    terms = out["terms"]
    assert terms[0] == out["H08"]
    assert terms[1].coeff(sp.I) == -sp.Symbol("[S03,[S03,[S03,H02]]]", commutative=False) / 6
    assert terms[2] == -sp.Symbol("[S03,[S03,H04]]", commutative=False) / 2
    assert terms[3] == sp.I * sp.Symbol("[S03,H06]", commutative=False)
    assert terms[4] == sp.I * sp.Symbol("[S05,H04]", commutative=False)
    assert terms[5] == -sp.Symbol("[S05,[S03,H02]]", commutative=False)
    assert terms[6] == sp.I * sp.Symbol("[S07,H02]", commutative=False)


def test_paper1_octic_dependency_statement_is_recorded_verbatim() -> None:
    text = paper1_octic_dependency_statement()["paper1_statement"]
    assert "linearly on the quartic potential" in text
    assert "quadratically on the cubic potential constants" in text


def test_paper1_octic_source_blocks_are_complete() -> None:
    out = paper1_octic_source_blocks()
    assert out["bare_block"] == ("H08",)
    assert len(out["block_diagonal_completion"]) == 6
    assert out["lower_order_inputs"] == ("H02", "H04", "H06", "S03", "S05", "S07")
