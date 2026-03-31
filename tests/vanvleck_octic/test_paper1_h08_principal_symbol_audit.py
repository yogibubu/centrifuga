#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_h08_principal_symbol_audit import paper1_h08_principal_symbol_audit


def test_visible_h08_has_single_degree8_source_term() -> None:
    out = paper1_h08_principal_symbol_audit()
    assert out["visible_h08_term_degrees"]["RkRlRmRklm"] == 9
    assert out["visible_h08_term_degrees"]["quartic_term"] == 8
    assert out["visible_h08_term_degrees"]["RpkRlRmRplm"] == 7
    assert out["visible_h08_term_degrees"]["RpkRl_comm"] == 6
    assert out["visible_h08_term_degrees"]["Rpk_sq"] == 2
    assert out["visible_principal_symbol_source_terms"] == ("quartic_term",)


def test_all_eq99_completion_terms_are_degree8() -> None:
    out = paper1_h08_principal_symbol_audit()
    assert out["eq99_completion_degrees"]["[S03,[S03,[S03,H02]]]"] == 8
    assert out["eq99_completion_degrees"]["[S03,[S03,H04]]"] == 8
    assert out["eq99_completion_degrees"]["[S03,H06]"] == 8
    assert out["eq99_completion_degrees"]["[S05,H04]"] == 8
    assert out["eq99_completion_degrees"]["[S05,[S03,H02]]"] == 8
    assert out["eq99_completion_degrees"]["[S07,H02]"] == 8
    assert out["all_eq99_completion_terms_are_degree8"] is True
