#!/usr/bin/env python3
"""Structural checks for the visible bare H08 entry of Table V."""

from __future__ import annotations

from scripts.paper1_h08_tablev_structure import (
    paper1_h08_tablev_exact_visible_statement,
    paper1_h08_tablev_leading_terms,
    paper1_h08_tablev_channel_structure,
    paper1_h08_tablev_minimal_statement,
    paper1_h08_tablev_visible_dependencies,
)


def test_h08_tablev_contains_uv_block_explicitly() -> None:
    ch = paper1_h08_tablev_channel_structure()
    assert len(ch["uv_block"]) == 1
    assert "U_km U_lm" in ch["uv_block"][0]
    assert "V_km V_lm" in ch["uv_block"][0]


def test_h08_tablev_visible_dependencies_include_expected_objects() -> None:
    dep = paper1_h08_tablev_visible_dependencies()
    assert dep["depends_on"] == ("k3", "k4", "R", "X", "F", "U", "V", "B", "zeta", "omega")
    assert dep["not_visibly_needed"] == ("mu3",)


def test_h08_tablev_statement_is_nonambiguous() -> None:
    text = paper1_h08_tablev_minimal_statement()
    assert "bare H08 entry of Table V" in text
    assert "Eq. (99) block-diagonal completion" in text


def test_h08_tablev_leading_terms_are_recorded() -> None:
    out = paper1_h08_tablev_leading_terms()
    assert "term1_rrrr" in out
    assert "term2_rrrrk4" in out
    assert "term3_rprime_rr_rprime" in out
    assert "term4_rprime_r_comm" in out
    assert "term5_rprime_sq" in out
    assert "visible_h08_formula" in out


def test_h08_first_line_is_declared_fully_legible() -> None:
    text = paper1_h08_tablev_exact_visible_statement()
    assert "fully legible" in text
    assert "exactly five terms" in text
