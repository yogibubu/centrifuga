#!/usr/bin/env python3
"""Exact checks for the Table V notation block used by H08."""

from __future__ import annotations

from scripts.paper1_h08_tablev_notation import (
    paper1_tablev_notation_ef_blocks,
    paper1_tablev_notation_minimal_dependency_map,
    paper1_tablev_notation_sums,
)


def test_tablev_notation_sums_are_recorded_verbatim() -> None:
    sums = paper1_tablev_notation_sums()
    assert sums["Sigma_tilde"] == "block-diagonal sum"
    assert sums["Sigma_star"] == "sum with resonant terms omitted"


def test_tablev_notation_contains_all_e_and_f_blocks() -> None:
    out = paper1_tablev_notation_ef_blocks()
    assert set(out) == {"E_kl", "E_lk", "E^kl", "E^lk", "F_kl", "F_lk", "F^kl", "F^lk"}
    assert "R'_kl" in str(out["E_kl"].rhs)
    assert "[X_kl,H02]" in str(out["E_kl"].rhs)
    assert "[X^kl,H02]" in str(out["E^kl"].rhs)


def test_tablev_notation_dependency_map_is_minimal_and_exact() -> None:
    dep = paper1_tablev_notation_minimal_dependency_map()
    assert dep["E"] == ("R_prime", "X", "H02")
    assert dep["F"] == ("R_prime", "X", "H02")
    assert dep["primitive_visible_objects"] == ("R_prime", "R_tilde", "X", "H02")
