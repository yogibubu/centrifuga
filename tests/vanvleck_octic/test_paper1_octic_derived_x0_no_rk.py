#!/usr/bin/env python3

from scripts.paper1_octic_derived_x0_no_rk import paper1_octic_derived_x0_no_rk


def test_derived_x0_no_rk_removes_rk_symbol() -> None:
    summary = paper1_octic_derived_x0_no_rk()
    text = str(summary["primitive_total_without_rk"])
    assert "Σ_l C_l r_kl" in text
    assert "r_k**2" not in text
