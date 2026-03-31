#!/usr/bin/env python3

from scripts.paper1_octic_derived_x0_no_rkl import paper1_octic_derived_x0_no_rkl


def test_derived_x0_no_rkl_removes_rkl_symbol() -> None:
    summary = paper1_octic_derived_x0_no_rkl()
    text = str(summary["primitive_total_without_rkl"])
    assert "r_kl" not in text
    assert "k'_{klm}" in text
