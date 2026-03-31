#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_octic_derived_x0_primitives import paper1_octic_derived_x0_primitives


def test_derived_x0_primitives_removes_intermediate_qws_symbols() -> None:
    summary = paper1_octic_derived_x0_primitives()
    total = summary["primitive_total_so_far"]
    text = str(total)
    for bad in ["q004", "q022", "q040", "w006", "w024", "w042", "w060", "s113", "s131", "rho_k"]:
        assert bad not in text
    assert "r_k" in text
