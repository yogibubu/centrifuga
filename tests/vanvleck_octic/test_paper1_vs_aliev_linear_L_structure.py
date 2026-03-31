#!/usr/bin/env python3

from scripts.paper1_vs_aliev_linear_L_structure import paper1_vs_aliev_linear_L_structure


def test_structure_comparison_exposes_operational_gap() -> None:
    summary = paper1_vs_aliev_linear_L_structure()
    assert "S111" in str(summary["ours_current_compact_linear"])
    assert "D_J" in str(summary["aliev_compact_linear"])
