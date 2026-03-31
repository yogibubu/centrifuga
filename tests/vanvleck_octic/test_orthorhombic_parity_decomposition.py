#!/usr/bin/env python3
from __future__ import annotations

from scripts.orthorhombic_parity_decomposition import (
    closed_form_orthorhombic_count,
    paper1_sector_count_summary,
)


def test_counts_match_paper1_sector_sizes() -> None:
    out = paper1_sector_count_summary()

    assert out["S03"]["total_count"] == 10
    assert out["S03"]["orthorhombic_count"] == 1
    assert out["S03"]["nonorthorhombic_count"] == 9
    assert out["S03"]["orthorhombic_monomials"] == ((1, 1, 1),)

    assert out["S05"]["total_count"] == 21
    assert out["S05"]["orthorhombic_count"] == 3
    assert out["S05"]["nonorthorhombic_count"] == 18
    assert set(out["S05"]["orthorhombic_monomials"]) == {(3, 1, 1), (1, 3, 1), (1, 1, 3)}

    assert out["S07"]["total_count"] == 36
    assert out["S07"]["orthorhombic_count"] == 6
    assert out["S07"]["nonorthorhombic_count"] == 30
    assert set(out["S07"]["orthorhombic_monomials"]) == {
        (5, 1, 1),
        (1, 5, 1),
        (1, 1, 5),
        (3, 3, 1),
        (3, 1, 3),
        (1, 3, 3),
    }

    assert out["H06"]["total_count"] == 28
    assert out["H06"]["orthorhombic_count"] == 10
    assert out["H06"]["nonorthorhombic_count"] == 18

    assert out["H08"]["total_count"] == 45
    assert out["H08"]["orthorhombic_count"] == 15
    assert out["H08"]["nonorthorhombic_count"] == 30


def test_closed_form_matches_explicit_counts() -> None:
    out = paper1_sector_count_summary()
    for key in ("S03", "S05", "S07", "H06", "H08"):
        degree = out[key]["degree"]
        assert out[key]["orthorhombic_count"] == closed_form_orthorhombic_count(degree)
