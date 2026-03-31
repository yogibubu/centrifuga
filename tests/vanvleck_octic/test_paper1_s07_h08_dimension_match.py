#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_s07_h08_dimension_match import s07_h08_dimension_match


def test_s07_matches_h08_nonorthorhombic_complement() -> None:
    out = s07_h08_dimension_match()
    assert out["S07_total"] == 36
    assert out["S07_orthorhombic"] == 6
    assert out["S07_nonorthorhombic"] == 30
    assert out["H08_total"] == 45
    assert out["H08_orthorhombic"] == 15
    assert out["H08_nonorthorhombic"] == 30
    assert out["dimension_match"] is True
