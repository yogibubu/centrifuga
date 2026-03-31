#!/usr/bin/env python3

from __future__ import annotations

from tools.vanvleck import derive_first_active_octic_sector as fao


def test_first_active_octic_sector_is_h12_two_mode_diagonal():
    out = fao.derive_first_active_octic_sector()
    assert out["sector"] == "H12_only_two_mode_diagonal"
    assert out["first_active_order"] == 4
    assert out["octic_count_order4"] == 36
    assert out["quartic_count_order2"] == 9
    assert out["octic_commuting_order4"] == 0
    assert out["octic_by_order"][1] == {}
    assert out["octic_by_order"][2] == {}
    assert out["octic_by_order"][3] == {}
    assert out["octic_by_order"][4]
