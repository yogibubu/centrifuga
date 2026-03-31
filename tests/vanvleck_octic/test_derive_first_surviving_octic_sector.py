#!/usr/bin/env python3

from derive_first_surviving_octic_sector import derive_first_surviving_octic_sector


def test_first_surviving_scan_classifies_known_zero_h12_channel():
    report = derive_first_surviving_octic_sector()
    analyses = {entry["name"]: entry for entry in report["analyses"]}
    h12 = analyses["H12_only"]
    assert h12["first_ordered_order"] == 4
    assert h12["first_commuting_order"] is None


def test_first_surviving_scan_returns_consistent_first_nonzero_or_none():
    report = derive_first_surviving_octic_sector()
    first = report["first_surviving"]
    if first is None:
        for entry in report["analyses"]:
            assert entry["first_commuting_order"] is None
    else:
        order = first["first_commuting_order"]
        assert order in (1, 2, 3, 4)
        assert first["commuting_by_order"][order] != 0
