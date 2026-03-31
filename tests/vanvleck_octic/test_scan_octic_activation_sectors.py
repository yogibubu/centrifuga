#!/usr/bin/env python3

from __future__ import annotations

import scan_octic_activation_sectors as scan


def test_default_scan_cases_are_ordered_from_minimal_to_nonminimal():
    cases = scan.default_scan_cases()
    assert tuple(case.name for case in cases) == (
        "1mode_diag",
        "1mode_fullrot",
        "2mode_diag",
        "2mode_fullrot",
    )


def test_minimal_case_is_inactive():
    out = scan.run_octic_activation_scan()
    first = out["cases"][0]
    assert first["name"] == "1mode_diag"
    assert first["octic_active"] is False
