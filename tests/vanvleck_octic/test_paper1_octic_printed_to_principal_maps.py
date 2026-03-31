#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_printed_to_principal_maps import paper1_printed_to_principal_maps


def test_printed_to_principal_sigma3_map() -> None:
    out = paper1_printed_to_principal_maps()
    sigma3, S111 = sp.symbols("sigma3 S111")
    assert out["sigma3_map"] == sp.Eq(sigma3, -sp.Rational(3, 2) * S111)


def test_printed_to_principal_s05_maps() -> None:
    out = paper1_printed_to_principal_maps()
    s311, s131, s113 = sp.symbols("s311 s131 s113")
    S311, S131, S113 = sp.symbols("S311 S131 S113")
    assert out["s05_maps"]["s311"] == sp.Eq(s311, -2 * S311)
    assert out["s05_maps"]["s131"] == sp.Eq(s131, -2 * S131)
    assert out["s05_maps"]["s113"] == sp.Eq(s113, -2 * S113)


def test_printed_to_principal_h06_maps() -> None:
    out = paper1_printed_to_principal_maps()
    w420, Wxxy = sp.symbols("w420 Wxxy")
    w600, Wxxx = sp.symbols("w600 Wxxx")
    assert out["h06_maps"]["w420"] == sp.Eq(w420, 2 * Wxxy)
    assert out["h06_maps"]["w600"] == sp.Eq(w600, Wxxx)

