#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_h04_principal_map import paper1_h04_principal_map


def test_h04_principal_map_diagonal_terms() -> None:
    out = paper1_h04_principal_map()
    q400, q040, q004 = sp.symbols("q400 q040 q004")
    tau_xxxx, tau_yyyy, tau_zzzz = sp.symbols("tau_xxxx tau_yyyy tau_zzzz")
    assert out["maps"]["q400"] == sp.Eq(q400, sp.Rational(1, 4) * tau_xxxx)
    assert out["maps"]["q040"] == sp.Eq(q040, sp.Rational(1, 4) * tau_yyyy)
    assert out["maps"]["q004"] == sp.Eq(q004, sp.Rational(1, 4) * tau_zzzz)


def test_h04_principal_map_cross_terms() -> None:
    out = paper1_h04_principal_map()
    q220, q202, q022 = sp.symbols("q220 q202 q022")
    tau_xxyy, tau_xxzz, tau_yyzz = sp.symbols("tau_xxyy tau_xxzz tau_yyzz")
    assert out["maps"]["q220"] == sp.Eq(q220, sp.Rational(3, 2) * tau_xxyy)
    assert out["maps"]["q202"] == sp.Eq(q202, sp.Rational(3, 2) * tau_xxzz)
    assert out["maps"]["q022"] == sp.Eq(q022, sp.Rational(3, 2) * tau_yyzz)
