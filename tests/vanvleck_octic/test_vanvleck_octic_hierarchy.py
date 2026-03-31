#!/usr/bin/env python3

from __future__ import annotations

import sympy as sp

import vanvleck_octic_hierarchy as voh


def test_printed_octic_hierarchy_matches_eq99_structure():
    h = voh.printed_octic_hierarchy()
    assert h.term0 == h.H08
    assert h.term1 == -sp.I / 6 * voh.comm(h.S03, voh.comm(h.S03, voh.comm(h.S03, h.H02)))
    assert h.term2 == -sp.Rational(1, 2) * voh.comm(h.S03, voh.comm(h.S03, h.H04))
    assert h.term3 == sp.I * voh.comm(h.S03, h.H06)
    assert h.term4 == sp.I * voh.comm(h.S05, h.H04)
    assert h.term5 == -voh.comm(h.S05, voh.comm(h.S03, h.H02))
    assert h.term6 == sp.I * voh.comm(h.S07, h.H02)
    assert sp.simplify(h.tilde_H08 - (h.term0 + h.term1 + h.term2 + h.term3 + h.term4 + h.term5 + h.term6)) == 0


def test_hierarchy_coefficients_are_exact():
    coeff = voh.hierarchy_coefficients()
    assert coeff["H08"] == 1
    assert coeff["S03S03S03H02"] == -sp.I / 6
    assert coeff["S03S03H04"] == -sp.Rational(1, 2)
    assert coeff["S03H06"] == sp.I
    assert coeff["S05H04"] == sp.I
    assert coeff["S05S03H02"] == -1
    assert coeff["S07H02"] == sp.I
