#!/usr/bin/env python3

from vanvleck_harmonic_rotor import build_h02


def test_build_h02_has_three_diagonal_rotor_terms():
    out = build_h02()
    assert len(out) == 3
