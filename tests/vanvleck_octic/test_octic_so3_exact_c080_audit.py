#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.octic_so3_exact_c080_audit import exact_c080_audit


def test_exact_c080_vanishes_for_all_commutator_channels() -> None:
    out = exact_c080_audit()["c080_exact"]
    for key in (
        "S03S03S03H02",
        "S03S03H04",
        "S03H06",
        "S05H04",
        "S05S03H02",
    ):
        assert sp.expand(out[key]) == 0
