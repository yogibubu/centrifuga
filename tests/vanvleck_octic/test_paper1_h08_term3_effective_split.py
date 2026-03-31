#!/usr/bin/env python3

import sympy as sp

from scripts.paper1_h08_term3_effective_split import paper1_h08_term3_effective_split


def test_term3_quotient_splits_into_x0_plus_lower_orders() -> None:
    summary = paper1_h08_term3_effective_split()
    assert summary["difference"] == 0
    assert summary["split"]["a_x0"] == sp.Rational(-63, 256)
