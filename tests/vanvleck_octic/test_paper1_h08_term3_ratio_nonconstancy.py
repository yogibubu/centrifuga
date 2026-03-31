#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_ratio_nonconstancy import paper1_h08_term3_ratio_nonconstancy


def test_ratio_is_generically_not_constant() -> None:
    out = paper1_h08_term3_ratio_nonconstancy()
    assert out["dratio_dC"] != 0
    assert out["dratio_dw"] != 0
    assert out["dratio_no_cubic_dC"] != 0
    assert out["dratio_no_cubic_dw"] != 0

