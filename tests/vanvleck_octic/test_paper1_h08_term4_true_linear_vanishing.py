#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term4_true_linear_vanishing import paper1_h08_term4_true_linear_vanishing


def test_term4_vanishes_in_true_linear_operator_limit() -> None:
    out = paper1_h08_term4_true_linear_vanishing()
    assert out["commutator"] == 0
    assert out["term4_linear"] == 0

