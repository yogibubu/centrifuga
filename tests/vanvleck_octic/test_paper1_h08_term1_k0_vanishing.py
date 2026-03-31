#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term1_k0_vanishing import paper1_h08_term1_k0_vanishing


def test_naive_term1_k0_diagonal_vanishes() -> None:
    out = paper1_h08_term1_k0_vanishing()
    assert all(val == 0 for val in out["k0_diagonal_values"].values())

