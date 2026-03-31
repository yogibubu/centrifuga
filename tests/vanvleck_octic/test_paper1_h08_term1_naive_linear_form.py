#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term1_naive_linear_form import paper1_h08_term1_naive_linear_form


def test_term1_naive_linear_is_not_scalar_x4() -> None:
    out = paper1_h08_term1_naive_linear_form()
    assert "X**3" in sp.sstr(out["term1_naive_linear"])
    assert "J_y^3+J_z^3" in sp.sstr(out["term1_naive_linear"])

