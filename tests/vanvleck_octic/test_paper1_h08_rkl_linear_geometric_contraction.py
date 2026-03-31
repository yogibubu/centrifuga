#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_rkl_linear_geometric_contraction import (
    paper1_h08_rkl_linear_geometric_contraction,
)


def test_rkl_scalar_geometric_factor() -> None:
    out = paper1_h08_rkl_linear_geometric_contraction()
    assert sp.sstr(out["rlm_scalar_geom"]) == "3*C_l*C_m*omega_l*omega_m/(4*B)"
