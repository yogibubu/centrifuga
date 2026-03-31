#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_rklm_linear_geometric_form import paper1_h08_rklm_linear_geometric_form


def test_rklm_linear_geometric_prefactor() -> None:
    out = paper1_h08_rklm_linear_geometric_form()
    assert sp.sstr(out["coefficient"]) == "-C_k*C_l*C_m*omega_k*omega_l*omega_m/(2*B**2)"

