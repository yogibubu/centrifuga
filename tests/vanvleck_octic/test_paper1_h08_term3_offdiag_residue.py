#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys

import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "scripts"))

from paper1_h08_term3_offdiag_residue import paper1_h08_term3_offdiag_residue


def test_offdiag_residue_is_exact_missing_piece() -> None:
    out = paper1_h08_term3_offdiag_residue()
    assert sp.simplify(out["target_S_n"] - out["diag_S_n"] - out["offdiag_residue"]) == 0
