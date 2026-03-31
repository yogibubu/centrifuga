#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_s05h04_principal_tensor import paper1_octic_s05h04_principal_tensor


def test_s05h04_principal_tensor_is_degree8_and_orthorhombic() -> None:
    out = paper1_octic_s05h04_principal_tensor()
    nonzero = out["nonzero_orth_terms"]
    assert nonzero
    assert all(sum(mon) == 8 for mon in nonzero)
    assert all(all(exp % 2 == 0 for exp in mon) for mon in nonzero)


def test_s05h04_depends_on_s05_and_h04_bases() -> None:
    out = paper1_octic_s05h04_principal_tensor()
    coeffs = tuple(out["orth_coeffs"].values())
    assert any(sp.Symbol("s311") in coeff.free_symbols for coeff in coeffs)
    assert any(sp.Symbol("q400") in coeff.free_symbols for coeff in coeffs)
    assert any(sp.Symbol("q022") in coeff.free_symbols for coeff in coeffs)


def test_s05h04_has_multiple_orthorhombic_targets() -> None:
    out = paper1_octic_s05h04_principal_tensor()
    assert len(out["nonzero_orth_terms"]) > 1

