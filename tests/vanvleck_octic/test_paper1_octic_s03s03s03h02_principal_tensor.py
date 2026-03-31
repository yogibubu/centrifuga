#!/usr/bin/env python3
from __future__ import annotations

import sympy as sp

from scripts.paper1_octic_s03s03s03h02_principal_tensor import (
    paper1_octic_s03s03s03h02_principal_tensor,
)


def test_s03s03s03h02_principal_tensor_is_degree8_and_orthorhombic() -> None:
    out = paper1_octic_s03s03s03h02_principal_tensor()
    nonzero = out["nonzero_orth_terms"]
    assert nonzero
    assert all(sum(mon) == 8 for mon in nonzero)
    assert all(all(exp % 2 == 0 for exp in mon) for mon in nonzero)


def test_s03s03s03h02_sigma3_matches_watson_s111_normalization() -> None:
    out = paper1_octic_s03s03s03h02_principal_tensor()
    sigma3, S111 = sp.symbols("sigma3 S111")
    assert out["sigma3_to_S111"] == sp.Eq(sigma3, -sp.Rational(3, 2) * S111)


def test_s03s03s03h02_has_multiple_orthorhombic_targets() -> None:
    out = paper1_octic_s03s03s03h02_principal_tensor()
    assert len(out["nonzero_orth_terms"]) > 1

