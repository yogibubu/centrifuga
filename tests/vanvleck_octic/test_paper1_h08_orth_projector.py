#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_h08_orth_projector import (
    degree8_selection_projectors,
    principal_symbol_reduction_summary,
)


def test_degree8_selection_projectors_have_expected_shapes() -> None:
    out = degree8_selection_projectors()
    assert out["shapes"]["P_orth"] == (15, 45)
    assert out["shapes"]["P_nonorth"] == (30, 45)
    assert len(out["orth_basis"]) == 15
    assert len(out["nonorth_basis"]) == 30


def test_principal_symbol_reduction_statement_dimensions() -> None:
    out = principal_symbol_reduction_summary()
    assert out["full_dimension"] == 45
    assert out["orth_dimension"] == 15
    assert out["nonorth_dimension"] == 30
    assert "direct orth projection" in out["principal_symbol_statement"]
