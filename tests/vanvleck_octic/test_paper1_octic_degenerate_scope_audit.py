#!/usr/bin/env python3

from scripts.paper1_octic_degenerate_scope_audit import paper1_octic_degenerate_scope_audit


def test_scope_audit_marks_qh_as_outside_scalar_branch() -> None:
    out = paper1_octic_degenerate_scope_audit()
    assert out["current_derived_branch"] == "pure_rotational_scalar_only"
    assert out["missing_labels"] == ("q_e", "q_J", "q_H")
