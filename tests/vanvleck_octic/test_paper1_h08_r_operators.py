#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_h08_r_operators import paper1_r_operator_definitions


def test_r_operator_block_contains_expected_keys() -> None:
    out = paper1_r_operator_definitions()
    for key in ("R_tilde_k", "R_k", "R_kl", "R_klm", "k'_klm", "k'_{klmn}"):
        assert key in out


def test_r_kl_and_r_klm_symmetries_are_recorded() -> None:
    out = paper1_r_operator_definitions()
    assert str(out["R_lk"].rhs) == str(out["R_kl"].lhs)
    assert str(out["R_mlk"].rhs) == str(out["R_klm"].lhs)
