#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_octic_rprimek_consistency import (
    paper1_octic_rprimek_consistency_verdict,
)


def test_rprimek_cannot_carry_phi4_or_phi3() -> None:
    out = paper1_octic_rprimek_consistency_verdict()
    assert out["only_consistent_potential_support_for_R'_k"] == tuple()
    assert "phi4^2 from (R'_k)^2" in out["forbidden_if_R'_k_contains_phi4"]
    assert "phi3-linear from (R'_k)^2 or R'_k R_l [R_k,R_l]" in out["forbidden_if_R'_k_contains_phi3"]


def test_rprimek_must_live_in_mu1_algebra() -> None:
    out = paper1_octic_rprimek_consistency_verdict()
    assert out["consistent_rotational_support_for_R'_k"] == ("mu1",)
