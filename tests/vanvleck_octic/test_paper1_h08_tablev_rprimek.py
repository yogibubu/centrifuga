#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_h08_tablev_rprimek import (
    paper1_tablev_rprimek_definition,
    paper1_tablev_rprimek_support,
)


def test_rprimek_visible_definition_contains_only_r_rprimekl_h02() -> None:
    eq = paper1_tablev_rprimek_definition()
    rhs = str(eq.rhs)
    assert "R_l" in rhs
    assert "R'_{kl}" in rhs
    assert "[R_k,H02]" in rhs


def test_rprimek_support_is_mu1_phi4_zeta_omega_b() -> None:
    out = paper1_tablev_rprimek_support()
    assert out["expanded"] == ("mu1", "phi4", "zeta", "omega", "B")
