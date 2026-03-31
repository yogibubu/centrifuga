#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_octic_s03h06_channel import paper1_octic_s03h06_channel


def test_s03h06_is_direct_mu1_phi3_squared_channel() -> None:
    out = paper1_octic_s03h06_channel()
    assert out["source_block"] == "[S03,H06]"
    assert out["principal_degree"] == 8
    assert out["survives_directly_in_orth_coefficients"] is True
    assert out["acts_as_elimination_only"] is False
    assert out["rotational_derivative_content"] == ("mu1",)
    assert out["potential_content"] == ("phi3", "phi3")
    assert "mu1*phi3^2" in out["channel_statement"]
