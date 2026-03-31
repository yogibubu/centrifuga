#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_octic_s05h04_channel import paper1_octic_s05h04_channel


def test_s05h04_is_direct_mu1_phi3_phi4_channel() -> None:
    out = paper1_octic_s05h04_channel()
    assert out["source_block"] == "[S05,H04]"
    assert out["principal_degree"] == 8
    assert out["survives_directly_in_orth_coefficients"] is True
    assert out["acts_as_elimination_only"] is False
    assert out["rotational_derivative_content"] == ("mu1",)
    assert out["potential_content"] == ("phi3", "phi4")
    assert "mu1*phi3*phi4" in out["channel_statement"]
