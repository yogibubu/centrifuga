#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_octic_s05s03h02_channel import paper1_octic_s05s03h02_channel


def test_s05s03h02_is_direct_phi3_phi3_phi4_channel() -> None:
    out = paper1_octic_s05s03h02_channel()
    assert out["source_block"] == "[S05,[S03,H02]]"
    assert out["principal_degree"] == 8
    assert out["survives_directly_in_orth_coefficients"] is True
    assert out["acts_as_elimination_only"] is False
    assert out["rotational_derivative_content"] == tuple()
    assert out["potential_content"] == ("phi3", "phi3", "phi4")
    assert "phi3^2*phi4" in out["channel_statement"]
