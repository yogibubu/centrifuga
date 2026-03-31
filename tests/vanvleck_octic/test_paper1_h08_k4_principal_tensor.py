#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_h08_k4_principal_tensor import paper1_h08_k4_principal_term


def test_h08_k4_principal_term_is_mu1_phi4_only_and_omega_free() -> None:
    out = paper1_h08_k4_principal_term()
    assert out["omega_cancellation"] is True
    assert out["rotational_support"] == ("mu1",)
    assert out["potential_support"] == ("phi4",)
    assert out["harmonic_support"] == tuple()
    assert "quartic force-constant tensor" in out["statement"]
