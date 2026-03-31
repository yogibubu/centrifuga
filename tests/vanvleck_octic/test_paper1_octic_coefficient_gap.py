#!/usr/bin/env python3
from __future__ import annotations

from scripts.paper1_octic_coefficient_gap import paper1_h06_h08_coefficient_status


def test_h06_has_final_coefficients_but_h08_does_not() -> None:
    out = paper1_h06_h08_coefficient_status()
    assert out["H06_status"]["final_orthorhombic_coefficients_available"] is True
    assert out["H06_status"]["equations"] == ("96", "97")
    assert out["H08_status"]["final_orthorhombic_coefficients_available"] is False
    assert out["H08_status"]["equation"] == "99"
    assert "has not been obtained yet" in out["H08_status"]["paper1_statement"]
    assert "pre-reduction source line" in out["consequence"]
