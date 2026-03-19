from __future__ import annotations

from ceditt_gui import _h22_from_fchk


def test_c2h2_h22_from_fchk_uses_linear_special_limit() -> None:
    res = _h22_from_fchk("c2h2.fchk", "I", "A")
    assert "D" in res["h22_total_khz"]
    assert res["h22_total_khz"]["D"] == res["h22_total_khz"]["D"]
    assert res["d_diagnostic"]["D"] >= 0.0
    assert res["dn_diagnostic"]["D_N"] >= 0.0
