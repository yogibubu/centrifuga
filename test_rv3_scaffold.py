from __future__ import annotations

from pathlib import Path

import pytest

from RV3 import RV3Request, RV3Source, run_rv3


REPO = Path(__file__).resolve().parent


def test_rv3_rejects_order2_without_hessian() -> None:
    request = RV3Request(
        max_derivative_order=2,
        geometry=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
    )
    with pytest.raises(ValueError, match="Hessian source"):
        run_rv3(request)


def test_rv3_order2_h2o_fchk_smoke() -> None:
    request = RV3Request(
        max_derivative_order=2,
        geometry=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        hessian=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        representation="I",
    )
    result = run_rv3(request)
    assert result.geometry_stage.point_group == "C2v"
    assert result.harmonic_stage is not None
    assert result.harmonic_stage.n_modes == 3
    assert "DJ" in result.harmonic_stage.watson_s_mhz


def test_rv3_order3_h2o_log_fchk_smoke() -> None:
    request = RV3Request(
        max_derivative_order=3,
        geometry=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        hessian=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        cubic=RV3Source(str(REPO / "gaussian" / "h2o.log")),
        representation="I",
    )
    result = run_rv3(request)
    assert result.harmonic_stage is not None
    assert result.cubic_stage is not None
    assert result.cubic_stage.phi3_reduced_shape == [3, 3, 3]
    assert len(result.cubic_stage.mode_reorder_map) == 3
