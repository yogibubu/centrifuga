from __future__ import annotations

import json
from pathlib import Path

import pytest

from RV3 import (
    RV3ManualQuarticRequest,
    RV3ManualSexticRequest,
    RV3Request,
    RV3Source,
    run_manual_quartic_transform,
    run_manual_sextic_transform,
    run_rv3,
)


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


def test_rv3_order4_hcn_linear_branch_smoke() -> None:
    request = RV3Request(
        max_derivative_order=4,
        geometry=RV3Source(str(REPO / "hcn.fchk")),
        hessian=RV3Source(str(REPO / "hcn.fchk")),
        cubic=RV3Source(str(REPO / "hcn.log")),
        quartic=RV3Source(str(REPO / "hcn.log")),
        representation="I",
    )
    result = run_rv3(request)
    assert result.quartic_stage is not None
    assert result.quartic_stage.linear_branch_status == "linear_order4_general_branch_live"
    assert result.quartic_stage.linear_general_observable is not None
    assert result.quartic_stage.linear_legacy_compact_observable is not None
    assert result.quartic_stage.linear_compactness_audit is not None
    assert result.quartic_stage.linear_optical_constant is not None
    text = json.dumps(result.to_jsonable(), allow_nan=False)
    assert "NaN" not in text


def test_rv3_manual_quartic_transform_smoke() -> None:
    result = run_manual_quartic_transform(
        RV3ManualQuarticRequest(
            A_mhz=10000.0,
            B_mhz=5000.0,
            C_mhz=3000.0,
            rep_in="I",
            reduction="A",
            constants=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
    )
    assert set(result.outputs) == {"II", "III"}
    assert all("tensor_roundtrip_max_error" in payload for payload in result.outputs.values())
    text = json.dumps(result.to_jsonable(), allow_nan=False)
    assert "NaN" not in text


def test_rv3_manual_sextic_transform_smoke() -> None:
    result = run_manual_sextic_transform(
        RV3ManualSexticRequest(
            A_mhz=10000.0,
            B_mhz=5000.0,
            C_mhz=3000.0,
            rep_in="I",
            reduction_in="S",
            reduction_out="S",
            constants=[1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 0.25],
        )
    )
    assert set(result.outputs) == {"II", "III"}
    assert all("roundtrip_max_error" in payload for payload in result.outputs.values())
    text = json.dumps(result.to_jsonable(), allow_nan=False)
    assert "NaN" not in text
