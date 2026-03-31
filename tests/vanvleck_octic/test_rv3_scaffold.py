from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaussian_vpt_parser import parse_gaussian_fchk_harmonic_data
from RV3 import (
    RV3ManualQuarticRequest,
    RV3ManualSexticRequest,
    RV3Request,
    RV3Source,
    build_rv3_integrated_report,
    build_rv3_summary_csv,
    export_rv3_outputs,
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


def test_rv3_order2_h2o_xyz_hessian_smoke(tmp_path: Path) -> None:
    fchk = parse_gaussian_fchk_harmonic_data(REPO / "gaussian" / "h2o.fchk")
    xyz_path = tmp_path / "h2o.xyz"
    xyz_lines = [str(len(fchk.atomic_numbers)), "h2o from fchk"]
    symbols = ["H" if z == 1 else "O" for z in fchk.atomic_numbers]
    coords_ang = fchk.coords_bohr / 1.8897261254578281
    for sym, row in zip(symbols, coords_ang):
        xyz_lines.append(f"{sym} {row[0]:.12f} {row[1]:.12f} {row[2]:.12f}")
    xyz_path.write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")
    hessian_path = tmp_path / "h2o.hess"
    flat = " ".join(f"{float(x):.16e}" for x in fchk.cartesian_force_constants.reshape(-1))
    hessian_path.write_text(flat + "\n", encoding="utf-8")
    request = RV3Request(
        max_derivative_order=2,
        geometry=RV3Source(str(xyz_path)),
        hessian=RV3Source(str(hessian_path)),
        representation="I",
    )
    result = run_rv3(request)
    assert result.geometry_stage.point_group == "C2v"
    assert result.harmonic_stage is not None
    assert result.harmonic_stage.n_modes == 3


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


def test_rv3_order3_h2o_alpha_routes_smoke() -> None:
    request = RV3Request(
        max_derivative_order=3,
        geometry=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        hessian=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        cubic=RV3Source(str(REPO / "gaussian" / "h2o.log")),
        alpha_excluded_modes=(1,),
        representation="I",
    )
    result = run_rv3(request)
    assert result.alpha_parser_stage is not None
    assert result.alpha_internal_stage is not None
    assert result.alpha_parser_stage.excluded_modes == [1]
    assert result.alpha_internal_stage.excluded_modes == [1]
    assert len(result.alpha_parser_stage.total_alpha_mhz) == 3
    assert len(result.alpha_internal_stage.alpha_total_sum_mhz) == 3
    assert result.alpha_internal_stage.cubic_matrix_origin == "derived_from_cubic_log"
    text = json.dumps(result.to_jsonable(), allow_nan=False)
    assert "NaN" not in text


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


def test_rv3_integrated_report_smoke() -> None:
    request = RV3Request(
        max_derivative_order=4,
        geometry=RV3Source(str(REPO / "hcn.fchk")),
        hessian=RV3Source(str(REPO / "hcn.fchk")),
        cubic=RV3Source(str(REPO / "hcn.log")),
        quartic=RV3Source(str(REPO / "hcn.log")),
        representation="I",
    )
    result = run_rv3(request)
    report = build_rv3_integrated_report(result)
    assert "[Standard Quartics]" in report
    assert "[Alpha From Harmonic + Cubic]" in report
    assert "[Sextic H22 Linear Diagnostic]" in report
    assert "[Linear Order-4]" in report


def test_rv3_exports_smoke(tmp_path: Path) -> None:
    request = RV3Request(
        max_derivative_order=3,
        geometry=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        hessian=RV3Source(str(REPO / "gaussian" / "h2o.fchk")),
        cubic=RV3Source(str(REPO / "gaussian" / "h2o.log")),
        representation="I",
    )
    result = run_rv3(request)
    report_text = build_rv3_integrated_report(result)
    csv_text = build_rv3_summary_csv(result)
    assert "Integrated Vibro-Rotational Analysis" in report_text
    assert "field,value" in csv_text
    json_path = tmp_path / "rv3.json"
    report_path = tmp_path / "rv3.txt"
    csv_path = tmp_path / "rv3.csv"
    export_rv3_outputs(result, json_path=json_path, report_path=report_path, csv_path=csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["geometry_stage"]["point_group"] == "C2v"
    assert "Integrated Vibro-Rotational Analysis" in report_path.read_text(encoding="utf-8")
    assert "field,value" in csv_path.read_text(encoding="utf-8")


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
    assert result.handedness_flip["constants"] == [1.0, 2.0, 3.0, -4.0, -5.0]
    text = json.dumps(result.to_jsonable(), allow_nan=False)
    assert "NaN" not in text


def test_rv3_manual_quartic_transform_handedness_flip_s_reduction() -> None:
    result = run_manual_quartic_transform(
        RV3ManualQuarticRequest(
            A_mhz=10000.0,
            B_mhz=5000.0,
            C_mhz=3000.0,
            rep_in="II",
            reduction="S",
            constants=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
    )
    assert result.handedness_flip["constants"] == [1.0, 2.0, 3.0, -4.0, 5.0]


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
    assert result.handedness_flip["constants"] == [1.0, 2.0, 3.0, 4.0, -5.0, 0.5, -0.25]
    assert result.handedness_flip["roundtrip_max_error"] < 1.0e-12
    text = json.dumps(result.to_jsonable(), allow_nan=False)
    assert "NaN" not in text
