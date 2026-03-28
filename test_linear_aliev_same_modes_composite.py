from linear_dv_aliev_terms import build_explicit_aliev_L_model, build_explicit_aliev_dv_general_model, make_linear_aliev_explicit_inputs_from_mapping
from scripts.build_linear_aliev_payload_from_gaussian import build_payload_same_modes_approx
from scripts.nno_composite_report import build_nno_composite_report


def test_nno_same_modes_composite_payload_is_finite_and_audited() -> None:
    payload = build_payload_same_modes_approx(
        harmonic_fchk_path="/Users/vincenzobarone/centrifugal/nno_DPCS3.fchk",
        anharmonic_log_path="/Users/vincenzobarone/centrifugal/nno_HPCS2.log",
        anharmonic_fchk_path="/Users/vincenzobarone/centrifugal/nno_HPCS2.fchk",
        force_constant_source="raw_au_reconverted",
        align_permutation_and_sign=True,
    )
    audit = payload["metadata"]["overlap_audit"]
    assert audit is not None
    assert audit["min_abs_overlap"] > 0.95
    inputs = make_linear_aliev_explicit_inputs_from_mapping(payload, quartic_mode="reduced")
    dv = build_explicit_aliev_dv_general_model(inputs)
    l_model = build_explicit_aliev_L_model(inputs)
    assert float(dv.value_for_state((0, 0, 0))) != 0.0
    assert float(l_model.value) != 0.0


def test_nno_same_modes_composite_report_contains_dual_level_h() -> None:
    report = build_nno_composite_report(
        "/Users/vincenzobarone/centrifugal/nno_HPCS2.fchk",
        "/Users/vincenzobarone/centrifugal/nno_HPCS2.log",
        "/Users/vincenzobarone/centrifugal/nno_DPCS3.fchk",
        "/Users/vincenzobarone/centrifugal/nno_DPCS3.log",
    )
    assert "comp : `H =" in report


def test_hc3n_same_modes_composite_report_contains_dual_level_h() -> None:
    report = build_nno_composite_report(
        "/Users/vincenzobarone/centrifugal/hc3n_HPCS2.fchk",
        "/Users/vincenzobarone/centrifugal/hc3n_HPCS2.log",
        "/Users/vincenzobarone/centrifugal/hc3n_DPCS3.fchk",
        "/Users/vincenzobarone/centrifugal/hc3n_DPCS3.log",
    )
    assert "comp : `H =" in report
